"""Fasttext Core"""

from typing import Callable

import os
import random
import re
import numpy as np
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
import time
from datetime import datetime, timedelta

from difflib import SequenceMatcher
import fasttext
import hashlib

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Flatten, Dense, Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from janome.tokenizer import Tokenizer as JanomeTokenizer

from .utils import path
from .logger import Logger, LoggerConfig
from .file_container import RecordContainer
from .datatype.workflow import WorkflowConfig, CacheMode
from .datatype.record import RecordType, RecordMG


EPOCHS = 30
MARGIN = 10
THRESHOLD = 5


@tf.function()
def euclidean_distance(vects: "list[np.ndarray]") -> np.ndarray:
    """
    2つのベクトルからユークリッド距離を計算する (method for tensorflow)

    - - -

    Parameters
    ----------
    vects: list[np.ndarray]
        2つのベクトルが格納されたリスト

    Return
    ------
    np.ndarray(?)
        numpy形式の数値型
    """
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)

    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    """(深くはわからないがおそらく成形用メソッド)"""
    shape1, _ = shapes
    return (shape1[0], 1)


def create_model(input_shape):
    """kerasによるモデルを生成する"""

    input = Input(shape=input_shape)
    layer = Flatten()(input)
    layer = Dense(4096, activation="relu")(layer)
    # layer = Dropout(0.005)(layer)
    layer = Dense(2048, activation="relu")(layer)

    return Model(input, layer)


class FasttextCore:
    """距離学習の実行やfasttextに関連する処理を担うクラス"""

    # 並列処理実行時にGPU等を使用するモジュールは、インスタンスの複製に問題が発生するためクラス変数で格納する
    fasttext_model: "fasttext.FastText._FastText" = None  # fasttextが格納されたモデル
    vector_model: Model = None  # fasttextベクターを入力として別のベクトルに推論するモデル
    metric_model: Model = None  # vector_modelに加えて距離学習まで含めた学習モデル
    janome_tokenizer = JanomeTokenizer()  # Janomeのトークナイザー

    def __init__(
        self,
        config: WorkflowConfig = None,
        target_filepath: str = None,
        log_filepath: str = None,
        inf_attr: "dict[str, RecordType]" = {},
    ):
        """コンストラクタ"""
        self.inf_attr = inf_attr  # 推論に用いる属性名
        self.config = config
        self.target_filepath = target_filepath
        self.log_filepath = log_filepath
        self.logger: Logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level),
            filepath=self.log_filepath,
        )

        # tensorflowがメモリを専有せず、他のプロセスとも共有できるように設定
        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

    def __get_metric_model(self) -> Model:
        """
        fasttextで距離学習を行うモデルを構築する

        - - -

        Return
        ------
        Model
            tensorflow形式のModelインスタンス
        """

        input_shape = (300,)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        FasttextCore.vector_model = create_model(input_shape)
        processed_a = FasttextCore.vector_model(input_a)
        processed_b = FasttextCore.vector_model(input_b)

        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        return Model([input_a, input_b], distance)

    def compute_accuracy(self, y_true, y_pred) -> float:
        """精度を計算する"""

        __pred = np.where(y_pred <= THRESHOLD, 1, 0).ravel()
        __true = np.where(y_true <= THRESHOLD, 1, 0).ravel()

        __match = np.sort(np.array(y_pred[np.where(__true == 1)]).ravel())
        __mismatch = np.sort(np.array(y_pred[np.where(__true == 0)]).ravel())

        # MatchとMismatchの最大値と最小値を表示
        self.logger.info(f"[Match] Len: {len(__match)}, Min: {np.min(__match)}, Max: {np.max(__match)}")
        self.logger.info(
            f"""
[Match]
    50%: {__match[int(len(__match) * 0.5)]}
    80%: {__match[int(len(__match) * 0.8)]}
    90%: {__match[int(len(__match) * 0.9)]}
    95%: {__match[int(len(__match) * 0.95)]}
    99%: {__match[int(len(__match) * 0.99)]}
  99.9%: {__match[int(len(__match) * 0.999)]}"""
        )
        self.logger.info(f"[Mismatch] Len: {len(__mismatch)}, Min: {np.min(__mismatch)}, Max: {np.max(__mismatch)}")
        self.logger.info(
            f"""
[Mismatch]
   0.1%: {__mismatch[int(len(__mismatch) * 0.001)]}
     1%: {__mismatch[int(len(__mismatch) * 0.01)]}
     5%: {__mismatch[int(len(__mismatch) * 0.05)]}
    10%: {__mismatch[int(len(__mismatch) * 0.1)]}
    20%: {__mismatch[int(len(__mismatch) * 0.2)]}
    50%: {__mismatch[int(len(__mismatch) * 0.5)]}"""
        )
        return np.mean(__pred == __true)

    def contrastive_loss(self, y_true: float, y_pred: any) -> float:
        """
        損失関数を計算する

        - - -

        Parameters
        ----------
        y_true: Tensor[float32]
            正誤を表す値 (1...同書誌, 0...異書誌)
        y_pred: Tensor[float32]
            距離が格納されたベクトルの行列


        Return
        ------
        float
            平均値
        """

        positive_pred = tf.square(tf.where(y_pred < THRESHOLD, y_true - y_pred, tf.math.pow(y_true - y_pred, 2)))
        negative_pred = tf.square((y_true - y_pred) / 10)

        # y_trueを1未満であれば0、1以上であれば1に変換し、さらにfloat32型にキャスト
        y_true_casted = tf.cast(tf.where(y_true <= 1, 0, 1), tf.float32)

        return tf.reduce_mean(y_true_casted * positive_pred + (1 - y_true_casted) * negative_pred)

    def __separate_keyword(self, text: str, record_type) -> set[str]:
        """Janomeでキーワードを分割する"""
        keyword = set([])

        if record_type == RecordType.TEXT_EN or record_type == RecordType.COMPLEMENT_ASCII:
            # 英語の場合は、空白で分割する
            for token in text.split(" "):
                keyword.add(token)

        else:
            # 日本語を含むその他の言語は、Janomeで形態素解析し分割する

            for token in FasttextCore.janome_tokenizer.tokenize(text):
                splitter = token.part_of_speech.split(",")
                # 空白は除外
                if splitter[0] in ["記号"] and splitter[1] in ["空白"]:
                    continue

                keyword.add(token.surface)

        return keyword

    def __get_fasttext_dist(self, text: str, record_type: RecordType) -> np.ndarray:
        """
        fasttextからのベクトル(分散表現)の取得

        - - -

        Parameters
        ----------
        text: str
            得たいベクトルの文字列

        Return
        ------
        np.ndarray
            要素数300で構成されるndarray
        """
        # 属性値を形態素解析して各々のベクトルを足し合わせる
        result = np.zeros(300)
        target = self.__separate_keyword(text, record_type)

        for t in target:
            # 和をとる
            result += np.array(FasttextCore.fasttext_model[t])

        # 属性値全体のベクトルを追加
        result += np.array(FasttextCore.fasttext_model[text])

        # 正規化
        if np.all(result != 0):
            result = result / np.linalg.norm(result)

        return result

    def __get_onehot_vector(self, text: str, hash: bool = True) -> np.ndarray:
        """
        年月からワンホットベクトルの取得 (下処理は入力規則を考慮する必要あり)

        - - -

        Parameters
        ----------
        text: str
            得たいベクトルの文字列
        hash: bool
            元のデータを数値のみでハッシュ化しベクトル化する, by default True

        Return
        ------
        np.ndarray
            要素数300で構成されるndarray
        """

        tmp = re.sub("[^0-9.]", "", text)

        if hash:
            # ハッシュ化を行う
            date = self.hash_number_for_date(tmp, 10)
        else:
            # 入力規則を考慮する
            # スラッシュまたはピリオドでセパレートされていた月を取得できた場合、月が1桁の末尾には0を追加する
            if "/" in tmp:
                date_list: "list[str]" = tmp.split("/")
            else:
                date_list: "list[str]" = tmp.split(".")

            if len(date_list) == 2:
                date_list[1] = "0" + date_list[1] if len(date_list[1]) == 1 else date_list[1]

            date = "".join(date_list)

        date_array = np.array([])

        for d in date:
            date_array = np.append(date_array, np.array(np.eye(10)[int(d)]))

        date_array = np.append(date_array, np.zeros(300 - len(date) * 10))

        # 正規化する
        date_array = date_array / np.linalg.norm(date_array)

        return date_array

    def construct_vector(self, record: RecordMG, counter: "Synchronized[int]" = None):
        """
        レコードからFasttextベクトルまたはOnehotベクトルからなる分散表現を構築する

        - - -

        Params
        ------
        record: RecordMG
            レコード
        """

        # 300次元ベクトルを0埋めで初期化
        tmp = np.zeros(300)

        for arg, record_type in self.inf_attr.items():
            # TEXT_KEYの場合は、fasttextベクトルを利用し、大きさを3倍する
            if record_type == RecordType.TEXT_KEY:
                tmp += self.__get_fasttext_dist(record.re.data[arg]) * 3

            # 日付データの場合は、Onehot vectorを用いる
            elif record_type == RecordType.COMPLEMENT_DATE:
                tmp += self.__get_onehot_vector(record.re.data[arg])

            # それ以外の場合は、fasttextベクトルを用いる
            else:
                tmp += self.__get_fasttext_dist(record.re.data[arg], record_type)

        # カウンターがあれば、カウンターを追加する
        if counter is not None:
            counter.value += 1

        return np.reshape(tmp, (300,))

    def construct_string(self, record: RecordMG):
        """
        レコードから結合文字列を構築する

        - - -

        Params
        ------
        record: RecordMG
            レコード
        """

        result = ""
        for arg, type in self.inf_attr.items():
            # 日付データの場合は、ハッシュ化する
            if type == RecordType.COMPLEMENT_DATE:
                result += self.hash_number_for_date(record.re.data[arg])
            # それ以外の場合は、通常通り結合する
            else:
                result += record.re.data[arg]

        return result

    def hash_number_for_date(self, text: str, str_slice: int = 0) -> str:
        """
        与えられた日付テキストに対して、下処理を行い数値のハッシュ値を返す

        - - -

        Parameters
        ----------
        text: str
            ハッシュ化する文字列
        str_slice: int
            ハッシュした文字の文字数制限, by default 0
        """

        # 空文字列の場合、300000以上の6桁の適当な数値文字列を代入する
        if text == "":
            text = "{}".format(random.randrange(300000, 1000000))

        # 下処理 (＊入力規則を考慮する)
        date_list: "list[str]" = text.split(".")
        if len(date_list) == 2:
            date_list[1] = ("0" + date_list[1])[1:]
        date = "".join(date_list)

        # 数値のみのハッシュを取得
        result = re.sub(r"\D", "", hashlib.sha256(date.encode()).hexdigest())

        if str_slice != 0:
            result = result[:str_slice]

        return result

    def load_crowdsourcing_pair(
        self,
        labeling_function: "Callable[[RecordMG, RecordMG, int], float]" = None,
    ) -> "tuple[list[RecordMG], list[RecordMG], list[int], list[int]]":
        """クラウドソーシングペアの一致不一致をRecordMGと件数に変換して返す"""

        re = RecordContainer(log_filepath=self.log_filepath)
        re.load_file(self.target_filepath)

        match_pairs: "list[RecordMG]" = []
        match_labels: "list[int | float]" = []
        mismatch_pairs: "list[RecordMG]" = []
        mismatch_labels: "list[int | float]" = []

        for key, value in self.config.crowdsourcing_result.items():
            if sum(value) / len(value) >= 0.5:
                match_pairs.append(RecordMG(re.records[key[0]], self.inf_attr))
                match_pairs.append(RecordMG(re.records[key[1]], self.inf_attr))

                # ラベリング関数があれば、それに従ってラベリングを行う
                if labeling_function is not None:
                    match_labels.append(labeling_function([match_pairs[-2], match_pairs[-1], 1]))
                else:
                    match_labels.append(1)

            else:
                mismatch_pairs.append(RecordMG(re.records[key[0]], self.inf_attr))
                mismatch_pairs.append(RecordMG(re.records[key[1]], self.inf_attr))

                # ラベリング関数があれば、それに従ってラベリングを行う
                if labeling_function is not None:
                    mismatch_labels.append(labeling_function([mismatch_pairs[-2], mismatch_pairs[-1], 0]))
                else:
                    mismatch_labels.append(0)

        return (match_pairs, mismatch_pairs, match_labels, mismatch_labels)

    def load_fasttext_dict(self, fasttext_path: str = "../cc.ja.300.bin"):
        """
        fasttext辞書をインスタンス変数に読み込む

        - - -

        Parameter
        ---------
        fasttext_path: str
            辞書ファイルのパス, by default '../cc.ja.300.bin
        """
        FasttextCore.fasttext_model = fasttext.load_model(fasttext_path)

    def difflib_driven_labeling(self, recordmg1: RecordMG, recordmg2: RecordMG, label: int) -> float:
        """
        difflibによるラベリングを行う

        - - -

        Parameters
        ----------
        recordmg1: RecordMG
            レコード1
        recordmg2: RecordMG
            レコード2
        label: int
            ラベル (1...同エンティティ, 0...異エンティティ)
        """

        # レコードの文字列を取得
        str1 = self.construct_string(recordmg1)
        str2 = self.construct_string(recordmg2)

        # difflibによるラベリング
        if label == 1:
            # 一致している場合は、1からSequenceMatcherの値を引いた値を返す
            return 1 - SequenceMatcher(None, str1, str2).ratio()
        else:
            # 一致していない場合は、1からSequenceMatcherの値を引き、その値に9をかけ、1を足した値を返す
            return (1 - SequenceMatcher(None, str1, str2).ratio()) * (MARGIN - 1) + 1

    def train(
        self,
        filepath: str,
        test_ratio: float = 0.1,
        data_shuffle: bool = False,
        match_num: int | None = None,
        mismatch_ratio: float = 1,
        max_in_cluster: int = 50,
        basemodel_filepath: "str | None" = None,
        image_dirpath: str = None,
        use_crowdsourcing: bool = True,
        use_best_model: bool = True,
    ):
        """
        シャムネットワークを構築し学習を行ったあと、インスタンス変数に格納する

        Parameter
        ---------
        filepath: str | None
            学習用データが格納されたymlファイルのパス
        test_ratio: float, by default 0.1
            テストデータの割合
        data_shuffle: bool, by default False
            データをシャッフルするか否か
        match_num: int, by default None

        max_in_cluster: int, by default 50
            クラスタ内の最大数
        basemodel_filepath: str | None, by default None
            学習済みの距離学習機のパラメータが格納されたh5ファイルパス
        use_crowdsourcing: bool, by default True
            クラウドソーシングの結果を利用するか否か
        use_best_model: bool, by default True
            最も良いモデルを利用するか否か
        """
        # TODO: 複数の学習データを読み込めるようにする

        # 学習用のデータを読み込む
        rc = RecordContainer(log_filepath=self.log_filepath)
        rc.load_file(filepath)
        rc_recordmg = rc.get_recordmg()
        (
            match_pairs,
            mismatch_pairs,
            match_labels,
            mismatch_labels,
        ) = rc.get_recordmg_for_train(
            match_num=match_num,
            mismatch_ratio=mismatch_ratio,
            max_in_cluster=max_in_cluster,
            labeling_function=self.difflib_driven_labeling,
        )

        # レコードのベクトル化とペアに基づいてベクトルを結合し行列を作成
        __record_array = self.get_fasttext_vectors_with_mp(CacheMode.WRITE, filepath, rc_recordmg)

        __record_list = []
        for i in range(len(match_pairs) // 2):
            __record_list.append(
                [
                    __record_array[match_pairs[i * 2].re.idx],
                    __record_array[match_pairs[i * 2 + 1].re.idx],
                ]
            )
        match_array = np.array(__record_list)

        __record_list = []
        for i in range(len(mismatch_pairs) // 2):
            __record_list.append(
                [
                    __record_array[mismatch_pairs[i * 2].re.idx],
                    __record_array[mismatch_pairs[i * 2 + 1].re.idx],
                ]
            )
        mismatch_array = np.array(__record_list)

        # クラウドソーシングの結果を反映する
        if use_crowdsourcing is True and len(self.config.crowdsourcing_result) > 0:
            # 学習用データに整形
            (
                match_pairs,
                mismatch_pairs,
                match_labels,
                mismatch_labels,
            ) = self.load_crowdsourcing_pair(labeling_function=self.difflib_driven_labeling)

            rc = RecordContainer(log_filepath=self.log_filepath)
            rc.load_file(self.target_filepath)
            rc_recordmg = rc.get_recordmg()
            __map, __mip, __mal, __mil = rc.get_crowdsourcing_recordmg_for_train(self.config.crowdsourcing_result)
            __record_array = self.get_fasttext_vectors_with_mp(CacheMode.WRITE, self.target_filepath, rc_recordmg)

            __record_list = []
            for i in range(len(__map) // 2):
                __record_list.append(
                    [
                        __record_array[__map[i * 2].re.idx],
                        __record_array[__map[i * 2 + 1].re.idx],
                    ]
                )
            match_array = np.concatenate([np.array(__record_list), match_array])

            __record_list = []
            for i in range(len(__mip) // 2):
                __record_list.append(
                    [
                        __record_array[__mip[i * 2].re.idx],
                        __record_array[__mip[i * 2 + 1].re.idx],
                    ]
                )
            mismatch_array = np.concatenate([np.array(__record_list), mismatch_array])

            # ラベル結合
            match_labels = __mal + match_labels
            mismatch_labels = __mil + mismatch_labels

        match_labels = np.array(match_labels)
        mismatch_labels = np.array(mismatch_labels)

        # ランダムにシャッフル
        if data_shuffle:
            shuffled_indices = np.random.permutation(len(match_array))
            match_array = match_array[shuffled_indices]
            match_labels = match_labels[shuffled_indices]
            shuffled_indices = np.random.permutation(len(mismatch_array))
            mismatch_array = mismatch_array[shuffled_indices]
            mismatch_labels = mismatch_labels[shuffled_indices]

        # 訓練用のデータ数を計算
        match_num = int(len(match_labels) * (1 - test_ratio))
        mismatch_num = int(len(mismatch_labels) * (1 - test_ratio))

        # 訓練用とテスト用に分割
        tr_pairs = np.concatenate([match_array[:match_num], mismatch_array[:mismatch_num]])
        tr_labels = np.concatenate([match_labels[:match_num], mismatch_labels[:mismatch_num]])
        te_pairs = np.concatenate([match_array[match_num:], mismatch_array[mismatch_num:]])
        te_labels = np.concatenate([match_labels[match_num:], mismatch_labels[mismatch_num:]])

        self.logger.info(tr_pairs[:, 0].shape)

        # 学習データが存在しなければ、ここで終了する
        if len(tr_pairs) == 0 or len(te_pairs) == 0:
            self.logger.warning("No training data or test data to train.")
            return

        # シャムネットワークを定義する
        model = self.__get_metric_model()

        # 指定があればベースモデルを読み込む
        if basemodel_filepath is not None:
            self.load_model(basemodel_filepath)

        # ログにsummaryを書き込む
        model.summary(print_fn=self.logger.info)
        FasttextCore.vector_model.summary(print_fn=self.logger.info)

        # 学習を行う
        tr_labels = tf.cast(tr_labels, dtype="float32")
        te_labels = tf.cast(te_labels, dtype="float32")

        # 最も良いモデルを保存する
        callback_for_tf = []
        if use_best_model:
            callback_for_tf.append(
                ModelCheckpoint(
                    "best_model.keras",
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                )
            )

        model.compile(loss=self.contrastive_loss, optimizer=RMSprop(), metrics=["accuracy"])

        H = model.fit(
            [tr_pairs[:, 0], tr_pairs[:, 1]],
            tr_labels,
            batch_size=128,
            epochs=EPOCHS,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_labels),
            callbacks=callback_for_tf,
        )

        # 結果を表示
        # ディレクトリ作成
        if not path.exists(image_dirpath):
            os.mkdir(image_dirpath)

        # ファイル名生成
        filepath = None
        for i in range(1, 10000):
            tmpname = path.join(image_dirpath, "loss-{:0=4}.png".format(i))
            if not path.exists(tmpname):
                filepath = tmpname
                break

        plt.clf()  # 初期化
        plt.title("Train/validation loss")
        plt.plot(H.history["loss"], label="training loss")
        plt.plot(H.history["val_loss"], label="validation loss")
        plt.grid()
        plt.legend()
        plt.savefig(filepath)

        # 最も良いモデルを利用する場合は、読み込みと削除を行う
        if use_best_model:
            self.logger.info("Load best model.")
            model.load_weights("best_model.keras")
            os.remove("best_model.keras")

        # 精度を確認
        # compute final accuracy on training and test sets
        tr_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = self.compute_accuracy(tr_labels, tr_pred)
        te_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = self.compute_accuracy(te_labels, te_pred)

        self.logger.info("Metric learning has been completed.")
        self.logger.info("Accuracy on training set: %0.2f%%" % (100 * tr_acc))
        self.logger.info("Accuracy on test set: %0.2f%%" % (100 * te_acc))

        # 精度のグラフ表示 (トレーニングデータ)
        # ファイル名生成
        filepath = None
        for i in range(1, 10000):
            tmpname = path.join(image_dirpath, "acc-tr-{:0=4}.png".format(i))
            if not path.exists(tmpname):
                filepath = tmpname
                break

        plt.clf()  # 初期化
        plt.figure(figsize=(10, 5))
        plt.plot(tr_labels, label="Pos or Neg", linestyle="none", marker="o", markersize=5, alpha=0.3)
        plt.plot(tr_pred, label="Distance", linestyle="none", marker="o", markersize=5, alpha=0.3)

        plt.legend()
        plt.grid()
        plt.savefig(filepath)

        # 精度のグラフ表示 (テストデータ)
        # ファイル名生成
        filepath = None
        for i in range(1, 10000):
            tmpname = path.join(image_dirpath, "acc-te-{:0=4}.png".format(i))
            if not path.exists(tmpname):
                filepath = tmpname
                break

        plt.clf()  # 初期化
        plt.figure(figsize=(10, 5))
        plt.plot(te_labels, label="Pos or Neg", linestyle="none", marker="o", markersize=5, alpha=0.3)
        plt.plot(te_pred, label="Distance", linestyle="none", marker="o", markersize=5, alpha=0.3)

        plt.legend()
        plt.grid()
        plt.savefig(filepath)

        FasttextCore.metric_model = model

    def load_model(self, filepath: str = "../model.keras"):
        """
        外部に保存された学習済み変換モデルのパラメータを読み込み、モデルを再構築する (kerasモデル)

        - - -

        Parameters
        ----------
        filepath: str
            学習モデルが保存されたパス, by default '../model.keras'
        """

        # h5モデルが読み込まれた場合にはエラーを出力する
        # 拡張子取得
        _, ext = path.splitext(filepath)

        if ext == ".h5":
            # 旧モデル
            raise RuntimeError(f"This is not a keras model: {filepath}")

        else:
            # kerasモデル
            self.logger.info(f"Start loading model: {filepath}")
            FasttextCore.metric_model = self.__get_metric_model()
            FasttextCore.vector_model = tf.keras.models.load_model(filepath)
            self.logger.info("Finished loading model")

    def save_model(self, filepath: str = "../model.keras"):
        """
        学習済みベクトル変換モデルのパラメータを保存する (kerasモデル)

        - - -

        Parameters
        ----------
        filepath: str
            学習モデルを保存するパス, by default '../model.keras'
        """

        if FasttextCore.metric_model is None:
            raise RuntimeError("No training data to save.")

        # 拡張子を取得し .keras でなければ追加する
        _, ext = path.splitext(filepath)
        if ext != ".keras":
            filepath += ".keras"

        self.logger.info(f"Start saving model: {filepath}")
        FasttextCore.vector_model.save(filepath)
        self.logger.info("Finished saving model")

    def get_distance(self, pair: np.ndarray):
        """
        fasttextにより表された2つのベクトルの距離を学習モデルを使って算出する

        ペアの行列を入力することで1度にまとめて距離を得られる

        - - -

        Parameter
        ---------
        pair: Tuple[np.ndattay]
            fasttextベクトルで表されたペアの行列
        """
        if FasttextCore.metric_model is None:
            raise RuntimeError("You should train or load h5 fasttext distance model.")

        # ----- Norm の値が異なっている問題 -----
        #
        # FasttextCore.metric_model.predict で得られる値と、
        # FasttextCore.vector_model.predict で変換した後に
        # np.linalg.norm で得られる値が異なる問題がある
        #
        # おそらく np.linalg.norm の値の方が正しいと思われるため、
        # 実装をこちらに切り替える (根本的な原因は不明)
        #
        # --------------------------------------
        #
        # return FasttextCore.metric_model.predict([pair[:, 0], pair[:, 1]])

        # 行列ペアをフラットにしてベクトルに変換する

        __result = FasttextCore.vector_model.predict(pair.reshape(-1, 300))
        result = np.linalg.norm(__result[::2] - __result[1::2], axis=1)

        return result

    def predict(self, vectors: np.ndarray):
        """
        fasttextにより表現されたベクトルを入力として学習モデルから新たなベクトルを推論する

        - - -

        Parameter
        ---------
        pair: Tuple[np.ndattay]
            fasttextベクトルで表されたペアの行列
        """

        if FasttextCore.vector_model is None:
            raise RuntimeError("You should train or load h5 fasttext distance model.")

        return FasttextCore.vector_model.predict(vectors)

    def match_pair_accuracy(self, image_dirpath: str = None):
        """ターゲットファイルについて、一致ペアの距離を計算する"""

        # ディレクトリ作成
        if not path.exists(image_dirpath):
            os.mkdir(image_dirpath)

        # 正解一致ペアのインデックスを作成
        rc = RecordContainer(log_filepath=self.log_filepath)
        rc.load_file(self.target_filepath)
        idx_pairs = rc.get_all_match_pairs_index()

        # 各レコードのベクトルを取得
        __record_array = self.get_fasttext_vectors_with_mp(CacheMode.WRITE, self.target_filepath)

        # ベクトルのペア生成
        vec_pairs: "list[list[np.ndarray, np.ndarray]]" = []
        for a, b in idx_pairs:
            vec_pairs.append([__record_array[a], __record_array[b]])
        vec_pairs = np.array(vec_pairs)

        # ペアの距離を計算
        pred = FasttextCore.metric_model.predict([vec_pairs[:, 0], vec_pairs[:, 1]])
        # tr_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])

        # ファイル名生成
        filepath = None
        for i in range(1, 10000):
            tmpname = path.join(image_dirpath, "matchpairs-{:0=4}.png".format(i))
            if not path.exists(tmpname):
                filepath = tmpname
                break

        # 可視化する
        plt.clf()  # 初期化
        plt.title("Distance of match pairs")
        plt.plot(pred, label="Distance", linestyle="none", marker="o", markersize=5, alpha=0.3)
        plt.grid()
        plt.legend()
        plt.savefig(filepath)

    def get_fasttext_vectors_with_mp(
        self,
        cache_mode: CacheMode,
        filepath: str = None,
        record: list[RecordMG] = None,
        max_cores: int = None,
        log_minutes: int = 5,
    ) -> np.ndarray:
        """
        Fasttextベクトルを生成する際に、キャッシュを利用して読み込む

        - - -

        Params
        ------
        cache: CacheMode
            キャッシュの利用方法
        filepath: str, by default None
            ターゲットのyamlファイルパス
        record: list[RecordMG], by default None
            RecordMGのリスト (あればターゲットファイルを読み込まずに、これを利用してベクトルを生成する)
        max_cores: int, by default None
            並列処理時に利用する最大コア数
        log_minutes: int, by default 5
            ログを出力する間隔
        """

        # filepathとrecordの指定のどちらもなければ、エラーを出す
        if filepath is None and record is None:
            raise ValueError("Either filepath or record must be specified.")

        # filepathがない場合は、キャッシュモードをNONEに変更する
        cache_mode = CacheMode.NONE if filepath is None else cache_mode

        cache_path = None

        # キャッシュが存在する場合は、読み込んでそれを返して終了する
        if cache_mode != CacheMode.NONE:
            # キャッシュの探索
            cache_path = path.join(
                path.dirname(filepath), ".cache", f"{path.splitext(path.basename(filepath))[0]}.ftv.npy"
            )

            if path.exists(cache_path):
                # 読み込みに失敗した場合は、エラーを出して生成処理に移る
                try:
                    self.logger.info(f"Load fasttext vectors from cache: {cache_path}")
                    return np.load(cache_path)
                except Exception as e:
                    self.logger.error(f"Failed to load cache: {cache_path}")
                    self.logger.error(e)

        # レコードが存在せずキャッシュが存在しない場合は、コンテナを生成してそれを読み込む
        if record is None:
            rc = RecordContainer()
            rc.load_file(filepath)
            record = rc.get_recordmg()

        # コア数の指定がない場合、応答用とメインプロセス用のコアを除いた全てのコアを使用する
        if max_cores is None:
            max_cores = mp.cpu_count() - 2
        max_cores = max(min(max_cores, mp.cpu_count(), int(len(record) / 100)), 1)

        # 並列処理によるベクトル生成
        # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化)
        self.logger.info("Start make_fasttext_dist_matrix.")
        calltime = datetime.now() + timedelta(minutes=log_minutes)
        with mp.Manager() as manager:
            counter = manager.Value("i", 0)
            with mp.Pool(processes=max_cores) as pool:
                __starmap = pool.starmap_async(
                    self.construct_vector,
                    [(r, counter) for r in record],
                )

                # 終了するまで監視しながら待機する
                while not __starmap.ready():
                    if datetime.now() > calltime:
                        lr = len(record)
                        cv = counter.value
                        self.logger.info("#Records: {} / {} ({}%)".format(cv, lr, int(cv * 100 / lr)))
                        calltime += timedelta(minutes=log_minutes)
                    time.sleep(0.1)

                record_array = __starmap.get()
        self.logger.info("Finished make_fasttext_dist_matrix.")

        # ベクトルをnumpy配列に変換
        record_array = np.array(record_array)

        # キャッシュモードが書き込みの場合は、キャッシュを保存する
        if cache_mode == CacheMode.WRITE:
            if not path.exists(path.dirname(cache_path)):
                os.makedirs(path.dirname(cache_path))

            np.save(cache_path, record_array)
            self.logger.info(f"Save fasttext vectors to cache: {cache_path}")

        return record_array
