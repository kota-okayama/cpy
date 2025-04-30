"""Inference"""

import os
import multiprocessing as mp
import numpy as np

from datetime import datetime

from sklearn.mixture import BayesianGaussianMixture
from scipy import stats
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import Levenshtein

from .utils import path
from .fasttext_core import FasttextCore, THRESHOLD
from .logger import Logger, LoggerConfig
from .file_container import RecordContainer, ParamsContainer, ContainerType, judge_container_type
from .datatype.workflow import WorkflowConfig, CacheMode
from .datatype.record import RecordMG
from .datatype.fitting_params import (
    DistributionType,
    ParamsType,
    BetaParams,
    GammaParams,
    GaussianParams,
)
from .datatype.sun_params import SunParams

THRESHOLD = THRESHOLD


class Inference:
    """確率分布へのフィッティングおよびベイズ更新を行うクラス"""

    def __init__(self, ft_core: FasttextCore, config: WorkflowConfig = None, log_filepath: str = None):
        """コンストラクタ"""

        self.config = config
        self.log_filepath = log_filepath
        self.logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level),
            filepath=self.log_filepath,
        )

        self.ft_core = ft_core
        self.params_container: ParamsContainer = None

    def _fit_distribution(
        self,
        params_type: ParamsType,
        data: np.ndarray,
        comoponents: int = 10,
    ) -> "list[BetaParams] | list[GammaParams] | list[GaussianParams]":
        """
        確率密度関数へのフィッティングを行う

        - - -

        Params
        ------
        dist_type: str
            フィッティングする確率密度関数の種類
        data: np.ndarray
            フィッティングするデータ
        components: int, by default 10
            混合分布の場合の分布最大数
        """

        result = None

        # 念のため変換
        data = np.array(data)

        if params_type == ParamsType.BETA:
            # データからベータ分布を得る
            fit_parameter = stats.beta.fit(data)
            result = [
                BetaParams(
                    alpha=float(fit_parameter[0]),
                    beta=float(fit_parameter[1]),
                    loc=float(fit_parameter[2]),
                    scale=float(fit_parameter[3]),
                    weight=1,
                )
            ]

        elif params_type == ParamsType.GAMMA:
            fit_parameter = stats.gamma.fit(data)
            result = [
                GammaParams(
                    alpha=float(fit_parameter[0]),
                    loc=float(fit_parameter[1]),
                    beta=float(fit_parameter[2]),
                    weight=1,
                )
            ]

        elif params_type == ParamsType.GAUSSIAN:
            # データからベイジアンガウス分布を得る
            bgm = BayesianGaussianMixture(n_components=comoponents, max_iter=1000, random_state=0)
            bgm.fit(data[:, np.newaxis])  # 2次元配列に変換する

            result = []
            # 各ガウス分布のパラメータを取得
            for i in range(bgm.n_components):
                if float(bgm.weights_[i]) > 0.0001:
                    result.append(
                        GaussianParams(
                            mean=float(bgm.means_[i, 0]),
                            cov=float(bgm.covariances_[i, 0, 0]),
                            weight=float(bgm.weights_[i]),
                        )
                    )

        return result

    def _draw_distribution(
        self,
        data: np.ndarray,
        params: "list[list[BetaParams] | list[GammaParams] | list[GaussianParams]]",
        title: str = "Fit probability density function",
        axs: Axes = None,
    ):
        """
        フィッティング結果を与えられたaxsにグラフ化して格納する

        - - -

        Params
        ------
        data: np.ndarray
            フィッティングするデータ
        params: list[BetaParams | GammaParams | GaussianParams]
            フィッティング結果のパラメータ集合
        title: str
            グラフタイトル
        axs: Axes, by default None
            グラフを格納するAxes
        """

        # 念のため変換
        data = np.array(data)

        ax1 = axs
        ax2 = ax1.twinx()

        ax1.hist(data, range=(0, max(1, max(data))), bins=50, label="training data")  # ヒストグラムの追加
        ax1.set_xlabel("Distance")
        ax1.set_ylabel("Number of pairs")
        ax2.set_ylabel(r"$p(x_{i,j})$")

        max_height = 0
        for p in params:
            x = np.linspace(0, max(1, max(data)), 601)

            if p[0].type == ParamsType.BETA:
                value = stats.beta.pdf(x, *p[0].params_for_draw()[0])
                ax2.plot(x, value, color="#FFA500", label="Beta")

            elif p[0].type == ParamsType.GAMMA:
                value = stats.gamma.pdf(x, *p[0].params_for_draw()[0])
                ax2.plot(x, value, color="#FFA500", label="Gamma")

            elif p[0].type == ParamsType.GAUSSIAN:
                value = 0
                for i in p:
                    value += i.weight * (np.exp(-((x - i.mean) ** 2) / (2 * i.cov)) / (np.sqrt(2 * np.pi * i.cov)))
                ax2.plot(x, value, color="#FF4040", label="Gaussian" if len(p) == 1 else "Mixed Gaussian")

            max_height = max(max_height, value.max())

        ax2.legend(loc=0)
        ax2.set_ylim(0, min(10, max_height) * 1.1)
        ax1.set_title(title)

    def _get_distribution(
        self,
        sim: float,
        params: "list[BetaParams] | list[GammaParams] | list[GaussianParams]",
    ) -> float:
        """
        与えられたパラメータと類似度から、確率密度関数から得られる値を返す

        - - -

        Params
        ------
        sim: float
            類似度・距離
        params: list[BetaParams] | list[GammaParams] | list[GaussianParams]
            フィッティング結果のパラメータ集合

        Return
        ------
        float
            類似度・距離を指定し、確率密度分布から得られた値
        """

        result = 0

        if params[0].type == ParamsType.BETA:
            if params[0].loc is not None and params[0].scale is not None:
                result = stats.beta.pdf(sim, params[0].alpha, params[0].beta, params[0].loc, params[0].scale)
            else:
                result = stats.beta.pdf(sim, params[0].alpha, params[0].beta)

            result = float(result) if result > 0.000001 else 0.000001

        elif params[0].type == ParamsType.GAMMA:
            if params[0].loc is not None:
                result = stats.gamma.pdf(sim, params[0].alpha, params[0].loc, params[0].beta)
            else:
                result = stats.gamma.pdf(sim, a=params[0].alpha, scale=params[0].beta)

            result = float(result) if result > 0.000001 else 0.000001

        elif params[0].type == ParamsType.GAUSSIAN:
            for p in params:
                result += p.weight * (np.exp(-((sim - p.mean) ** 2) / (2 * p.cov)) / (np.sqrt(2 * np.pi * p.cov)))

        return result

    def make_fasttext_dist_matrix(self, record: RecordMG, counter=None) -> np.ndarray:
        """
        [API] 書誌データから、学習したモデルを用いたfasttextの距離の算出を行う際に使用する行列を生成する

        - - -

        Params
        ------
        record: RecordMG
            fasttextベクトルに変換したい書誌データ

        Return
        ------
        np.ndarray
            300次元ベクトルが属性値数分格納された行列
        """

        return self.ft_core.construct_vector(record, counter)

    def convert_vector_by_metric_learner(self, vectors: "list[np.ndarray]") -> np.ndarray:
        """
        [API] 書誌データから、距離学習モデルによって変換される行列を生成する

        - - -

        Params
        ------
        pairs: list[np.ndattay]
            fasttextベクトルで表された書誌データペアの行列

        Return
        ------
        np.ndarray
            各々のベクトルが格納された配列
        """

        return self.ft_core.predict(vectors)

    def fasttext_predict(self, pairs: "list[np.ndarray]") -> np.ndarray:
        """
        (convert_vector_by_metric_learner)

        [API] fasttextにより表された2つのベクトルの距離を学習モデルを使って算出する

        ペアの行列を入力することで1度にまとめて距離を得られる

        - - -

        Params
        ------
        pairs: list[np.ndattay]
            fasttextベクトルで表された書誌データペアの行列

        Return
        ------
        np.ndarray
            各々のペアの距離が格納された1次元ベクトル
        """

        return np.ravel(self.ft_core.get_distance(pairs))

    def get_fasttext_vectors_with_mp(
        self,
        cache_mode: CacheMode,
        filepath: str = None,
        record: list[RecordMG] = None,
        max_cores: int = None,
        log_minutes: int = 5,
    ) -> np.ndarray:
        """
        [API] Fasttextベクトルを生成する際に、キャッシュを利用して読み込む

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

        return self.ft_core.get_fasttext_vectors_with_mp(cache_mode, filepath, record, max_cores, log_minutes)

    def load_params(self, params_path: str = "./params.yaml"):
        """
        ベータ分布やガンマ分布のフィッティングで用いるパラメータをyamlファイルから読み込みインスタンス変数に格納する

        - - -

        Params
        ------
        params_path: str
            パラメータが格納されたyamlファイル, by default './params.yaml'
        """

        self.params_container = ParamsContainer.load_params(params_path)

    def _calc_p_of_each_indicator(
        self,
        fasttext_sim,
        difflib_sim,
        leven_sim,
        jaro_sim,
    ) -> "tuple[float, float, float, float]":
        """各々の指標値と確率密度関数から、一致確率値を算出する"""

        # フィッティング結果を格納
        same = []
        same.append(self._get_distribution(fasttext_sim, self.params_container.fasttext_match))
        same.append(self._get_distribution(difflib_sim, self.params_container.difflib_match))
        same.append(self._get_distribution(leven_sim, self.params_container.leven_match))
        same.append(self._get_distribution(jaro_sim, self.params_container.jaro_match))

        not_same = []
        not_same.append(self._get_distribution(fasttext_sim, self.params_container.fasttext_mismatch))
        not_same.append(self._get_distribution(difflib_sim, self.params_container.difflib_mismatch))
        not_same.append(self._get_distribution(leven_sim, self.params_container.leven_mismatch))
        not_same.append(self._get_distribution(jaro_sim, self.params_container.jaro_mismatch))

        return (
            same[0] / (same[0] + not_same[0]),
            same[1] / (same[1] + not_same[1]),
            same[2] / (same[2] + not_same[2]),
            same[3] / (same[3] + not_same[3]),
        )

    def _sun_huge_area(self, p: "tuple[float, float, float, float]", min_area: float = 0.5):
        """SUN推論時の各確率値を渡すことで、一番大きい領域を出力する (min_areaを超えない場合はNoneを返す)"""
        # not-same値を含めたリスト
        prob = [[1 - p[0], p[0]], [1 - p[1], p[1]], [1 - p[2], p[2]], [1 - p[3], p[3]]]

        # 全ての領域の割り出し
        area = []
        for i in range(len(p) ** 2):
            tmp = []
            for j in range(len(p)):
                tmp.append((i >> j) & 1)
            area.append(tuple(tmp))

        result = [-1 for _ in range(len(p))]
        for a in area:
            p = 1
            for i in range(len(a)):
                p *= prob[i][a[i]]

            if p > min_area:
                result = a

        return tuple(result)

    def generate_params(
        self,
        origin_path: str = "./data_te.yaml",
        target_filepath: str = None,
        distribution_type: DistributionType | None = None,
        image_title: str = None,
        image_dirpath: str = None,
        max_cores: int = None,
        log_minutes: int = 10,
    ):
        """
        渡されたyaml形式の書誌データの fasttext(学習した距離) / SequenceMatcher / leven / jaro の距離を算出し、

        フィッテング結果のパラメータをインスタンス変数に保存する

        - - -

        Parameters
        ----------
        origin_path: str
            学習に利用するyaml形式のレコードが格納されたファイルパス or パラメータが格納されたファイルパス, by default './data_te.yaml'
        target_filepath: str, by default None
            クラウドソーシングの結果が保存されている "絶対" ファイルパス
        distribution_type: DistributionType, by default None
            フィッティングに用いる確率分布の種類
        image_title: str, by default None
            フィッティング画像のタイトル
        image_dirpath: str, by default None
            フィッティング画像の保存先ディレクトリ
        max_cores: int, by default None
            並列処理時に利用する最大コア数
        log_minutes: int, by default 10
            ログを出力する間隔
        """
        # 分布についての指定がない場合に設定されるデフォルト
        if distribution_type is None:
            distribution_type = DistributionType.MIXED_GAUSSIAN

        recordmg: list[RecordMG] = []
        labels: list[int] = []
        record_array: np.ndarray = None

        # === レコードペアから分散値を取得する (トレーニングデータ読み込み) ===
        # origin_path が指定されていて ContainerType.TARGET であれば、そのファイルを読み込む
        if origin_path is not None and judge_container_type(origin_path) == ContainerType.TARGET:
            score = datetime.now().timestamp()
            rc = RecordContainer(log_filepath=self.log_filepath)
            rc.load_file(origin_path)
            __rc_data = rc.get_recordmg_for_train()

            recordmg, labels = __rc_data[0] + __rc_data[1], __rc_data[2] + __rc_data[3]

            self.logger.info(f"ft_core.load_file: {int((datetime.now().timestamp() - score) * 1000)}")

            score = datetime.now().timestamp()
            # recordからそれぞれをベクトルに変換し、行列化する (multiprocessによる並列化 / 大元の学習データ)
            record_array = np.array([])
            __record_list = []
            __record_array = self.get_fasttext_vectors_with_mp(
                CacheMode.WRITE, origin_path, None, max_cores, log_minutes
            )
            for r in recordmg:
                __record_list.append(__record_array[r.re.idx])
            record_array = np.array(__record_list)
            self.logger.info(
                f"get_fasttext_vectors_with_mp (standard): {int((datetime.now().timestamp() - score) * 1000)}"
            )

        # === レコードペアから分散値を取得する (クラウドソーシング済みデータ) ===
        score = datetime.now().timestamp()
        if target_filepath is not None:
            # 学習用データに整形
            __cs_data = self.ft_core.load_crowdsourcing_pair()
            tmp_re, tmp_la = __cs_data[0] + __cs_data[1], __cs_data[2] + __cs_data[3]
            recordmg += tmp_re
            labels += tmp_la

            # ターゲットファイルのレコードをベクトルに変換し、行列化する (multiprocessによる並列化 / クラウドソーシングで追加されたデータ)
            __record_list = []
            __record_array = self.get_fasttext_vectors_with_mp(
                CacheMode.WRITE,
                target_filepath,
                None,
                max_cores,
                log_minutes,
            )
            for r in tmp_re:
                __record_list.append(__record_array[r.re.idx])

            if record_array is None:
                record_array = np.array(__record_list)
            elif len(__record_list) > 0:
                record_array = np.concatenate([record_array, np.array(__record_list)])

        self.logger.info(f"load_crowdsourcing_pair: {int((datetime.now().timestamp() - score) * 1000)}")

        record_array_pair = {"fasttext": [], "difflib": [], "leven": [], "jaro": []}
        record_array_not = {"fasttext": [], "difflib": [], "leven": [], "jaro": []}

        score = datetime.now().timestamp()
        for i in range(len(recordmg) // 2):
            # 結合文字列を作成する (その他用)
            record1_string = self.ft_core.construct_string(recordmg[i * 2])
            record2_string = self.ft_core.construct_string(recordmg[i * 2 + 1])

            if labels[i] == 1:
                record_array_pair["fasttext"].append([record_array[i * 2], record_array[i * 2 + 1]])
                record_array_pair["difflib"].append(SequenceMatcher(None, record1_string, record2_string).ratio())
                record_array_pair["leven"].append(
                    Levenshtein.distance(record1_string, record2_string) / len(record1_string + record2_string)
                )
                record_array_pair["jaro"].append(Levenshtein.jaro_winkler(record1_string, record2_string))
            else:
                record_array_not["fasttext"].append([record_array[i * 2], record_array[i * 2 + 1]])
                record_array_not["difflib"].append(SequenceMatcher(None, record1_string, record2_string).ratio())
                record_array_not["leven"].append(
                    Levenshtein.distance(record1_string, record2_string) / len(record1_string + record2_string)
                )
                record_array_not["jaro"].append(Levenshtein.jaro_winkler(record1_string, record2_string))
        self.logger.info(f"make_dist_matrix: {int((datetime.now().timestamp() - score) * 1000)}")

        # fasttextデータを距離学習で算出した距離を出力
        score = datetime.now().timestamp()

        if len(record_array_pair["fasttext"]) > 0:
            record_array_pair["fasttext"] = self.fasttext_predict(np.array(record_array_pair["fasttext"]))
        if len(record_array_not["fasttext"]) > 0:
            record_array_not["fasttext"] = self.fasttext_predict(np.array(record_array_not["fasttext"]))

        self.logger.info(f"fasttext_predict: {int((datetime.now().timestamp() - score) * 1000)}")

        original_data = [
            np.array(record_array_pair["fasttext"]),
            np.array(record_array_not["fasttext"]),
            np.array(record_array_pair["difflib"]),
            np.array(record_array_not["difflib"]),
            np.array(record_array_pair["leven"]),
            np.array(record_array_not["leven"]),
            np.array(record_array_pair["jaro"]),
            np.array(record_array_not["jaro"]),
        ]

        # === 既存パラメータ値から分散値を取得する ===
        # origin_path が指定されていて ContainerType.PARAMS であれば、そのファイルを読み込む
        if origin_path is not None and judge_container_type(origin_path) == ContainerType.PARAMS:
            __params_container = ParamsContainer.load_params(origin_path)
            # weightが大きいと処理が止まってしまうため、サンプリング値を設定
            # __weight = [__params_container.weight_match, __params_container.weight_mismatch]
            __weight = [5000, 5000]

            for i, params in enumerate(__params_container.get_params_list()):
                __sample = np.array([])

                if params[0].type == ParamsType.BETA:
                    __sample = np.random.beta(params[0].alpha, params[0].beta, __weight)

                    if params[0].loc is not None and params[0].scale is not None:
                        __sample = params[0].loc + params[0].scale * __sample

                elif params[0].type == ParamsType.GAMMA:
                    __sample = np.random.gamma(params[0].alpha, params[0].beta, __weight)

                    if params[0].loc is not None:
                        __sample = params[0].loc + __sample

                elif params[0].type == ParamsType.GAUSSIAN:
                    gmm = BayesianGaussianMixture()
                    gmm.means_ = np.array([p.mean for p in params]).reshape(-1, 1)
                    gmm.covariances_ = np.array([p.cov for p in params]).reshape(-1, 1, 1)
                    gmm.weights_ = np.array([p.weight for p in params])

                    __sample = gmm.sample(n_samples=__weight[i % 2])[0].flatten()

                original_data[i] = np.concatenate([original_data[i], __sample])

        score = datetime.now().timestamp()

        # ===== Fitting =====

        # 並列処理でベータ分布とガンマ分布のフィッティングを行う
        with mp.Pool(max_cores) as pool:
            static_beta_gamma = pool.starmap(
                self._fit_distribution,
                [(ParamsType.GAMMA if i & 2 else ParamsType.BETA, d) for i, d in enumerate(original_data)],
            )
        static_beta_gamma.append(original_data[0].size)  # weight_match の追加
        static_beta_gamma.append(original_data[1].size)  # weight_mismatch の追加

        self.logger.info(f"_fit_distribution(beta_gamma): {int((datetime.now().timestamp() - score) * 1000)}")

        score = datetime.now().timestamp()

        self.logger.info(f"{[d.size for d in original_data]}")
        self.logger.info(original_data[0])

        # 並列処理で混合ガウス分布のフィッティングを行う
        with mp.Pool(max_cores) as pool:
            mixed_gaussian = pool.starmap(self._fit_distribution, [(ParamsType.GAUSSIAN, d) for d in original_data])
        mixed_gaussian.append(original_data[0].size)  # weight_match の追加
        mixed_gaussian.append(original_data[1].size)  # weight_mismatch の追加

        self.logger.info(f"_fit_distribution(mixed_gaussian): {int((datetime.now().timestamp() - score) * 1000)}")

        # ParamsContainerの初期化
        self.params_container = ParamsContainer()

        # ベータ分布とガンマ分布パラメータ格納 (Bubble)
        score = datetime.now().timestamp()
        if distribution_type == DistributionType.STATIC_BETA_GAMMA:
            self.params_container.update_params(*static_beta_gamma)

        # 混合ガウス分布のパラメータ格納
        elif distribution_type == DistributionType.MIXED_GAUSSIAN:
            self.params_container.update_params(*mixed_gaussian)

        self.logger.info(f"update_params: {int((datetime.now().timestamp() - score) * 1000)}")

        score = datetime.now().timestamp()
        # フィッティングと分布に対する画像生成
        if image_dirpath is not None:
            axs = None

            # 画像の出力
            if image_title is not None:
                plt.clf()  # 初期化
                figure, axs = plt.subplots(2, 4, figsize=(20, 8))
                figure.suptitle(image_title)

                # 画像の生成
                distribution_title = ["fasttext", "difflib", "leven", "jaro"]
                i = 0
                for d, j, k in zip(original_data, static_beta_gamma, mixed_gaussian):
                    self._draw_distribution(
                        d,
                        [j, k],
                        f"{distribution_title[i // 2]} - {'mismatch' if i & 1 else 'match'}",
                        axs[i % 2, i // 2],
                    )
                    i += 1

                # ディレクトリ作成
                if not path.exists(image_dirpath):
                    os.mkdir(image_dirpath)

                # ファイル名生成
                filepath = None
                for i in range(1, 10000):
                    tmpname = path.join(image_dirpath, "fitting-{:0=4}.png".format(i))
                    if not path.exists(tmpname):
                        filepath = tmpname
                        break

                # 既に規定数以上のファイルが保存されている場合はエラーを出す
                if filepath is None:
                    raise FileExistsError("Cannot save PNG file because there are already too many files (over 10000).")

                # 画像の出力
                plt.tight_layout()
                plt.legend()
                plt.savefig(filepath)

        self.logger.info(f"draw_distribution: {int((datetime.now().timestamp() - score) * 1000)}")

        score = datetime.now().timestamp()
        # SUN推論時に使うパラメータを計算  TODO: 統計的アプローチでもう少し詰められると良いかも。とりあえずいまは経験則で決定で...
        # 一致に関するパラメータの推定
        same_tendency = {}
        for i in range(len(record_array_pair.keys()) ** 2):
            tmp = []
            for j in range(len(record_array_pair.keys())):
                tmp.append((i >> j) & 1)
            same_tendency[(tuple(tmp))] = 0
        same_tendency[(tuple([-1 for _ in range(len(record_array_pair.keys()))]))] = 0

        for fasttext_sim, difflib_sim, leven_sim, jaro_sim in zip(
            record_array_pair["fasttext"],
            record_array_pair["difflib"],
            record_array_pair["leven"],
            record_array_pair["jaro"],
        ):
            fitting_result = self._calc_p_of_each_indicator(fasttext_sim, difflib_sim, leven_sim, jaro_sim)
            same_tendency[self._sun_huge_area(fitting_result)] += 1

        # 不一致に関するパラメータの推定
        not_same_tendency = {}
        for i in range(len(record_array_not.keys()) ** 2):
            tmp = []
            for j in range(len(record_array_not.keys())):
                tmp.append((i >> j) & 1)
            not_same_tendency[(tuple(tmp))] = 0
        not_same_tendency[(tuple([-1 for _ in range(len(record_array_not.keys()))]))] = 0

        for fasttext_sim, difflib_sim, leven_sim, jaro_sim in zip(
            record_array_not["fasttext"],
            record_array_not["difflib"],
            record_array_not["leven"],
            record_array_not["jaro"],
        ):
            fitting_result = self._calc_p_of_each_indicator(fasttext_sim, difflib_sim, leven_sim, jaro_sim)
            not_same_tendency[self._sun_huge_area(fitting_result)] += 1

        same_area = [tuple([1 for _ in range(len(record_array_pair.keys()))])]
        not_same_area = [tuple([0 for _ in range(len(record_array_pair.keys()))])]
        exclude = [
            (tuple([1 for _ in range(len(record_array_pair.keys()))])),
            (tuple([0 for _ in range(len(record_array_pair.keys()))])),
            (tuple([-1 for _ in range(len(record_array_pair.keys()))])),
        ]
        threshold = 0.4  # 一致と不一致で比較した際にこの値未満の差があれば領域に追加する  TODO: このパラメータはクラウドソーシング数と精度の要のためもっと検討すべき
        for key in same_tendency.keys():
            entire = same_tendency[key] + not_same_tendency[key]
            if key in exclude or entire < 1:
                continue

            if same_tendency[key] / entire < threshold:
                not_same_area.append(key)
            if not_same_tendency[key] / entire < threshold:
                same_area.append(key)

        self.params_container.sun_params = SunParams(
            0.3, 0.5, same_area, not_same_area
        )  # TODO: tau値も動的に決定できると良い

        self.logger.info(f"calc_sun_params: {int((datetime.now().timestamp() - score) * 1000)}")

        self.logger.info(same_tendency)
        self.logger.info(not_same_tendency)

    def save_params(self, filepath: str = "./params.yaml"):
        """
        パラメータをyamlファイルを出力する

        - - -

        Params
        ------
        filepath: str
            出力先のパス
        """

        self.params_container.save_params_yaml(filepath)

    def greet_bayesian(
        self,
        record1: RecordMG,
        record2: RecordMG,
        dist: float = None,
    ) -> "tuple[float, float, float, list[float], list[float], list[float], list[float]]":
        """
        ベイズ推論により同定する2つのレコードを指定し、一致・不一致の確率密度関数にフィッティングし確率を推定するメソッド

        - - -

        Params
        ------
        record1: RecordMG
            1つ目のレコード
        record2: RecordMG
            2つ目のレコード

        Return
        ------
        tuple[float, float, list[float], list[float]]
            [same, not_same, sameの属性それぞれの確率密度の値のリスト, not_sameの属性それぞれの確率密度の値のリスト] (別のデータクラスを定義の検討もあり？)
        """

        # 疑似距離使用の可否
        e_pseudo = self.config.ngraph_construction == "DIST" and not self.config.ngraph_params.enable_pseudo_dist

        # 結合文字列を作成する (その他用)
        record1_string = self.ft_core.construct_string(record1)
        record2_string = self.ft_core.construct_string(record2)

        # 類似度・距離算出
        if dist is not None:
            fasttext_sim = dist
        elif e_pseudo and dist is None:
            # ペアを格納したnumpy配列を作成する (fasttext用)
            record_array = [self.ft_core.construct_vector(record1), self.ft_core.construct_vector(record2)]
            fasttext_sim = float(self.fasttext_predict(np.array([record_array]))[0])
        else:
            fasttext_sim = self.config.ngraph_params.dist * 2

        difflib_sim = SequenceMatcher(None, record1_string, record2_string).ratio()
        leven_sim = Levenshtein.distance(record1_string, record2_string) / len(record1_string + record2_string)
        jaro_sim = Levenshtein.jaro_winkler(record1_string, record2_string)

        sim = [fasttext_sim, difflib_sim, leven_sim, jaro_sim]

        # フィッティング結果を格納
        same = []
        same.append(self._get_distribution(fasttext_sim, self.params_container.fasttext_match))
        same.append(self._get_distribution(difflib_sim, self.params_container.difflib_match))
        same.append(self._get_distribution(leven_sim, self.params_container.leven_match))
        same.append(self._get_distribution(jaro_sim, self.params_container.jaro_match))

        not_same = []
        not_same.append(self._get_distribution(fasttext_sim, self.params_container.fasttext_mismatch))
        not_same.append(self._get_distribution(difflib_sim, self.params_container.difflib_mismatch))
        not_same.append(self._get_distribution(leven_sim, self.params_container.leven_mismatch))
        not_same.append(self._get_distribution(jaro_sim, self.params_container.jaro_mismatch))

        # ベイズ推論で用いる事前分布の設定(一様分布)
        # P_same = 0.024390244
        # P_not_same = 0.975609756097561
        P_same = 0.1
        P_not_same = 0.9
        P_theta_same = (
            np.prod(same) * P_same / np.maximum((np.prod(same) * P_same + np.prod(not_same) * P_not_same), 0.0001)
        )
        P_theta_not_same = (
            np.prod(not_same)
            * P_not_same
            / np.maximum((np.prod(same) * P_same + np.prod(not_same) * P_not_same), 0.0001)
        )
        P_unknwon_de = np.prod([s + n for s, n in zip(same, not_same)])
        P_unknown = 1 - (float(np.prod(same) / P_unknwon_de) + float(np.prod(not_same) / P_unknwon_de))

        return (float(P_theta_same), float(P_unknown), float(P_theta_not_same), sim, same, not_same, dist)

    def greet_sun(
        self, record1: RecordMG, record2: RecordMG, dist: float = None, use_sun_params: bool = False
    ) -> "tuple[float, float, float, list[float], list[float], list[float], list[float]]":
        """
        SUN推論により、同定する2つのレコードを指定し、一致・不一致の確率密度関数にフィッティングし確率を推定するメソッド

        - - -

        Parameters
        ----------
        record1: RecordMG
            1つ目のレコード
        record2: RecordMG
            2つ目のレコード

        Return
        ------
        tuple[float, float, float, list[float], list[float]]
            [same, not_same, sameの属性それぞれの確率密度の値のリスト, not_sameの属性それぞれの確率密度の値のリスト] (別のデータクラスを定義の検討もあり？)
        """

        # ペアを格納したnumpy配列を作成する (fasttext用)
        record_array = []
        if dist is None:
            record_array.append(self.ft_core.construct_vector(record1))
            record_array.append(self.ft_core.construct_vector(record2))

        # 結合文字列を作成する (その他用)
        record1_string = self.ft_core.construct_string(record1)
        record2_string = self.ft_core.construct_string(record2)

        # 類似度・距離算出
        fasttext_sim = float(self.fasttext_predict(np.array([record_array]))[0]) if dist is None else dist
        difflib_sim = SequenceMatcher(None, record1_string, record2_string).ratio()
        leven_sim = Levenshtein.distance(record1_string, record2_string) / len(record1_string + record2_string)
        jaro_sim = Levenshtein.jaro_winkler(record1_string, record2_string)

        sim = [fasttext_sim, difflib_sim, leven_sim, jaro_sim]

        # フィッティング結果を格納
        fitting_result = self._calc_p_of_each_indicator(fasttext_sim, difflib_sim, leven_sim, jaro_sim)
        index: "list[tuple[float, float]]" = []
        index.append([1 - fitting_result[0], fitting_result[0]])
        index.append([1 - fitting_result[1], fitting_result[1]])
        index.append([1 - fitting_result[2], fitting_result[2]])
        index.append([1 - fitting_result[3], fitting_result[3]])

        # 確率密度関数から確率を算出
        prob = []
        for i in range(len(index)):
            prob.append([index[i][0] / (index[i][0] + index[i][1]), index[i][1] / (index[i][0] + index[i][1])])

        # TODO: 教師データを解析して動的に決められると良いが、精度的に決めにくいので統計的アプローチを入れたい
        same = set([(1, 1, 1, 1), (0, 1, 1, 1)])
        not_same = set(
            [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                # (0, 1, 0, 0),
                # (0, 0, 1, 0),
                # (0, 0, 0, 1),
            ]
        )
        unknown = set()

        # 教師データから得られたsun_paramsを使う
        if use_sun_params:
            same = self.params_container.sun_params.same_area
            not_same = self.params_container.sun_params.not_same_area

        for i in range(len(index) ** 2):
            tmp = []
            for j in range(len(index)):
                tmp.append((i >> j) & 1)
            unknown.add(tuple(tmp))

        unknown = unknown - (same | not_same)

        # sameの値の計算
        P_theta_same = 0
        for s in same:
            p = 1
            for i in range(len(s)):
                p *= prob[i][s[i]]
            P_theta_same += p

        # not_sameの値の計算
        P_theta_not_same = 0
        for s in not_same:
            p = 1
            for i in range(len(s)):
                p *= prob[i][s[i]]
            P_theta_not_same += p

        P_theta_unknown = 1 - (P_theta_same + P_theta_not_same)

        return (P_theta_same, P_theta_unknown, P_theta_not_same, sim, same, not_same, dist)

    # def _jaro_for_deim(self, origin_path: str = "./data_te.yaml", target_dirpath: str = None):
    #     """DEIM論文のグラフ用データを出力する(jaro)"""

    #     record, labels = self.ft_core.load_file(origin_path, 5000, 5000)

    #     record_array_pair = {"jaro": []}
    #     record_array_not = {"jaro": []}

    #     for i in range(len(record) // 2):
    #         # 結合文字列を作成する (その他用)
    #         record1_string = self.ft_core.construct_string(record[i * 2])
    #         record2_string = self.ft_core.construct_string(record[i * 2 + 1])

    #         if labels[i] == 1:
    #             record_array_pair["jaro"].append(Levenshtein.jaro_winkler(record1_string, record2_string))
    #         else:
    #             record_array_not["jaro"].append(Levenshtein.jaro_winkler(record1_string, record2_string))

    #     # 各々を適切な確率分布にフィッティングさせ結果を出力
    #     be_title = "Fit probability density function (beta)"

    #     fitting_params_same_pair = FittingParams(
    #         fasttext=Params(0, 0),
    #         difflib=None,
    #         leven=None,
    #         jaro=self._fit_beta(
    #             record_array_pair["jaro"],
    #             path.join(path.dirname(path.abspath(__file__)), "..", "img", "jaro_same.png"),
    #             "{} / jaro - same".format(be_title),
    #         ),
    #     )
    #     # fitting_params_same_pair.save_yaml(params_same_pair_path)  # TODO: 書き換える

    #     fitting_params_not = FittingParams(
    #         fasttext=None,
    #         difflib=None,
    #         leven=None,
    #         jaro=self._fit_beta(
    #             record_array_not["jaro"],
    #             path.join(path.dirname(path.abspath(__file__)), "..", "img", "jaro_not.png"),
    #             "{} / jaro - not-same".format(be_title),
    #         ),
    #     )
    #     # fitting_params_not.save_yaml(params_not_pair_path)  # TODO: 書き換える

    #     self.params_same: FittingParams = fitting_params_same_pair
    #     self.params_not: FittingParams = fitting_params_not

    #     # dat出力
    #     result = {}
    #     for value in record_array_pair["jaro"]:
    #         result[int(value * 20)] = result.get(int(value * 20), 0) + 1
    #     with open(path.join(target_dirpath, "same.dat"), "w") as f:
    #         f.write("C\tV\n")
    #         for key in sorted(result.keys()):
    #             f.write("{:.02f}\t{}\n".format(key / 20, result[key]))

    #     result = {}
    #     for value in record_array_not["jaro"]:
    #         result[int(value * 20)] = result.get(int(value * 20), 0) + 1
    #     with open(path.join(target_dirpath, "not_same.dat"), "w") as f:
    #         f.write("C\tV\n")
    #         for key in sorted(result.keys()):
    #             f.write("{:.02f}\t{}\n".format(key / 20, result[key]))

    #     # params出力
    #     ParamsContainer.save_params_yaml(
    #         path.join(target_dirpath, "params.yaml"),
    #         self.params_same,
    #         self.params_not,
    #         SunParams(None, None, [], []),
    #     )
