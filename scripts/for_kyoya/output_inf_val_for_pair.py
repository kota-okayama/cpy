"""ペアを入力にして、推論値をyamlファイルで出力する (モジュールが読み込めないのでディレクトリのトップで実行してください)"""

import os
import codecs
import yaml
from uuid import uuid4

from triwave.datatype.record import RecordMG, Record, RecordType
from triwave.fasttext_core import FasttextCore
from triwave.inference import Inference


class PairContainer:
    """レコードペアを生成するクラス"""

    def __init__(self, fasttext_path: str, model_path: str, params_path: str):
        """コンストラクタ

        Params
        ------
        fasttext_path: str
            fasttextの辞書ファイルパス
        model_path: str
            距離学習モデルのパス
        params_path: str
            確率密度関数パラメータのパス
        """

        # 変数初期化
        self.match_pairs: "list[tuple[RecordMG, RecordMG, float]]" = []  # レコード1, レコード2, 推論値
        self.mismatch_pairs: "list[tuple[RecordMG, RecordMG, float]]" = []  # レコード1, レコード2, 推論値

        # データ型
        self.inf_attr = {
            "title": RecordType.TEXT,
            "author": RecordType.COMPLEMENT_JA,
            "publisher": RecordType.COMPLEMENT_JA,
            "pubdate": RecordType.COMPLEMENT_DATE,
        }

        dirpath = os.path.dirname(os.path.abspath(__file__))

        # モデル読み込み
        self.fasttext_core = FasttextCore(inf_attr=self.inf_attr)
        self.fasttext_core.load_fasttext_dict(os.path.join(dirpath, fasttext_path))
        self.fasttext_core.load_h5model(os.path.join(dirpath, model_path))
        self.inference = Inference(self.fasttext_core)
        self.inference.load_params(os.path.join(dirpath, params_path))

    def load_yaml(self, filepath):
        """yamlファイルを読み込む"""

        dirpath = os.path.dirname(os.path.abspath(__file__))

        # yamlファイルの存在チェックと読み込み
        with open(os.path.join(dirpath, filepath), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            # データ格納
            for i, target in enumerate([self.mismatch_pairs, self.match_pairs]):
                for pair in data["books"][abs(i - 1)][f"{i}"]:
                    target.append(
                        [
                            RecordMG(Record(str(uuid4()), "", pair[0]), self.inf_attr),
                            RecordMG(Record(str(uuid4()), "", pair[1]), self.inf_attr),
                            -1,
                        ]
                    )

    def get_inference(self):
        """推論する"""
        for pair in self.match_pairs:
            pair[2] = self.inference.greet_bayesian(pair[0], pair[1])[0]

        for pair in self.mismatch_pairs:
            pair[2] = self.inference.greet_bayesian(pair[0], pair[1])[0]

    def save_yaml(self, filepath):
        """yamlファイルへ結果の出力"""

        dirpath = os.path.dirname(os.path.abspath(__file__))

        with codecs.open(os.path.join(dirpath, filepath), "w", "utf-8") as f:
            data = {"books": {"1": [], "0": []}}

            # データ格納
            for i, target in enumerate([self.mismatch_pairs, self.match_pairs]):
                for pair in target:
                    data["books"][f"{i}"].append(pair[2])

            yaml.dump(data, f, indent=2, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    # ペアを入力にして、推論値をyamlファイルで出力する
    pair_container = PairContainer(
        "../cc.ja.300.bin",
        "benchmark/bib_kyoto/model_kyoto.h5",
        "benchmark/bib_kyoto/params_kyoto.yaml",
    )
    pair_container.load_yaml("pair_data_ver2.yaml")
    pair_container.get_inference()
    pair_container.save_yaml("output.yaml")
