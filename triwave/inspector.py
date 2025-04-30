"""[Deprecated] Inspector"""

from .file_container import BookdataContainer
from .inference import Inference
from .logger import Logger


class Inspector:
    """フレームワークを精査するためのクラス"""

    def __init__(self):
        """コンストラクタ"""

        self.logger = Logger(__name__)
        self.tau_same = 0.3
        self.tau_not = 0.5

    def sun_inf_for_correct_groups(self, inference: Inference, filepath: str):
        """正解一致ペアに対してSUN推論を適用し、結果を得る"""

        # カウンターの初期化
        same = 0
        unknown = 0
        not_same = 0

        # 書誌データグループを読み込む
        bd_container = BookdataContainer()
        bd_container.load_file(filepath)

        # 一致ペアをすべて抽出する
        bdmg, _ = bd_container.get_bookdata_for_train(num_of_mismatch=0)

        for i in range(len(bdmg) // 2):
            s, n, _, _, _, _ = inference.greet_sun(bdmg[i * 2], bdmg[i * 2 + 1])

            if 1 - s < self.tau_same:
                same += 1

            elif 1 - n < self.tau_not:
                not_same += 1

            else:
                unknown += 1

        result = "\n"
        result += "[Inference results for matched pairs (SUN)]\n"
        result += "     Same |  Unknown | Not-same \n"
        result += "--------------------------------\n"
        result += " {:>8d} | {:>8d} | {:>8d} ".format(same, unknown, not_same)

        self.logger.info(result)

    def sun_inf_for_all_groups(self, inference: Inference, filepath: str):
        """すべてのペアに対してSUN推論を適用し、結果を得る"""

        # カウンターの初期化
        match_same = 0
        match_unknown = 0
        match_not = 0
        mismatch_same = 0
        mismatch_unknown = 0
        mismatch_not = 0

        # 書誌データグループを読み込む
        bd_container = BookdataContainer()
        bd_container.load_file(filepath)

        # 書誌データ数と一致数を計算する
        num_of_bd = 0
        num_of_match = 0
        for v in bd_container.bookdata_group.values():
            num_of_bd += len(v)
            num_of_match += len(v) * (len(v) - 1) // 2

        num_of_mismatch = (num_of_bd) * (num_of_bd - 1) // 2 - num_of_match

        # ペアをすべて抽出する(不一致ペアは重複の可能性あり)
        bdmg, bdmg_pair = bd_container.get_bookdata_for_train(
            num_of_match=num_of_match, num_of_mismatch=num_of_mismatch
        )

        for i in range(len(bdmg) // 2):
            s, n, _, _, _, _ = inference.greet_sun(bdmg[i * 2], bdmg[i * 2 + 1])

            if bdmg_pair[i] == 1:  # 一致ペア
                if 1 - s < self.tau_same:
                    match_same += 1

                elif 1 - n < self.tau_not:
                    match_not += 1

                else:
                    match_unknown += 1

            else:  # 不一致ペア
                if 1 - s < self.tau_same:
                    mismatch_same += 1

                elif 1 - n < self.tau_not:
                    mismatch_not += 1

                else:
                    mismatch_unknown += 1

        result = "\n"
        result += "[Inference results for matched pairs (SUN)]\n"
        result += "                    | inference\n"
        result += "                    |     Same |  Unknown | Not-same \n"
        result += "-------------------------------------------------\n"
        result += " correct |     Same | {:>8d} | {:>8d} | {:>8d} \n".format(match_same, match_unknown, match_not)
        result += "           Not-same | {:>8d} | {:>8d} | {:>8d} ".format(
            mismatch_same, mismatch_unknown, mismatch_not
        )

        self.logger.info(result)
