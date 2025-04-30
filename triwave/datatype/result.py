"""Result data class"""

from dataclasses import dataclass, asdict


@dataclass
class ContractionResult:
    """縮約結果を格納するデータクラス"""

    true_positive: int = 0  # 肯定的推論かつ一致
    false_negative: int = 0  # 否定的推論かつ一致 (書誌割れ)
    false_positive: int = 0  # 肯定的推論かつ不一致 (書誌誤同定)
    true_negative: int = 0  # 否定的推論かつ不一致

    def sum(self) -> int:
        """全ての値の合計値を返す"""
        return self.true_positive + self.false_negative + self.false_positive + self.true_negative

    def calc_precision(self) -> float:
        """適合率を計算し結果を返す"""
        if self.true_positive + self.false_positive == 0:  # TODO: 吟味
            return 1.0

        return self.true_positive / (self.true_positive + self.false_positive)

    def calc_recall(self) -> float:
        """再現率を計算し結果を返す"""
        if self.true_positive + self.false_negative == 0:  # TODO: 吟味
            return 1.0

        return self.true_positive / (self.true_positive + self.false_negative)

    def calc_f1(self) -> float:
        """f1値を計算する"""
        if self.calc_precision() + self.calc_recall() == 0:
            return 0.0

        return 2 * self.calc_precision() * self.calc_recall() / (self.calc_precision() + self.calc_recall())

    def to_dict(self) -> "dict[str, str]":
        """
        データを辞書型に変換する

        Return
        ------
        result: dict[str, str]
            ContractionResultの持つデータを辞書形式で返す
        """

        return asdict(self)
