"""Group evaluation"""

from dataclasses import dataclass


@dataclass
class GroupEvaluation:
    """グループ評価を行う際のデータ保持・計算を行うデータクラス"""

    match_nu: float = 0  # 一致指標の分子 (R and I)
    match_de: int = 0  # 一致指標の分母 (R or I)
    mismatch_nu: float = 0  # 不一致指標の分子 ((R - I) or (I - R))
    mismatch_de: int = 0  # 不一致指標の分母 (R or I)
    precision_nu: float = 0  # 書誌誤同定指標の分子 (I - R)
    precision_de: int = 0  # 書誌誤同定指標の分母 (I)
    recall_nu: float = 0  # 書誌割れ指標の分子 (R - I)
    recall_de: int = 0  # 書誌割れ指標の分母 (R)
    match2_nu: float = 0  # 正解群数から誤った書誌データ数を引いた指標の分子 (len(R) + 1 - len((R - I) | (I - R)))
    match2_de: int = 0  # 正解群数から誤った書誌データ数を引いた指標の分母 (len(R) + 1)
    complete_nu: int = 0  # 正解群と推論群が完全に一致した群数
    complete_de: int = 0  # 正解群の群数

    def __add__(self, group_evaluation: "GroupEvaluation") -> "GroupEvaluation":
        """分子分母をそれぞれ合計する"""

        # GroupEvaluation以外の型の場合はTypeError
        if not isinstance(group_evaluation, GroupEvaluation):
            raise TypeError(
                f"unsupported operand type(s) for +: 'GroupEvaluation' and '{group_evaluation.__class__.__name__}'"
            )

        result = GroupEvaluation(
            self.match_nu + group_evaluation.match_nu,
            self.match_de + group_evaluation.match_de,
            self.mismatch_nu + group_evaluation.mismatch_nu,
            self.mismatch_de + group_evaluation.mismatch_de,
            self.precision_nu + group_evaluation.precision_nu,
            self.precision_de + group_evaluation.precision_de,
            self.recall_nu + group_evaluation.recall_nu,
            self.recall_de + group_evaluation.recall_de,
            self.match2_nu + group_evaluation.match2_nu,
            self.match2_de + group_evaluation.match2_de,
            self.complete_nu + group_evaluation.complete_nu,
            self.complete_de + group_evaluation.complete_de,
        )

        return result

    def calc_evaluation(self) -> "GroupEvaluation":
        """分子分母の形から浮動小数型に直す"""

        result = GroupEvaluation(
            self.match_nu / self.match_de if self.match_de > 0 else 1.0,
            1,
            self.mismatch_nu / self.mismatch_de if self.mismatch_de > 0 else 0.0,
            1,
            self.precision_nu / self.precision_de if self.precision_de > 0 else 0.0,
            1,
            self.recall_nu / self.recall_de if self.recall_de > 0 else 0.0,
            1,
            self.match2_nu / self.match2_de if self.match2_de > 0 else 0.0,
            1,
            self.complete_nu / self.complete_de if self.complete_de > 0 else 0.0,
            1,
        )

        return result
