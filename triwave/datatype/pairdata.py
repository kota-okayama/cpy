"""Pair data"""

from dataclasses import dataclass, asdict


@dataclass
class Pairdata:
    """ペアの推論結果等を格納しておくデータ型"""

    inf_dist: float = None  # 距離学習によって算出した距離, by default None
    inf_same: float = None  # 一致と判定した推論結果, by default None
    inf_not: float = None  # 不一致と判定した推論結果, by default None
    inf_unknown: float = None  # 不明と判定した推論結果, by default None

    def to_dict(self) -> "dict[str, float]":
        """
        データを辞書型に変換する

        - - -

        Return
        ------
        result: dict[str, str]
            Pairdataの持つデータを辞書形式で返す
        """

        return asdict(self)
