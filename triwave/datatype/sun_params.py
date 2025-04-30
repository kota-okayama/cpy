"""Params for SUN inference"""

from dataclasses import dataclass


@dataclass
class SunParams:
    """SUN推論のパラメータを格納するデータクラス"""

    same_tau: float
    not_same_tau: float
    same_area: "set[tuple[int, int, int, int]]"
    not_same_area: "set[tuple[int, int, int, int]]"

    def to_dict(self) -> "dict[str, str]":
        """
        データを辞書型に変換する

        - - -

        Return
        ------
        result: dict[str, str]
            SunParamsの持つデータを辞書形式で返す
        """

        return {
            "same_tau": self.same_tau,
            "not_same_tau": self.not_same_tau,
            "same_area": [list(a) for a in self.same_area],
            "not_same_area": [list(a) for a in self.not_same_area],
        }
