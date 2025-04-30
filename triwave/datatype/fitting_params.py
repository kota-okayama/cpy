"""Fitting params"""

from dataclasses import dataclass


class DistributionType:
    """taskごとに設定可能な分布型"""

    STATIC_BETA_GAMMA = "STATIC_BETA_GAMMA"  # Bubbleで採用。ベータ分布とガンマ分布の組み合わせ
    MIXED_GAUSSIAN = "MIXED_GAUSSIAN"  # 混合ガウス分布


class ParamsType:
    """ParamsContainerで利用するパラメータ型"""

    BETA = "BETA"
    GAMMA = "GAMMA"
    GAUSSIAN = "GAUSSIAN"


@dataclass
class BetaParams:
    """ベータ分布のパラメータを格納するデータクラス"""

    type: ParamsType = ParamsType.BETA
    alpha: float | None = None
    beta: float | None = None
    loc: float | None = None
    scale: float | None = None
    weight: float = 1

    def params_for_draw(self) -> "list[float]":
        """ベータ分布のパラメータをリストにして返す"""
        return [self.alpha, self.beta, self.loc, self.scale], self.weight


@dataclass
class GammaParams:
    """ガンマ分布のパラメータを格納するデータクラス"""

    type: ParamsType = ParamsType.GAMMA
    alpha: float | None = None
    beta: float | None = None
    loc: float | None = None
    weight: float = 1

    def params_for_draw(self) -> "list[float]":
        """ガンマ分布のパラメータをリストにして返す"""
        return [self.alpha, self.loc, self.beta], self.weight


@dataclass
class GaussianParams:
    """ガウス分布のパラメータを格納するデータクラス"""

    type: ParamsType = ParamsType.GAUSSIAN
    mean: float | None = None
    cov: float | None = None
    weight: float = 1
