"""GPU status"""

from dataclasses import dataclass


@dataclass
class GPUStatus:
    """GPU(Cuda)を利用したモジュール起動の可否"""

    tensorflow: bool = False
    cupy: bool = False
    faiss: bool = False
