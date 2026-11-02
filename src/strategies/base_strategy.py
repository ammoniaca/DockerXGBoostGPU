import gc
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, Self
import cupy as cp
import numpy as np
import numpy.typing as npt
from sklearn.base import clone, BaseEstimator

DeviceType = Literal["cpu", "cuda"]

class Regressor(Protocol):

    def set_params(self, **params: Any) -> Self:
        ...

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...

    @property
    def feature_importances_(self) -> npt.NDArray[np.floating[Any]]:
        ...

class BaseStrategy(ABC):

    def __init__(
            self,
            estimator: BaseEstimator
    ) -> None:

        self.estimator: BaseEstimator = estimator

    def close(self) -> None:
        """Release GPU memory. Results remain readable afterwards."""
        self._free_gpu_memory()

    @abstractmethod
    def fit(self, X: Any, y: Any, device: DeviceType = "cuda") -> Regressor:
        """Return the final model trained on (X, y) on the given device."""

    def _fit(
            self,
            X: Any,
            y: Any,
            device: DeviceType,
            params: dict[str, Any] | None = None
    ) -> Regressor:

        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

        model: Any = clone(self.estimator)

        if params:
            model.set_params(**params)
        if device == "cpu":
            model.set_params(device="cpu")
            return model.fit(self._to_host(X), self._to_host(y))

        model.set_params(device="cuda")

        X_gpu, y_gpu = cp.asarray(X), cp.asarray(y)  # no-op se già CuPy

        try:
            return model.fit(X_gpu, y_gpu)
        finally:
            del X_gpu, y_gpu
            self._free_gpu_memory()

    @staticmethod
    def _to_host(a: Any) -> Any:
        return a.get() if isinstance(a, cp.ndarray) else a

    @staticmethod
    def _free_gpu_memory() -> None:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()