from typing import Any, Literal
from sklearn.base import BaseEstimator

from src.strategies.base_strategy import BaseStrategy, Regressor

DeviceType = Literal["cpu", "cuda"]

class VanillaStrategy(BaseStrategy):

    def fit(self, X: Any, y: Any, device: DeviceType = "cuda") -> Regressor:
        return self._fit(X, y, device)
