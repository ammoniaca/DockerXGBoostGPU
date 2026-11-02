
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from src.strategies.base_strategy import BaseStrategy, Regressor

from src.strategies.search_strategy.metrics import check_metric

# One CV fold: (train_indices, validation_indices)
CVSplit = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]

DeviceType = Literal["cpu", "cuda"]


class SearchStrategy(BaseStrategy, ABC):
    """Common interface for hyperparameter search strategies."""

    def __init__(
            self,
            estimator: BaseEstimator,
            tuner: Any,
            *,
            n_trials: int = 100,
            metric: str = "rmse"
    ) -> None:
        """
        Args:
            estimator: Estimator whose hyperparameters are tuned.
            n_trials: Number of search trials.
            metric: Name of the metric to optimize (e.g. "rmse").
            random_state: Seed for reproducibility (None = non-deterministic).
            verbose: Verbosity level (0 = silent).
        """
        super().__init__(estimator)
        check_metric(metric)
        self.tuner = tuner
        self.estimator: BaseEstimator = estimator
        self.n_trials: int = n_trials
        self.metric: str = metric

    @abstractmethod
    def optimize(self, X: Any, y: Any, search_space: dict[str, Any], cv_splits: Sequence[CVSplit]) -> None:
        """Run the hyperparameter search.
        Args:
            X: Feature matrix (array-like or DataFrame).
            y: Target values.
            cv_splits: Pre-computed CV folds as (train_idx, val_idx) pairs.
                A Sequence is required (not a generator) because the folds
                are iterated once per trial.
            search_space:
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_params(self) -> dict[str, Any]:
        """Return the best hyperparameters found.
        Must be called after `optimize`."""
        raise NotImplementedError

    @abstractmethod
    def get_results(self) -> Any:
        """Return the full search results (e.g. a DataFrame with one row per trial).
        Must be called after `optimize`."""
        raise NotImplementedError

    @abstractmethod
    def get_best_score(self) -> float:
        """Return the best CV score, following the sklearn convention
        (higher is better, so errors are negated: -RMSE).

        All implementations must use the same sign.
        """
        raise NotImplementedError

    def get_best_n_estimators(self) -> int | None:
        """Number of trees for the final refit; None if the strategy doesn't provide one."""
        return None

    def fit(self, X: Any, y: Any, device: DeviceType = "cuda") -> Regressor:
        params = dict(self.get_best_params())
        params["early_stopping_rounds"] = None

        n_est = self.get_best_n_estimators()
        if n_est is not None:
            params["n_estimators"] = n_est

        return self._fit(X, y, device, params)

    def summary(self) -> dict[str, Any]:
        """Compact, loggable summary of the search (non-abstract helper)."""
        return {
            "best_params": self.get_best_params(),
            "best_score": self.get_best_score(),
        }