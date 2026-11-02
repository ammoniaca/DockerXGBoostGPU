from collections.abc import Sequence
from typing import Any

import cupy as cp
import numpy as np
import pandas as pd

from sklearn.base import clone, BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RandomizedSearchCV

from src.strategies.search_strategy.metrics import METRICS
from src.strategies.search_strategy.base_search_strategy import SearchStrategy, CVSplit
from src.strategies.search_strategy.search_space_builder import make_search_space

class RandomSearchStrategy(SearchStrategy):
    """Hyperparameter search backed by sklearn's RandomizedSearchCV."""

    def __init__(
            self,
            estimator: BaseEstimator,
            tuner: RandomizedSearchCV,
            *,
            n_trials: int = 100,
            metric: str = "rmse"
    ) -> None:
        super().__init__(
            estimator=estimator,
            tuner=tuner,
            n_trials=n_trials,
            metric=metric
        )
        self._metric_fn, self._greater_is_better = METRICS[self.metric]
        self._search: RandomizedSearchCV | None = None

    def _check_fitted(self) -> RandomizedSearchCV:
        if self._search is None:
            raise NotFittedError("Call optimize() before accessing results.")
        return self._search

    def optimize(
            self,
            X: Any,
            y: Any,
            search_space: dict[str, Any],
            cv_splits: Sequence[CVSplit]
    ) -> None:
        # Data on device once (no-op if already CuPy arrays).
        X_gpu, y_gpu = cp.asarray(X), cp.asarray(y)
        # Folds on device once; accepts NumPy or CuPy indices.
        folds = [(cp.asarray(tr), cp.asarray(va)) for tr, va in cv_splits]

        suggested_params = make_search_space(space=search_space, strategy="rscv")

        params = {
            "estimator": self.estimator,
            "n_iter": self.n_trials,
            "param_distributions": suggested_params,
            "cv": folds,
            "scoring": make_scorer(
                self._metric_fn, greater_is_better=self._greater_is_better
            ),
            "n_jobs": 1, # 1: a single process owns the GPU
            "refit": False,
            "verbose": 0
        }

        search = clone(self.tuner).set_params(**params)
        try:
            search.fit(X_gpu, y_gpu)
            self._search = search
        finally:
            self._free_gpu_memory()

    def get_best_params(self) -> dict[str, Any]:
        return self._check_fitted().best_params_

    def get_results(self) -> Any:
        return pd.DataFrame(self._check_fitted().cv_results_)

    def get_best_score(self) -> float:
        score = self._check_fitted().best_score_
        return score if self._greater_is_better else -score