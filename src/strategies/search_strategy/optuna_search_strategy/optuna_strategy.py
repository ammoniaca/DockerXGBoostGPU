from collections.abc import Callable, Sequence
from typing import Any

import optuna
from optuna.trial import Trial

import cupy as cp
import numpy as np

from sklearn.base import clone, BaseEstimator
from sklearn.exceptions import NotFittedError

from src.strategies.search_strategy.base_search_strategy import SearchStrategy, CVSplit
from src.strategies.search_strategy.metrics import METRICS
from src.strategies.search_strategy.search_space_builder import make_search_space


class OptunaSearchStrategy(SearchStrategy):

    def __init__(
            self,
            estimator: BaseEstimator,
            tuner: optuna.Study,
            *,
            n_trials: int = 100,
            metric: str = "rmse",
    ) -> None:
        super().__init__(
            estimator=estimator,
            tuner=tuner,
            n_trials=n_trials,
            metric=metric,
        )
        self._metric_fn, self._greater_is_better = METRICS[self.metric]
        self._is_fitted = False  # Flag

        expected = "MAXIMIZE" if self._greater_is_better else "MINIMIZE"
        if tuner.direction.name != expected:
            raise ValueError(f"Study direction {tuner.direction.name} does not match metric '{self.metric}'.")

    def __enter__(self) -> "OptunaSearchStrategy":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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

        objective = self._make_objective(X_gpu, y_gpu, folds, search_space)
        try:
            self.tuner.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=False
            )
            self._is_fitted = True
        finally:
            del objective, X_gpu, y_gpu, folds
            self._free_gpu_memory()

    def get_best_params(self) -> dict[str, Any]:
        """Return the hyperparameters of the best trial found by the study."""
        return self._check_fitted().best_params

    def get_results(self) -> Any:
        """Return the results of all trials of the study as a DataFrame.

        Each row corresponds to a trial, and includes its number, objective
        value, start/end time, duration, state, and the hyperparameters
        sampled (columns prefixed with ``params_``). Trials of every state
        (COMPLETE, PRUNED, FAIL) are included.

        Returns:
            pd.DataFrame: One row per trial, as returned by ``study.trials_dataframe()``.
        """
        return self._check_fitted().trials_dataframe()

    def get_best_score(self) -> float:
        """Return the best objective value found by the study.

        This is the lowest value if the study minimizes the objective,
        the highest if it maximizes it. Only completed trials are considered.

        Returns:
            float: Objective value of the best trial.
        """
        return self._check_fitted().best_value

    def _check_fitted(self) -> optuna.Study:
        if not self._is_fitted:
            raise NotFittedError("Call optimize() before accessing results.")
        return self.tuner

    def get_best_n_estimators(self) -> int | None:
        return self._check_fitted().best_trial.user_attrs.get("best_n_estimators")

    def _make_objective(
            self,
            X_gpu: Any,
            y_gpu: Any,
            folds: Sequence[tuple[Any, Any]],
            search_space: dict[str, Any],
    ) -> Callable[[optuna.Trial], float]:
        """Build the Optuna objective: mean CV score with per-fold pruning."""

        use_es = (
                self.estimator.get_params().get("early_stopping_rounds")
                is not None
        )

        def objective(trial: Trial) -> float:
            suggested_params = make_search_space(space=search_space, strategy="optuna", trial=trial)

            scores: list[float] = []
            n_trees: list[int] = []
            for fold_idx, (tr, va) in enumerate(folds):

                X_tr, y_tr = X_gpu[tr], y_gpu[tr]
                X_va, y_va = X_gpu[va], y_gpu[va]

                model = clone(self.estimator).set_params(
                    **suggested_params
                )

                eval_set = [(X_va, y_va)] if use_es else None

                # model.fit_on_gpu(X_tr, y_tr, eval_set=eval_set, verbose=False)
                model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
                pred = cp.asarray(model.predict(X_va))

                scores.append(self._metric_fn(y_va, pred))

                n_trees.append(
                    model.best_iteration + 1
                    if use_es
                    else model.get_booster().num_boosted_rounds()
                )

                # Report the running mean after each completed fold.
                intermediate_score = float(np.mean(scores))
                trial.report(intermediate_score, step=fold_idx)

                # Stop this trial if the pruner considers it unpromising.
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned after fold {fold_idx + 1}; "
                        f"intermediate score={intermediate_score:.6f}"
                    )


            trial.set_user_attr("n_trees", [int(n) for n in n_trees])
            trial.set_user_attr("best_n_estimators", int(round(np.mean(n_trees))))
            return float(np.mean(scores))

        return objective
