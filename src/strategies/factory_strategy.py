from enum import StrEnum
from collections.abc import Callable

import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from src.dto.config_dto import SamplerDTO, PrunerDTO
from src.strategies.search_strategy.base_search_strategy import SearchStrategy
from src.strategies.search_strategy.optuna_search_strategy.optuna_strategy import OptunaSearchStrategy
from src.strategies.search_strategy.optuna_search_strategy.optuna_factory_sampler import make_sampler
from src.strategies.search_strategy.optuna_search_strategy.optuna_factory_pruner import make_pruner
from src.strategies.search_strategy.rscv_search_strategy.rscv_strategy import RandomSearchStrategy

from typing import Literal

StudyDirection  = Literal["minimize", "maximize"] | None


class SearchStrategyName(StrEnum):
    OPTUNA = "optuna"
    RSCV = "rscv"


_STRATEGIES: dict[SearchStrategyName, Callable[..., SearchStrategy]] = {
    SearchStrategyName.OPTUNA: OptunaSearchStrategy,
    SearchStrategyName.RSCV: RandomSearchStrategy,
}

# Tipo atteso per `study` di ciascuna strategia.
_TUNER_TYPES: dict[SearchStrategyName, type] = {
    SearchStrategyName.OPTUNA: optuna.Study,
    SearchStrategyName.RSCV: RandomizedSearchCV,
}

def _make_optuna_tuner(
        direction: StudyDirection = None,
        sampler: SamplerDTO | None = None,
        pruner: PrunerDTO | None = None,
        seed: int | None = None
) -> optuna.Study:
    spr = (
        make_sampler(name=sampler.name, seed=seed, **sampler.params)
        if sampler is not None else None
    )
    pnr = (
        make_pruner(name=pruner.name, **pruner.params)
        if pruner is not None else None
    )
    return optuna.create_study(
        direction=direction,
        sampler=spr,
        pruner=pnr,
    )


def make_search_strategy(
    name: str | SearchStrategyName,
    estimator: BaseEstimator,
    *,
    n_trials: int = 100,
    metric: str = "rmse",
    seed: int | None = None,
    direction: StudyDirection = None,
    sampler: SamplerDTO | None = None,
    pruner: PrunerDTO | None = None,
    **kwargs,
) -> SearchStrategy:
    """Create a hyperparameter search strategy."""

    key = SearchStrategyName(name)

    if key is SearchStrategyName.OPTUNA:
        tuner = _make_optuna_tuner(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            seed=seed,
        )
    else:
        # Build the RandomizedSearchCV instance here,
        # using the required search configuration.
        tuner = RandomizedSearchCV(
            estimator=estimator,
            random_state=seed,
            **kwargs,
        )

    return _STRATEGIES[key](
        estimator=estimator,
        tuner=tuner,
        n_trials=n_trials,
        metric=metric,
    )
