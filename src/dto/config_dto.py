from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from src.strategies.search_strategy.optuna_search_strategy.optuna_factory_sampler import SamplerName
from src.strategies.search_strategy.optuna_search_strategy.optuna_factory_pruner import PrunerName



class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

class VanillaDTO(DTO):
    subsample: float
    colsample_bytree: float


class SamplerDTO(DTO):
    name: SamplerName
    params: dict[str, Any] = Field(default_factory=dict)


class PrunerDTO(DTO):
    name: PrunerName
    params: dict[str, Any] = Field(default_factory=dict)


class OptunaDTO(DTO):
    direction: str
    sampler: SamplerDTO
    pruner: PrunerDTO


class StrategyDTO(DTO):
    metric: str
    n_trials: int = Field(gt=0)
    n_k_folds: int = Field(ge=2)
    vanilla: VanillaDTO
    optuna: OptunaDTO
    rscv: dict[str, Any] = Field(default_factory=dict)


class DistributionDTO(DTO):
    prob: str
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None


class ModelDTO(DTO):
    fixed: dict[str, Any]
    suggested: dict[str, DistributionDTO | list[str]]


class SettingDTO(DTO):
    strategy: StrategyDTO
    model: ModelDTO


class ConfigDTO(DTO):
    setting: SettingDTO