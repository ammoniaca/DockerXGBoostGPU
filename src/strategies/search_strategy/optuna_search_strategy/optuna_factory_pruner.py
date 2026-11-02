from enum import StrEnum
from typing import Any

from optuna.pruners import (
    BasePruner,
    MedianPruner,               # Stops trials whose intermediate results are worse than the median of previous trials at the corresponding step.
    NopPruner,                  # Disables pruning and allows every trial to run to completion.
    PatientPruner,              # Wraps another pruner and tolerates a specified number of steps without sufficient improvement before pruning a trial.
    PercentilePruner,           # Stops trials whose intermediate results are worse than a specified percentile of previous trials at the corresponding step.
    SuccessiveHalvingPruner,    # Allocates progressively more resources to promising trials and eliminates less promising ones.
    HyperbandPruner,            # Combines multiple Successive Halving brackets to allocate computational resources efficiently across trials.
    ThresholdPruner,            # Stops trials when intermediate results fall outside predefined lower or upper thresholds.
    WilcoxonPruner              # Uses a statistical Wilcoxon signed-rank test to determine whether a trial should be pruned based on intermediate performance.
)

class PrunerName(StrEnum):
    MEDIAN = "median"
    NOP = "nop"
    PATIENT = "patient"
    PERCENTILE = "percentile"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    THRESHOLD = "threshold"
    WILCOXON = "wilcoxon"

_PRUNERS: dict[PrunerName, type[BasePruner]] = {
    PrunerName.MEDIAN: MedianPruner,
    PrunerName.NOP: NopPruner,
    PrunerName.PATIENT: PatientPruner,
    PrunerName.PERCENTILE: PercentilePruner,
    PrunerName.SUCCESSIVE_HALVING: SuccessiveHalvingPruner,
    PrunerName.HYPERBAND: HyperbandPruner,
    PrunerName.THRESHOLD: ThresholdPruner,
    PrunerName.WILCOXON: WilcoxonPruner,
}

def make_pruner(name: str | PrunerName, **kwargs: Any) -> BasePruner:
    """Factory method for Optuna pruner"""
    cls = _PRUNERS[PrunerName(name)]
    if cls is NopPruner:
        return cls()
    if cls is PatientPruner:
        wrapped = kwargs.pop("wrapped_pruner", None)
        if isinstance(wrapped, dict):
            wrapped = make_pruner(wrapped["name"], **wrapped.get("params", {}))
        kwargs["wrapped_pruner"] = wrapped
    return cls(**kwargs)