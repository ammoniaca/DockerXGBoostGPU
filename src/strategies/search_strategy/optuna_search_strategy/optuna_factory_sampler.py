from enum import StrEnum
from typing import Any

from optuna.samplers import (
    BaseSampler,
    TPESampler,             # Uses Tree-structured Parzen Estimator modeling to guide the search toward promising hyperparameter configurations.
    RandomSampler,          # Samples hyperparameter configurations randomly from the search space.
    GridSampler,            # Exhaustively evaluates configurations from a predefined parameter grid.
    BruteForceSampler,      # Systematically explores all possible configurations in a finite search space.
    QMCSampler,             # Uses quasi-Monte Carlo sequences to improve coverage of the search space.
    GPSampler,              # Uses Gaussian process-based Bayesian optimization to select promising hyperparameter configurations.
    CmaEsSampler,           # Uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize continuous and numerical hyperparameters.
    NSGAIISampler,          # Uses the NSGA-II evolutionary algorithm for multi-objective optimization.
    NSGAIIISampler,         # Uses the NSGA-III evolutionary algorithm for multi-objective optimization, particularly with many objectives.
    PartialFixedSampler     # Fixes specified hyperparameters and delegates the remaining parameters to another sampler.
)

class SamplerName(StrEnum):
    TPE = "tpe"
    RANDOM = "random"
    GRID = "grid"
    BRUTE = "brute"
    QMC = "qmc"
    GP = "gp"
    CMAES = "cmaes"
    NSGA2 = "nsga2"
    NSGA3 = "nsga3"
    PFIX = "pfix"


_SAMPLERS: dict[SamplerName, type[BaseSampler]] = {
    SamplerName.TPE: TPESampler,
    SamplerName.RANDOM: RandomSampler,
    SamplerName.GRID: GridSampler,
    SamplerName.BRUTE: BruteForceSampler,
    SamplerName.QMC: QMCSampler,
    SamplerName.GP: GPSampler,
    SamplerName.CMAES: CmaEsSampler,
    SamplerName.NSGA2: NSGAIISampler,
    SamplerName.NSGA3: NSGAIIISampler,
    SamplerName.PFIX: PartialFixedSampler,
}

def make_sampler(name: str | SamplerName, seed: int | None = None, **kwargs: Any) -> BaseSampler:
    """Factory method for Optuna sampler"""
    cls = _SAMPLERS[SamplerName(name)]
    if cls is not GridSampler:
        kwargs.setdefault("seed", seed)
    return cls(**kwargs)


# sampler = make_sampler("tpe", seed=42, n_startup_trials=20, multivariate=True)
# study = optuna.create_study(direction="minimize", sampler=sampler)
# from JSON
# import json
# from pathlib import Path
# {"sampler": {"name":  "tpe", "params":  { "n_startup_trials": 100, "n_ei_candidates": 64}}}
# config = json.loads(Path("config/sampler.json").read_text())
# cfg = config["sampler"]
# sampler = make_sampler(cfg["name"], cfg.get("seed"), **cfg.get("params", {}))
