from typing import Literal

import numpy as np

StrategyType = Literal["rscv", "optuna"]

from optuna.trial import Trial
from scipy.stats import (
    randint,
    uniform,
    loguniform
)

def _optuna_search_space(trial: Trial, space: dict) -> dict:
    """build the search space for optuna strategy."""
    suggested: dict = {}
    for name, p in space.items():
        if isinstance(p, list):
            suggested[name] = trial.suggest_categorical(name, p)
            continue

        if p.prob == "randint":
            suggested[name] = trial.suggest_int(
                name=name, low=p.min, high=p.max, step=p.step or 1
            )
        elif p.prob == "uniform":
            suggested[name] = trial.suggest_float(
                name=name, low=p.min, high=p.max, step=p.step
            )
        elif p.prob == "loguniform":
            suggested[name] = trial.suggest_float(
                name=name, low=p.min, high=p.max, log=True
            )
        else:
            raise ValueError(
                f"Invalid type for '{name}': {p.prob!r} "
                "(use 'randint' or 'uniform' or 'loguniform')"
            )
    if suggested.get("grow_policy") == "depthwise":
        suggested.pop("max_leaves", None)
    return suggested


def _rscv_search_space(space: dict) -> dict:
    """build the search space for random search cv strategy."""
    suggested: dict = {}
    for name, p in space.items():
        if isinstance(p, list):
            suggested[name] = p
            continue

        prob_kind = p["prob"]
        step = p.get("step")

        if prob_kind == "randint":
            if step is None:
                suggested[name] = randint(low=p['min'], high=p['max']+1)
            else:
                suggested[name] = [int(v) for v in range(int(p['min']), int(p['max']) + 1, int(step))]
        elif prob_kind == "uniform":
            if step is None:
                suggested[name] = uniform(loc=p['min'], scale=p['max'] - p['min'])
            else:
                n = int(np.floor((p['max'] - p['min']) / step + 1e-9)) + 1
                suggested[name] = [round(p['min'] + step * k, 10) for k in range(n)]
        elif prob_kind == "loguniform":
            suggested[name] = loguniform(a=p['min'], b=p['max'])
        else:
            raise ValueError(
                f"Invalid type for '{name}': {prob_kind!r} "
                "(use 'randint' or 'uniform' or 'loguniform')"
            )
    return suggested


def make_search_space(
        space: dict,
        strategy: StrategyType,
        trial: Trial | None = None
) -> dict:
    """Factory method to build the search space for the given strategy."""
    searched_space: dict = {}
    if strategy == "rscv":
        return _rscv_search_space(space=space)
    elif strategy == "optuna":
        if trial is None:
            raise ValueError("strategy='optuna' requires a trial")
        return _optuna_search_space(trial=trial, space=space)
    raise ValueError(f"Unknown strategy: {strategy!r} (use 'rscv' or 'optuna')")

### ----- TEST ----- 
#
#
# space =  {
#     "n_estimators":      {"prob":"randint",   "min": 200,    "max": 2000,  "step":  10},
#     "eta":               {"prob":"loguniform",  "min": 0.01,  "max": 0.3,  "step":  None},
#     "max_depth":         {"prob":"randint",     "min": 3,     "max": 8,    "step":  1},
#     "max_leaves":        {"prob":"randint",     "min": 16,    "max": 128,  "step":  1},
#     "subsample":         {"prob":"uniform",     "min": 0.5,   "max": 1.0,  "step":  None},
#     "colsample_bytree":  {"prob":"uniform",     "min": 0.5,   "max": 1.0,  "step":  None},
#     "colsample_bylevel": {"prob":"uniform",     "min": 0.5,   "max": 1.0,  "step":  None},
#     "min_child_weight":  {"prob":"loguniform",  "min": 0.1,   "max": 30.0, "step":  None},
#     "reg_alpha":         {"prob":"loguniform",  "min": 1e-8,  "max": 7.0,  "step":  None},
#     "reg_lambda":        {"prob":"loguniform",  "min": 1e-8,  "max": 7.0,  "step":  None},
#     "gamma":             {"prob":"loguniform",  "min": 1e-8,  "max": 2.0,  "step":  None},
#     "grow_policy":       ["depthwise", "lossguide"]
# }
#
# import optuna
# study = optuna.create_study()
# trial = study.ask()
# p_optuna = make_search_space(space=space, strategy="optuna", trial=trial)
# print(p_optuna)
# p_rscv = make_search_space(space=space, strategy="rscv")
# print(p_rscv)