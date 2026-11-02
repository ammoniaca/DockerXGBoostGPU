from collections.abc import Callable
from typing import Any, NamedTuple

import cupy as cp

MetricFn = Callable[[Any, Any], float]

# --- Errors (lower is better) --------------------------------------------

def _rmse(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Root mean squared error regression loss."""
    xp = cp.get_array_module(y_true)
    return float(xp.sqrt(xp.mean((y_pred - y_true) ** 2)))


def _mse(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Mean squared error regression loss."""
    xp = cp.get_array_module(y_true)
    return float(xp.mean((y_pred - y_true) ** 2))


def _mae(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Mean absolute error regression loss."""
    xp = cp.get_array_module(y_true)
    return float(xp.mean(xp.abs(y_pred - y_true)))


def _median_ae(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Median absolute error regression loss."""
    xp = cp.get_array_module(y_true)
    return float(xp.median(xp.abs(y_pred - y_true)))


def _max_error(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Maximum residual error, the worst-case error between y_true and y_pred."""
    xp = cp.get_array_module(y_true)
    return float(xp.max(xp.abs(y_pred - y_true)))


def _mape(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Mean absolute percentage error regression loss."""
    # Same as sklearn: denominator clipped at eps, zeros are NOT excluded.
    xp = cp.get_array_module(y_true)
    eps = xp.finfo(xp.float64).eps
    return float(xp.mean(xp.abs(y_pred - y_true) / xp.maximum(xp.abs(y_true), eps)))


def _msle(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Mean squared logarithmic error regression loss."""
    # Requires y_true > -1 and y_pred > -1.
    xp = cp.get_array_module(y_true)
    return float(xp.mean((xp.log1p(y_pred) - xp.log1p(y_true)) ** 2))


def _rmsle(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Root mean squared logarithmic error regression loss."""
    # Requires y_true > -1 and y_pred > -1.
    xp = cp.get_array_module(y_true)
    return float(xp.sqrt(xp.mean((xp.log1p(y_pred) - xp.log1p(y_true)) ** 2)))


def _poisson_deviance(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Mean Poisson deviance regression loss."""
    # Requires y_true >= 0 and y_pred > 0. Convention: 0 * log(0) = 0.
    xp = cp.get_array_module(y_true)
    pos = y_true > 0
    y_log = xp.where(pos, y_true * xp.log(xp.where(pos, y_true / y_pred, 1.0)), 0.0)
    return float(2.0 * xp.mean(y_log - (y_true - y_pred)))


def _gamma_deviance(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Mean Gamma deviance regression loss."""
    # Requires y_true > 0 and y_pred > 0.
    xp = cp.get_array_module(y_true)
    return float(2.0 * xp.mean(xp.log(y_pred / y_true) + y_true / y_pred - 1.0))

# --- Scores (higher is better) -------------------------------------------

def _r2(y_true: Any, y_pred: Any) -> float:
    """[CuPy] R^2 (coefficient of determination) regression score function."""
    xp = cp.get_array_module(y_true)
    ss_res = xp.sum((y_true - y_pred) ** 2)
    ss_tot = xp.sum((y_true - xp.mean(y_true)) ** 2)
    if ss_tot == 0:  # constant target: sklearn force_finite behaviour
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _explained_variance(y_true: Any, y_pred: Any) -> float:
    """[CuPy] Explained variance regression score function."""
    xp = cp.get_array_module(y_true)
    num = xp.var(y_true - y_pred)
    den = xp.var(y_true)
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(1.0 - num / den)


def _d2_absolute_error(y_true: Any, y_pred: Any) -> float:
    """[CuPy] D^2 regression score function, fraction of absolute error explained."""
    xp = cp.get_array_module(y_true)
    num = xp.sum(xp.abs(y_true - y_pred))
    den = xp.sum(xp.abs(y_true - xp.median(y_true)))  # baseline: constant median predictor
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(1.0 - num / den)

# --- Registry ------------------------------------------------------------

class MetricSpec(NamedTuple):
    fn: MetricFn
    greater_is_better: bool
    sklearn_scoring: str  # string accepted by RandomizedSearchCV(scoring=...)


_SPECS: dict[str, MetricSpec] = {
    "r2": MetricSpec(_r2, True, "r2"),
    "explained_variance": MetricSpec(_explained_variance, True, "explained_variance"),
    "d2_absolute_error": MetricSpec(_d2_absolute_error, True, "d2_absolute_error_score"),
    "rmse": MetricSpec(_rmse, False, "neg_root_mean_squared_error"),
    "mse": MetricSpec(_mse, False, "neg_mean_squared_error"),
    "mae": MetricSpec(_mae, False, "neg_mean_absolute_error"),
    "median_ae": MetricSpec(_median_ae, False, "neg_median_absolute_error"),
    "max_error": MetricSpec(_max_error, False, "neg_max_error"),
    "mape": MetricSpec(_mape, False, "neg_mean_absolute_percentage_error"),
    "msle": MetricSpec(_msle, False, "neg_mean_squared_log_error"),
    "rmsle": MetricSpec(_rmsle, False, "neg_root_mean_squared_log_error"),
    "poisson_deviance": MetricSpec(_poisson_deviance, False, "neg_mean_poisson_deviance"),
    "gamma_deviance": MetricSpec(_gamma_deviance, False, "neg_mean_gamma_deviance"),
}

# name -> (function, greater_is_better)
METRICS: dict[str, tuple[MetricFn, bool]] = {
    name: (spec.fn, spec.greater_is_better) for name, spec in _SPECS.items()
}

# name -> sklearn scoring string (same keys as METRICS by construction)
SKLEARN_SCORING: dict[str, str] = {
    name: spec.sklearn_scoring for name, spec in _SPECS.items()
}

def check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {sorted(METRICS)}")