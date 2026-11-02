import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,           # RMSE (sklearn >= 1.4)
    r2_score,                          # R2
    mean_absolute_error,               # MAE
    mean_absolute_percentage_error,    # MAPE
    median_absolute_error,             # MedAE
    max_error,                         # max_error
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
        "RMSE%": 100.0 * rmse / float(np.mean(y_true)),
    }