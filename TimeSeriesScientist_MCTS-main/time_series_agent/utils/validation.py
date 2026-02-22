"""
Validation strategies for the time series pipeline.

Two differentiated strategies by design:

Stage 1 — MCTS Simulation  (speed-first)
    last_block_split:  sub_train = data[:-H], sub_val = data[-H:]
    One fit, one predict, one score.  **No CV whatsoever.**

Stage 2 — Tuning Agent  (robustness + OOF for ensemble)
    rolling_cv_with_oof:  sklearn TimeSeriesSplit(n_splits=3~5)
    Train on each fold, predict on each validation window,
    concatenate all fold predictions → oof_predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CV_SPLITS = 3


# ---------------------------------------------------------------------------
# Stage 1: Last-Block Validation (MCTS fast reward)
# ---------------------------------------------------------------------------

def last_block_split(
    series: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split *train_data* into sub-train / sub-val for one-shot evaluation.

    Parameters
    ----------
    series : 1-D array
        The training series.  **Must not contain test data** to avoid leakage.
    horizon : int
        Forecast horizon — the last ``horizon`` time-steps become sub-val.

    Returns
    -------
    sub_train, sub_val : np.ndarray
        ``sub_train = series[: N-horizon]``, ``sub_val = series[N-horizon :]``
    """
    series = np.asarray(series).flatten()
    if len(series) <= horizon:
        # Edge case: not enough data — fall back to half-half
        mid = max(1, len(series) // 2)
        return series[:mid], series[mid:]
    return series[:-horizon], series[-horizon:]


# ---------------------------------------------------------------------------
# Stage 2: Rolling Time Series Cross-Validation with OOF
# ---------------------------------------------------------------------------

def rolling_cv_with_oof(
    series: np.ndarray,
    horizon: int,
    model_fn: Callable[[Dict[str, Any], Dict[str, Any], int], Any],
    model_params: Dict[str, Any],
    n_splits: int = _DEFAULT_CV_SPLITS,
    enriched_data: Any = None,
    scaler: Any = None,
    series_original: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Rolling Time-Series CV that produces Out-of-Fold predictions.

    Uses ``sklearn.model_selection.TimeSeriesSplit(n_splits)``.

    For each fold:
        1. Fit *model_fn* on the fold's training portion (scaled series).
        2. Predict ``min(len(val), horizon)`` steps.
        3. If *scaler* and *series_original* are provided, metrics and stored
           predictions are in **original scale** (inverse_transform applied).

    Parameters
    ----------
    series : 1-D array
        Full training series in scaled space (no test data).
    horizon : int
        Forecast horizon.
    model_fn : callable
        ``model_fn(data_dict, params, pred_len) -> predictions``
        where ``data_dict = {"value": train_array}``.
    model_params : dict
        Hyper-parameters forwarded to *model_fn*.
    n_splits : int
        Number of rolling splits (default 3).
    enriched_data : dict or None
        If provided, a dict with ``"value"`` plus extra L2-feature columns.
    scaler : object or None
        If provided, must have ``inverse_transform(arr)``; used to convert
        predictions to original scale before computing metrics and storing OOF.
    series_original : 1-D array or None
        If provided with *scaler*, same length as *series*; used as ground truth
        (original scale) for metric computation.

    Returns
    -------
    avg_metrics : dict
        ``{"mse": ..., "mae": ..., "mape": ...}`` in **original scale** when
        scaler/series_original given.
    oof_predictions : np.ndarray
        Predictions at validation indices (original scale when scaler given).
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    series = np.asarray(series, dtype=float).flatten()
    n = len(series)
    use_original_scale = (
        scaler is not None
        and series_original is not None
        and len(np.asarray(series_original).flatten()) == n
        and hasattr(scaler, "inverse_transform")
    )
    if use_original_scale:
        series_original = np.asarray(series_original, dtype=float).flatten()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []
    oof_preds = np.full(n, np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(series)):
        train_vals = series[train_idx]
        val_vals = series[val_idx]
        pred_len = min(len(val_vals), horizon)

        if len(train_vals) < 2 or pred_len < 1:
            continue

        try:
            train_dict: Dict[str, Any] = {"value": train_vals}
            if isinstance(enriched_data, dict):
                for k, v in enriched_data.items():
                    if k in ("value", "value_original", "scaler"):
                        continue
                    arr = np.asarray(v).flatten()
                    if len(arr) == n:
                        train_dict[k] = arr[train_idx]

            preds = model_fn(train_dict, model_params, pred_len)
            preds = np.asarray(preds, dtype=float).flatten()[:pred_len]
            if use_original_scale:
                preds = scaler.inverse_transform(preds)
                actuals = series_original[val_idx[:pred_len]]
            else:
                actuals = val_vals[:pred_len]

            mse = float(mean_squared_error(actuals, preds))
            mae = float(mean_absolute_error(actuals, preds))
            nonzero = actuals != 0
            if nonzero.any():
                mape = float(
                    np.mean(np.abs((actuals[nonzero] - preds[nonzero]) / actuals[nonzero])) * 100
                )
            else:
                mape = 0.0
            fold_metrics.append({"mse": mse, "mae": mae, "mape": mape})
            oof_preds[val_idx[:pred_len]] = preds
        except Exception as exc:
            logger.warning("rolling_cv fold %d failed: %s", fold_idx, exc)
            fold_metrics.append(
                {"mse": float("inf"), "mae": float("inf"), "mape": float("inf")}
            )

    if fold_metrics:
        avg_metrics = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in ("mse", "mae", "mape")
        }
    else:
        avg_metrics = {"mse": float("inf"), "mae": float("inf"), "mape": float("inf")}

    return avg_metrics, oof_preds
