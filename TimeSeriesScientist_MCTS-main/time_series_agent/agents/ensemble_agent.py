"""
Ensemble Agent (Stage 3 of the funnel pipeline).

Receives OOF (out-of-fold) or validation predictions from multiple tuned models,
computes optimal fusion weights via Greedy Forward Selection (Caruana et al.)
or Ridge stacking, and returns the final ensemble prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


def _as_array(x: Union[List[float], np.ndarray]) -> np.ndarray:
    return np.asarray(x, dtype=float).flatten()


def _metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def _metric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true != 0, y_true, 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def greedy_ensemble(
    predictions_dict: Dict[str, Union[List[float], np.ndarray]],
    y_true: Union[List[float], np.ndarray],
    metric: str = "mse",
    method: Literal["greedy", "ridge"] = "greedy",
    ridge_alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute ensemble weights and final prediction from multiple model OOF predictions.

    Inputs:
        predictions_dict: {model_name: array of OOF/validation predictions}
        y_true: ground truth for the same indices
        metric: "mse" (minimize) or "mae"
        method: "greedy" (Caruana forward selection) or "ridge" (Ridge regression stacking)
        ridge_alpha: regularization for Ridge when method="ridge"

    Outputs:
        ensemble_pred: final combined prediction array
        info: dict with "weights", "selected_models" (for greedy), "method"
    """
    y_true = _as_array(y_true)
    names = list(predictions_dict.keys())
    if not names:
        return y_true * np.nan, {"weights": {}, "selected_models": [], "method": method}

    preds = {k: _as_array(v) for k, v in predictions_dict.items()}
    n = len(y_true)
    for k, v in preds.items():
        if len(v) != n:
            raise ValueError(f"Model {k} predictions length {len(v)} != y_true length {n}")

    if metric == "mae":
        def score_fn(yt: np.ndarray, yp: np.ndarray) -> float:
            return float(np.mean(np.abs(yt - yp)))
    elif metric == "mape":
        score_fn = _metric_mape
    else:
        score_fn = _metric_mse

    if method == "ridge":
        return _ensemble_ridge(preds, y_true, ridge_alpha, score_fn)
    return _ensemble_greedy(preds, y_true, score_fn)


def _ensemble_ridge(
    preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    alpha: float,
    score_fn: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stacking: Ridge regression on model predictions -> weights."""
    from sklearn.linear_model import Ridge
    names = list(preds.keys())
    X = np.column_stack([preds[k] for k in names])
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X, y_true)
    coef = reg.coef_
    intercept = float(reg.intercept_)
    weights = {names[i]: float(coef[i]) for i in range(len(names))}
    ensemble_pred = reg.predict(X)
    info = {
        "weights": weights,
        "intercept": intercept,
        "selected_models": names,
        "method": "ridge",
    }
    return ensemble_pred, info


def _ensemble_greedy(
    preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    score_fn: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Greedy forward selection (Caruana-style), but start from the best single model.
    This avoids the unstable 'start from zeros' behavior that can derail search/metrics.
    """
    names = list(preds.keys())
    if len(names) == 0:
        info = {"weights": {}, "selected_models": [], "method": "greedy"}
        return np.zeros_like(y_true), info

    # ---- Initialize with best single model ----
    best_single = None
    best_single_score = float("inf")
    for k in names:
        sc = score_fn(y_true, preds[k])
        if sc < best_single_score:
            best_single_score = sc
            best_single = k

    # Safety: should not happen, but keep it robust
    if best_single is None:
        info = {"weights": {}, "selected_models": [], "method": "greedy"}
        return np.zeros_like(y_true), info

    selected: List[str] = [best_single]
    weights: Dict[str, float] = {best_single: 1.0}
    best_pred = preds[best_single].copy()
    best_score = best_single_score

    # ---- Forward add more models ----
    for _ in range(len(names) - 1):
        best_candidate = None
        best_new_score = best_score
        best_w = 0.0

        for cand in names:
            if cand in selected:
                continue
            # line search w: 0, 0.1, ..., 1.0
            for w in np.linspace(0, 1, 11):
                new_pred = best_pred + w * (preds[cand] - best_pred)
                sc = score_fn(y_true, new_pred)
                if sc < best_new_score - 1e-12:
                    best_new_score = sc
                    best_candidate = cand
                    best_w = float(w)

        # no improvement -> stop
        if best_candidate is None:
            break

        selected.append(best_candidate)
        weights[best_candidate] = best_w
        best_pred = best_pred + best_w * (preds[best_candidate] - best_pred)
        best_score = best_new_score

    info = {"weights": weights, "selected_models": selected, "method": "greedy"}
    return best_pred, info


def apply_ensemble_weights(
    predictions_dict: Dict[str, Union[List[float], np.ndarray]],
    ensemble_info: Dict[str, Any],
) -> np.ndarray:
    """Apply pre-fitted ensemble weights to new predictions (no refit, no leakage).

    Use when ensemble was fitted on validation holdout and we need predictions
    on held-out test data.
    """
    weights = ensemble_info.get("weights", {})
    selected = ensemble_info.get("selected_models", [])
    method = ensemble_info.get("method", "greedy")
    preds = {k: _as_array(v) for k, v in predictions_dict.items()}
    available = [m for m in selected if m in preds]
    if not available:
        names = list(preds.keys())
        if not names:
            return np.array([])
        first = names[0]
        return preds[first] * np.nan
    n = len(preds[available[0]])
    if method == "ridge":
        intercept = ensemble_info.get("intercept", 0.0)
        coef = np.array([weights.get(m, 0) for m in available])
        X = np.column_stack([preds[m] for m in available])
        return (X @ coef + intercept).flatten()
    best_pred = np.zeros(n)
    for m in available:
        w = weights.get(m, 0)
        best_pred = best_pred + w * (preds[m] - best_pred)
    return best_pred


class EnsembleAgent:
    """
    Stage 3 agent: takes tuned model OOF predictions and produces ensemble forecast.
    """

    def __init__(
        self,
        method: Literal["greedy", "ridge"] = "greedy",
        ridge_alpha: float = 1.0,
        metric: str = "mse",
    ):
        self.method = method
        self.ridge_alpha = ridge_alpha
        self.metric = metric

    def run(
        self,
        predictions_dict: Dict[str, Union[List[float], np.ndarray]],
        y_true: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute ensemble prediction and metadata.

        Returns:
            ensemble_pred: 1d array
            info: weights, selected_models, method
        """
        return greedy_ensemble(
            predictions_dict,
            y_true,
            metric=self.metric,
            method=self.method,
            ridge_alpha=self.ridge_alpha,
        )

    def apply_weights(
        self,
        predictions_dict: Dict[str, Union[List[float], np.ndarray]],
        ensemble_info: Dict[str, Any],
    ) -> np.ndarray:
        """Apply pre-fitted weights to new predictions (for test set, no leakage)."""
        return apply_ensemble_weights(predictions_dict, ensemble_info)
