"""
Ensemble Agent (Stage 3 of the funnel pipeline).

Receives OOF (out-of-fold) or validation predictions from multiple tuned models,
computes ensemble weights / selection, and returns the final ensemble prediction.
Supports several methods plus an "auto" selector on validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np


EnsembleMethod = Literal["greedy", "ridge", "mean", "median", "single_best", "auto"]


def _as_array(x: Union[List[float], np.ndarray]) -> np.ndarray:
    return np.asarray(x, dtype=float).flatten()


def _metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _metric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mean_abs = float(np.mean(np.abs(yt))) if len(yt) else 0.0
    eps = max(1e-8, 1e-3 * max(1.0, mean_abs))
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def _get_score_fn(metric: str):
    if metric == "mae":
        return _metric_mae
    if metric == "mape":
        return _metric_mape
    return _metric_mse


def _single_best(preds: Dict[str, np.ndarray], y_true: np.ndarray, score_fn: Any):
    best_name, best_score = None, float("inf")
    for k, v in preds.items():
        sc = score_fn(y_true, v)
        if sc < best_score:
            best_name, best_score = k, sc
    if best_name is None:
        return y_true * np.nan, {"weights": {}, "selected_models": [], "method": "single_best"}
    return preds[best_name].copy(), {
        "weights": {best_name: 1.0},
        "selected_models": [best_name],
        "method": "single_best",
    }


def _mean_ensemble(preds: Dict[str, np.ndarray]):
    names = list(preds.keys())
    if not names:
        return np.array([]), {"weights": {}, "selected_models": [], "method": "mean"}
    X = np.column_stack([preds[k] for k in names])
    pred = np.mean(X, axis=1)
    w = {k: 1.0 / len(names) for k in names}
    return pred, {"weights": w, "selected_models": names, "method": "mean"}


def _median_ensemble(preds: Dict[str, np.ndarray]):
    names = list(preds.keys())
    if not names:
        return np.array([]), {"weights": {}, "selected_models": [], "method": "median"}
    X = np.column_stack([preds[k] for k in names])
    pred = np.median(X, axis=1)
    # weights are only descriptive here
    w = {k: 1.0 / len(names) for k in names}
    return pred, {"weights": w, "selected_models": names, "method": "median"}


def _ensemble_ridge(
    preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    """Greedy forward selection, initialized from best single model."""
    names = list(preds.keys())
    if not names:
        return np.zeros_like(y_true), {"weights": {}, "selected_models": [], "method": "greedy"}

    best_single = None
    best_single_score = float("inf")
    for k in names:
        sc = score_fn(y_true, preds[k])
        if sc < best_single_score:
            best_single_score = sc
            best_single = k

    if best_single is None:
        return np.zeros_like(y_true), {"weights": {}, "selected_models": [], "method": "greedy"}

    selected: List[str] = [best_single]
    weights: Dict[str, float] = {best_single: 1.0}
    best_pred = preds[best_single].copy()
    best_score = best_single_score

    for _ in range(len(names) - 1):
        best_candidate = None
        best_new_score = best_score
        best_w = 0.0
        for cand in names:
            if cand in selected:
                continue
            for w in np.linspace(0.0, 1.0, 11):
                new_pred = best_pred + w * (preds[cand] - best_pred)
                sc = score_fn(y_true, new_pred)
                if sc < best_new_score - 1e-12:
                    best_new_score = sc
                    best_candidate = cand
                    best_w = float(w)
        if best_candidate is None:
            break
        selected.append(best_candidate)
        weights[best_candidate] = best_w
        best_pred = best_pred + best_w * (preds[best_candidate] - best_pred)
        best_score = best_new_score

    return best_pred, {"weights": weights, "selected_models": selected, "method": "greedy"}


def greedy_ensemble(
    predictions_dict: Dict[str, Union[List[float], np.ndarray]],
    y_true: Union[List[float], np.ndarray],
    metric: str = "mse",
    method: EnsembleMethod = "greedy",
    ridge_alpha: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_true = _as_array(y_true)
    names = list(predictions_dict.keys())
    if not names:
        return y_true * np.nan, {"weights": {}, "selected_models": [], "method": method}

    preds = {k: _as_array(v) for k, v in predictions_dict.items()}
    n = len(y_true)
    for k, v in preds.items():
        if len(v) != n:
            raise ValueError(f"Model {k} predictions length {len(v)} != y_true length {n}")

    score_fn = _get_score_fn(metric)

    if method == "single_best":
        return _single_best(preds, y_true, score_fn)
    if method == "mean":
        return _mean_ensemble(preds)
    if method == "median":
        return _median_ensemble(preds)
    if method == "ridge":
        return _ensemble_ridge(preds, y_true, ridge_alpha)
    if method == "auto":
        candidates = {}
        infos = {}
        for meth in ("single_best", "mean", "median", "greedy", "ridge"):
            try:
                pred, info = greedy_ensemble(preds, y_true, metric=metric, method=meth, ridge_alpha=ridge_alpha)
                candidates[meth] = pred
                infos[meth] = info
            except Exception:
                continue
        if not candidates:
            return _single_best(preds, y_true, score_fn)
        best_method, best_score = None, float("inf")
        for meth, pred in candidates.items():
            sc = score_fn(y_true, pred)
            if sc < best_score:
                best_method, best_score = meth, sc
        info = infos[best_method]
        info = dict(info)
        info["method"] = f"auto->{best_method}"
        return candidates[best_method], info
    return _ensemble_greedy(preds, y_true, score_fn)


def apply_ensemble_weights(
    predictions_dict: Dict[str, Union[List[float], np.ndarray]],
    ensemble_info: Dict[str, Any],
) -> np.ndarray:
    """Apply pre-fitted ensemble weights to new predictions (no refit, no leakage)."""
    weights = ensemble_info.get("weights", {})
    selected = ensemble_info.get("selected_models", [])
    method = ensemble_info.get("method", "greedy")
    base_method = method.split("->")[-1] if isinstance(method, str) and "auto->" in method else method
    preds = {k: _as_array(v) for k, v in predictions_dict.items()}
    available = [m for m in selected if m in preds]
    if not available:
        names = list(preds.keys())
        if not names:
            return np.array([])
        return preds[names[0]] * np.nan
    n = len(preds[available[0]])
    if base_method == "single_best":
        m = available[0]
        return preds[m].copy()
    if base_method == "mean":
        X = np.column_stack([preds[m] for m in available])
        return np.mean(X, axis=1)
    if base_method == "median":
        X = np.column_stack([preds[m] for m in available])
        return np.median(X, axis=1)
    if base_method == "ridge":
        intercept = ensemble_info.get("intercept", 0.0)
        coef = np.array([weights.get(m, 0.0) for m in available], dtype=float)
        X = np.column_stack([preds[m] for m in available])
        return (X @ coef + intercept).flatten()
    # greedy-style progressive blend
    best_pred = np.zeros(n)
    for m in available:
        w = float(weights.get(m, 0.0))
        best_pred = best_pred + w * (preds[m] - best_pred)
    return best_pred


class EnsembleAgent:
    """Stage 3 agent: takes tuned model OOF predictions and produces ensemble forecast."""

    def __init__(
        self,
        method: EnsembleMethod = "auto",
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
        return apply_ensemble_weights(predictions_dict, ensemble_info)
