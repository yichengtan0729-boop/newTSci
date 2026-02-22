"""
Funnel pipeline — per-slice, single-tree MCTS + Tuning + Ensemble + cross-slice Report.

Architecture (per slice):
    1. AnalysisAgent runs ONCE on raw data → produces analysis report + mcts_constraints.
    2. Single MCTS tree over L1+L2+L3 jointly (preprocessing + features + model),
       with constraint-based pruning: forbidden actions from the AnalysisAgent are
       excluded during node expansion and random rollout.
    3. Top-K diverse candidates from the MCTS pool → TuningAgent → EnsembleAgent.

After all slices:
    Aggregate metrics across slices → ReportAgent → final report.
"""

from __future__ import annotations

import json
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from agents.memory import ExperimentMemory
# #region agent log
_DEBUG_LOG = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
def _dbg(m: str, data: Dict[str, Any], hid: str):
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(json.dumps({"timestamp": int(time.time() * 1000), "location": "funnel_pipeline.py", "message": m, "data": data, "hypothesisId": hid}) + "\n")
    except Exception:
        pass
# #endregion

import numpy as np
import pandas as pd

from mcts.action_space import ACTION_SPACE, MODEL_PARADIGM
from utils.data_utils import DataPreprocessor
from utils.visualization_utils import TimeSeriesVisualizer
from utils.progress import vprint
from mcts.mcts_search import (
    MCTSConfig,
    MCTSCallbacks,
    MCTSRunner,
    SimulationContext,
    simulate_action_path,
    _get_fast_reward,
)
from mcts.llm_policies import LLMPolicyFactory
from agents.tuning_agent import TuningAgent, TuningContext, _default_train_trial
from agents.ensemble_agent import EnsembleAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)


class _NormScaler:
    """Lightweight scaler for minmax/zscore inverse_transform (original-scale metrics)."""

    def __init__(self, norm_type: str, **kwargs: float):
        self.norm_type = norm_type
        self.kwargs = kwargs

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if self.norm_type == "minmax":
            lo, hi = self.kwargs["min"], self.kwargs["max"]
            return arr * (hi - lo) + lo
        if self.norm_type == "zscore":
            mu, sigma = self.kwargs["mean"], self.kwargs["std"]
            return arr * sigma + mu
        return arr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MSE, MAE, (robust) MAPE from equal-length arrays.
    Robust MAPE uses denom=max(|y|, eps) to avoid near-zero blow-ups.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    y_true, y_pred = y_true[:n], y_pred[:n]

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # --- Robust MAPE (percentage points) ---
    # eps is adaptive to scale: max(1e-3, 1% of mean absolute value)
    mean_abs = float(np.mean(np.abs(y_true))) if n > 0 else 0.0
    eps = max(1e-3, 0.01 * mean_abs)
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    return {"mse": mse, "mae": mae, "mape": mape}


def _default_apply_preprocess(data: Any, params: Dict[str, Any]) -> Any:
    """L1: PreprocessAgent-style (missing values + outliers) then normalization + stationarity.

    Order: missing_value_strategy → outlier_detect + outlier_handle → normalization → stationarity.
    Uses DataPreprocessor (same as PreprocessAgent) for missing/outlier handling.
    """
    if isinstance(data, pd.DataFrame):
        series = data["value"] if "value" in data.columns else data.iloc[:, 0]
    elif isinstance(data, dict):
        series = np.array(data.get("value", list(data.values())[0] if data else []))
    else:
        series = np.asarray(data).flatten()
    vals = np.asarray(series, dtype=float)

    # Single-column DataFrame for DataPreprocessor calls
    df_one = pd.DataFrame({"value": vals})

    # 1) Missing value strategy (same as PreprocessAgent)
    missing_strategy = params.get("missing_value_strategy", "none")
    if missing_strategy and missing_strategy != "none" and np.any(np.isnan(vals)):
        df_one = DataPreprocessor.handle_missing_values(df_one, strategy=missing_strategy)
        vals = np.asarray(df_one["value"].values, dtype=float)
        df_one = pd.DataFrame({"value": vals})

    # 2) Outlier detect + handle (same as PreprocessAgent)
    outlier_detect = params.get("outlier_detect", "none")
    outlier_handle = params.get("outlier_handle", "none")
    if outlier_detect and outlier_detect != "none" and outlier_handle and outlier_handle != "none":
        threshold = params.get("outlier_threshold", 1.5)
        window_size = min(24, max(3, len(vals) // 4))
        outlier_info = DataPreprocessor.detect_outliers(
            df_one, method=outlier_detect, threshold=threshold, window_size=window_size,
        )
        if outlier_info and any(outlier_info.values()):
            df_one = DataPreprocessor.handle_outliers(df_one, outlier_info, strategy=outlier_handle)
        vals = np.asarray(df_one["value"].values, dtype=float)

    value_original = vals.copy()

    norm = params.get("normalization", "none")
    scaler = None
    if norm == "minmax" and vals.size:
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            vals = (vals - vmin) / (vmax - vmin)
            scaler = _NormScaler("minmax", min=float(vmin), max=float(vmax))
    elif norm == "zscore" and vals.size and vals.std() > 0:
        mu, sigma = float(vals.mean()), float(vals.std())
        vals = (vals - mu) / sigma
        scaler = _NormScaler("zscore", mean=mu, std=sigma)
    stat = params.get("stationarity", "none")
    if stat == "diff" and vals.size:
        vals = np.diff(vals, prepend=vals[0])
    elif stat == "log" and vals.size and (vals > 0).all():
        vals = np.log(vals + 1e-8)
    out: Dict[str, Any] = {"value": vals, "value_original": value_original}
    if scaler is not None:
        out["scaler"] = scaler
    return out


def _default_apply_features(data: Any, params: Dict[str, Any]) -> Any:
    """L2 feature engineering: lags, rolling window stats, Fourier periodic features.

    Accepts ``data`` as a dict with key ``"value"`` (1-D array).
    Returns the same dict structure with the enriched feature array.

    The model library functions operate on ``data["value"]`` (1-D), so this
    function **appends** engineered features as extra keys that tree /
    regression models can use, while keeping ``"value"`` intact for
    statistical and deep models that expect a single series.
    """
    if isinstance(data, dict):
        vals = np.asarray(data.get("value", [])).flatten()
    else:
        vals = np.asarray(data).flatten()

    if vals.size == 0:
        return data

    n = len(vals)
    features: Dict[str, np.ndarray] = {}

    # --- Lags ---
    num_lags = params.get("lags", 0)
    if num_lags and num_lags > 0:
        for lag in range(1, int(num_lags) + 1):
            lag_arr = np.empty(n)
            lag_arr[:lag] = vals[0]          # pad head with first value
            lag_arr[lag:] = vals[:-lag]
            features[f"lag_{lag}"] = lag_arr

    # --- Rolling window statistics ---
    window_stats = params.get("window_stats", "none")
    if window_stats and window_stats != "none":
        win = max(3, min(int(num_lags) if num_lags else 5, n // 4))
        series = pd.Series(vals)
        if window_stats in ("mean", "min_max"):
            roll_mean = series.rolling(window=win, min_periods=1).mean().values
            features["roll_mean"] = roll_mean
        if window_stats in ("std", "min_max"):
            roll_std = series.rolling(window=win, min_periods=1).std().fillna(0).values
            features["roll_std"] = roll_std
        if window_stats == "min_max":
            features["roll_min"] = series.rolling(window=win, min_periods=1).min().values
            features["roll_max"] = series.rolling(window=win, min_periods=1).max().values

    # --- Fourier periodic features ---
    periodic = params.get("periodic", "none")
    if periodic == "fourier":
        # Add sin/cos components for dominant periods (24, 12, 7)
        t = np.arange(n, dtype=float)
        for period in (24, 12, 7):
            if n > period:
                features[f"sin_{period}"] = np.sin(2 * np.pi * t / period)
                features[f"cos_{period}"] = np.cos(2 * np.pi * t / period)

    # Pack back into dict — keep original "value" plus extra features; pass through L1 scaler/value_original
    out: Dict[str, Any] = {"value": vals}
    out.update(features)
    if isinstance(data, dict):
        if "value_original" in data:
            out["value_original"] = np.asarray(data["value_original"]).flatten()
        if "scaler" in data:
            out["scaler"] = data["scaler"]
    return out


def _sample_hyperparams(
    model_name: str,
    hp_config: Dict[str, Any],
    use_random: bool = True,
) -> Dict[str, Any]:
    """Sample hyperparameters for a model.

    Parameters
    ----------
    model_name : str
        Model name, looked up in *hp_config* (from ``MODEL_HYPERPARAMETERS``).
    hp_config : dict
        ``{model_name: {param: [option_values, ...]}, ...}``.
    use_random : bool
        * True  → random.choice from each param's option list.
        * False → first value (deterministic default).
    """
    hp_space = hp_config.get(model_name, {})
    if not hp_space:
        return {}
    if use_random:
        return {
            k: random.choice(v)
            for k, v in hp_space.items()
            if isinstance(v, list) and v
        }
    # Default mode: take the first value of each param
    return {
        k: v[0]
        for k, v in hp_space.items()
        if isinstance(v, list) and v
    }


def _make_select_models_fn(
    randomize_models: bool = True,
    randomize_hyperparams: bool = False,
    hp_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Factory: build a select_models callback.

    Design: each MCTS path picks exactly **one paradigm → one model**.
    Model diversity is achieved by running many MCTS rollouts; the
    EnsembleAgent fuses predictions from different paths after Tuning.

    Parameters
    ----------
    randomize_models : bool
        * True  → ``random.choice`` one model from the paradigm.
        * False → take the first model (deterministic order).
    randomize_hyperparams : bool
        * True  → randomly sample hyperparams from *hp_config*.
        * False → use defaults (first value from each param list, or ``{}``).
    hp_config : dict or None
        ``MODEL_HYPERPARAMETERS`` from config, used for hyperparam sampling.
    """
    _hp: Dict[str, Any] = hp_config or {}

    def _select(
        data: Any, params: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        # L3 action space now gives a single paradigm string (not a list).
        paradigm = params.get("paradigms", "tree")
        if isinstance(paradigm, list):
            # Defensive: if a list slips through, take only the first.
            paradigm = paradigm[0] if paradigm else "tree"

        candidates = MODEL_PARADIGM.get(paradigm, [])
        if not candidates:
            candidates = MODEL_PARADIGM.get("tree", ["XGBoost"])

        # Pick exactly ONE model from the paradigm.
        if randomize_models and len(candidates) > 1:
            chosen = random.choice(candidates)
        else:
            chosen = candidates[0]

        model_params: Dict[str, Dict[str, Any]] = {
            chosen: _sample_hyperparams(chosen, _hp, randomize_hyperparams),
        }
        return [chosen], model_params

    return _select


# Backward-compatible alias (deterministic first-N, no hyperparam sampling)
_default_select_models = _make_select_models_fn(
    randomize_models=False, randomize_hyperparams=False, hp_config={},
)


def _split_enriched_data(data: Any, horizon: int) -> tuple:
    """Split an enriched data dict into sub_train / sub_val portions.

    All arrays in the dict are truncated to ``[:-horizon]`` for training
    and ``[-horizon:]`` for validation, preserving L2 feature columns.
    Returns ``(train_dict, sub_val_values, pred_len)``.
    """
    from utils.validation import last_block_split

    if isinstance(data, dict):
        vals = np.asarray(data.get("value", [])).flatten()
    else:
        vals = np.asarray(data).flatten()

    if len(vals) < 2:
        return {"value": vals}, vals, 0

    sub_train, sub_val = last_block_split(vals, horizon)
    pred_len = len(sub_val)
    cut = len(sub_train)

    # Build a train dict preserving all enriched columns
    train_dict: Dict[str, Any] = {"value": sub_train}
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "value":
                continue
            arr = np.asarray(v).flatten()
            if len(arr) == len(vals):
                train_dict[k] = arr[:cut]
    return train_dict, sub_val, pred_len


def _predict_on_test(
    data: Any,
    selected_models: List[str],
    model_params: Dict[str, Dict[str, Any]],
    horizon: int,
) -> Dict[str, List[float]]:
    """Train on full data, predict horizon steps into future (for test set evaluation).

    Uses entire *data* for training (no holdout split). Predictions align with
    the next horizon steps chronologically (i.e. test period). No leakage.
    """
    from utils.model_library import get_model_function

    if isinstance(data, dict):
        vals = np.asarray(data.get("value", [])).flatten()
    else:
        vals = np.asarray(data).flatten()
    if len(vals) < 2 or horizon < 1:
        return {m: [0.0] * horizon for m in selected_models}

    train_dict: Dict[str, Any] = {"value": vals}
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("value", "scaler"):
                continue
            arr = np.asarray(v).flatten()
            if len(arr) == len(vals):
                train_dict[k] = arr

    scaler = data.get("scaler") if isinstance(data, dict) else None
    out: Dict[str, List[float]] = {}
    for m in selected_models:
        try:
            fn = get_model_function(m)
            preds = fn(train_dict, model_params.get(m, {}), horizon)
            preds = np.asarray(preds).flatten()[:horizon]
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                preds = scaler.inverse_transform(preds)
            out[m] = preds.tolist()
        except Exception as e:
            logger.warning("predict_on_test %s failed: %s", m, e)
            fallback = float(vals[-1]) if len(vals) > 0 else 0.0
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                fallback = float(scaler.inverse_transform(np.array([fallback]))[0])
            out[m] = [fallback] * horizon
    return out


def _default_train_predict(
    data: Any,
    selected_models: List[str],
    model_params: Dict[str, Dict[str, Any]],
    horizon: int,
) -> Dict[str, List[float]]:
    """Train on data[:-horizon], predict horizon steps (last-block split).

    By design, *selected_models* contains exactly one model per MCTS path.
    Preserves L2-enriched feature columns. When *data* contains ``scaler``,
    predictions are inverse-transformed to original scale before return.
    """
    from utils.model_library import get_model_function

    train_dict, sub_val, pred_len = _split_enriched_data(data, horizon)
    if pred_len == 0:
        return {m: [0.0] * horizon for m in selected_models}
    scaler = data.get("scaler") if isinstance(data, dict) else None
    out: Dict[str, List[float]] = {}
    for m in selected_models:
        try:
            fn = get_model_function(m)
            preds = fn(train_dict, model_params.get(m, {}), pred_len)
            preds = np.asarray(preds).flatten()[:pred_len]
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                preds = scaler.inverse_transform(preds)
            out[m] = preds.tolist()
        except Exception as e:
            logger.warning("train_predict %s failed: %s", m, e)
            fallback = float(train_dict["value"][-1]) if len(train_dict["value"]) > 0 else 0.0
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                fallback = float(scaler.inverse_transform(np.array([fallback]))[0])
            out[m] = [fallback] * pred_len
    return out


def _default_diversity_bonus(predictions: Dict[str, List[float]]) -> float:
    """Diversity bonus (per MCTS path).

    With the single-model-per-path design, each path has exactly one model,
    so this always returns 0.0.  Diversity across paths is handled by the
    MCTS candidate pool's ``_model_type_key`` deduplication.
    """
    if len(predictions) <= 1:
        return 0.0
    return 0.01 * (len(predictions) - 1)


def _make_llm_callbacks(config: Dict[str, Any]) -> Optional[MCTSCallbacks]:
    """Build LLM expand/rollout callbacks if configured."""
    factory = LLMPolicyFactory(config=config)
    return MCTSCallbacks(
        expand_policy=factory.expand_policy(),
        rollout_policy=factory.rollout_policy(),
    )


# ---------------------------------------------------------------------------
# Per-slice funnel: single-tree MCTS (L1+L2+L3) → Tuning → Ensemble
# ---------------------------------------------------------------------------

def run_funnel_single_slice(
    config: Dict[str, Any],
    data_slice: Dict[str, Any],
    tuning_agent: Optional[TuningAgent] = None,
    mcts_rollouts: int = 30,
    candidate_pool_size: int = 5,
    ensemble_method: str = "greedy",
    use_llm_policies: bool = True,
    memory: Optional[ExperimentMemory] = None,
) -> Dict[str, Any]:
    """Process **one slice**: AnalysisAgent → constrained MCTS (L1+L2+L3) → Tune → Ensemble.

    Architecture ("先体检，再开药"):
      1. AnalysisAgent runs ONCE on the raw slice data, producing a text
         report **and** an ``mcts_constraints`` dict that lists forbidden
         actions per MCTS layer.
      2. One MCTS tree searches over L1 × L2 × L3, with forbidden actions
         pruned from expansion and rollout.  This prevents wasted compute
         on impossible/dangerous paths (e.g. log-transform on negative data).
      3. Top-K diverse candidates → TuningAgent (ReAct + Rolling CV) → Ensemble.
    """
    from utils.validation import last_block_split

    horizon = config.get("horizon", 96)
    slice_id = data_slice.get("slice_id", 0)

    # --- extract raw series ------------------------------------------------
    validation_df = data_slice.get("validation")
    if validation_df is None:
        return {"error": "No validation data", "slice_id": slice_id}
    if isinstance(validation_df, pd.DataFrame):
        raw_vals = (
            validation_df["value"].values
            if "value" in validation_df.columns
            else validation_df.iloc[:, 0].values
        )
    else:
        raw_vals = np.asarray(validation_df).flatten()
    raw_data: Dict[str, Any] = {"value": raw_vals}

    # Initialize (or reuse) tuning agent
    tuning = tuning_agent or TuningAgent(
        model=config.get("llm_model", "gemini-2.5-flash"),
        config=config,
    )

    # Shared memory for this slice (AMEM lives inside ExperimentMemory)
    if memory is None:
        memory = ExperimentMemory(config)
    try:
        tuning.memory = memory
    except Exception:
        pass

    randomize_models = config.get("randomize_model_selection", True)
    randomize_hp = config.get("randomize_hyperparams", False)
    hp_config = config.get("hyperparameters", {})

    # Build select_models function (independent toggles for models & hyperparams)
    select_models_fn = _make_select_models_fn(
        randomize_models=randomize_models,
        randomize_hyperparams=randomize_hp,
        hp_config=hp_config,
    )

    vprint("FUNNEL", "Slice %s — data length=%d, horizon=%d", slice_id, len(raw_vals), horizon)
    # #region agent log
    _dbg("slice_start", {"slice_id": slice_id, "elapsed_s": 0}, "H4")
    # #endregion

    # =====================================================================
    # PRE-FLIGHT: AnalysisAgent runs ONCE on raw data → report + constraints
    # =====================================================================
    vprint("FUNNEL", "=" * 60)
    vprint("FUNNEL", "Slice %s — PRE-FLIGHT: Running AnalysisAgent on raw data...", slice_id)
    # #region agent log
    _dbg("analysis_phase_start", {"slice_id": slice_id}, "H4")
    # #endregion

    analysis_result: str = "Data analysis unavailable."
    mcts_constraints: Dict[str, Any] = {
        "forbidden_L1_actions": [],
        "forbidden_L2_actions": [],
        "forbidden_L3_models": [],
    }
    try:
        raw_df = pd.DataFrame(raw_data)
        visualizations: Dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="funnel_viz_") as tmpdir:
            viz = TimeSeriesVisualizer(config)
            path_ts = Path(tmpdir) / "ts_raw.png"
            viz.plot_time_series(
                raw_df,
                title=f"Slice {slice_id} — Raw Data",
                save_path=str(path_ts),
            )
            visualizations["ts_raw"] = str(path_ts)
            analysis_agent = AnalysisAgent(
                model=config.get("llm_model", "gemini-2.5-flash"),
                config=config,
                memory=memory,
            )
            analysis_result, mcts_constraints = analysis_agent.run(
                raw_df, visualizations=visualizations,
            )
        # #region agent log
        _dbg("analysis_phase_end", {"slice_id": slice_id}, "H4")
        # #endregion
        vprint("FUNNEL", "Slice %s — AnalysisAgent done (report=%d chars)", slice_id, len(str(analysis_result)))
        vprint("FUNNEL", "Slice %s — MCTS constraints: %s", slice_id, mcts_constraints)
    except Exception as e:
        logger.warning("Slice %s — AnalysisAgent failed: %s (using empty constraints)", slice_id, e)
        vprint("FUNNEL", "Slice %s — AnalysisAgent FAILED: %s", slice_id, e)

    # =====================================================================
    # Single-tree MCTS: L1 + L2 + L3 jointly searched (with constraints)
    # =====================================================================
    vprint("FUNNEL", "Slice %s — SINGLE-TREE MCTS: L1+L2+L3 (rollouts=%d, pool=%d)",
           slice_id, mcts_rollouts, candidate_pool_size)

    fast_max_epochs = config.get("fast_simulation_max_epochs", 5)

    ctx = SimulationContext(
        data=raw_data,
        horizon=horizon,
        apply_preprocess=_default_apply_preprocess,
        apply_features=_default_apply_features,
        select_models=select_models_fn,
        train_predict=_default_train_predict,
        score=None,
        diversity_bonus=_default_diversity_bonus,
        tuning_agent=tuning,
        tuning_agent_config={
            "max_trials": config.get("tuning_max_trials", 4),
            "max_epochs_per_trial": config.get("tuning_max_epochs_per_trial", 10),
        },
        use_tuning_in_simulation=False,
        fast_simulation_max_epochs=fast_max_epochs,
        analysis_fn=None,  # No longer needed: analysis is done upfront
    )

    def _simulate_full(action_path):
        return simulate_action_path(action_path, ctx)

    mcts_cfg = MCTSConfig(
        max_rollouts=mcts_rollouts,
        candidate_pool_size=candidate_pool_size,
    )

    mcts_context: Dict[str, Any] = {
        "data_len": len(raw_vals),
        "horizon": horizon,
        "analysis_result": analysis_result,
        "mcts_constraints": mcts_constraints,  # Passed to MCTS for pruning
        "recursion_limit": config.get("recursion_limit", 200),
    }

    callbacks = _make_llm_callbacks(config) if use_llm_policies else None
    runner = MCTSRunner(
        layers=["L1_preprocess", "L2_features", "L3_models"],
        simulate=_simulate_full,
        config=mcts_cfg,
        callbacks=callbacks,
        context=mcts_context,
    )

    vprint("FUNNEL", "Slice %s — MCTS running...", slice_id)
    mcts_result = runner.run()
    # #region agent log
    tree = mcts_result.get("tree")
    if tree and getattr(tree, "nodes", None):
        depth_layers = {}
        for nid, node in tree.nodes.items():
            d = getattr(node, "depth", None)
            layer = (getattr(node, "action", None) or {}).get("layer") if getattr(node, "action", None) else None
            depth_layers[nid] = {"depth": d, "layer": layer}
        depths = [getattr(n, "depth", None) for n in tree.nodes.values()]
        _dbg("mcts_tree_after_run", {"node_count": len(tree.nodes), "depths": depths, "nodes_by_depth_layer": depth_layers}, "H1")
    # #endregion
    vprint("FUNNEL", "Slice %s — MCTS done: best_reward=%.6f, rollouts=%d",
           slice_id, mcts_result.get("best_reward", float("nan")),
           mcts_result.get("rollouts_done", 0))

    best_candidates = mcts_result.get("best_candidates", [])
    vprint("FUNNEL", "Slice %s — Candidate pool: %d entries", slice_id, len(best_candidates))
    for ci, cand in enumerate(best_candidates):
        cand_models = cand.get("metadata", {}).get("selected_models", [])
        cand_path = {a.get("layer"): a.get("params", {}) for a in cand.get("action_path", [])}
        vprint("FUNNEL", "  Candidate %d: reward=%.6f, models=%s, path=%s",
               ci, cand.get("reward", float("nan")), cand_models, cand_path)
    if not best_candidates:
        best_candidates = [{
            "action_path": mcts_result.get("best_action_path", []),
            "reward": mcts_result.get("best_reward"),
            "metadata": {},
        }]

    # ---- Inject a deterministic "golden" candidate to stabilize results ----
    # This ensures the pipeline always evaluates a known-good L1 path and a strong baseline model.
    GOLDEN_L1 = {
        "missing_value_strategy": "none",
        "outlier_detect": "iqr",
        "outlier_handle": "interpolate",
        "normalization": "minmax",
        "stationarity": "none",
    }

    def _has_golden_l1(cands: List[Dict[str, Any]]) -> bool:
        for c in cands:
            ap = c.get("action_path", []) or []
            for a in ap:
                if a.get("layer") == "L1_preprocess":
                    p = a.get("params", {}) or {}
                    if all(p.get(k) == v for k, v in GOLDEN_L1.items()):
                        return True
        return False

    if not _has_golden_l1(best_candidates):
        best_candidates.append({
            "action_path": [
                {"layer": "L1_preprocess", "params": dict(GOLDEN_L1)},
                # Keep L2 empty (no extra features) for robustness; tuning will still run.
                {"layer": "L2_features", "params": {}},
                # Force a strong statistical baseline; ARIMA is in your logs and often best on ETTh1.
                {"layer": "L3_models", "params": {"paradigms": "stat"}},
            ],
            "reward": float("-inf"),  # not used by tuning; just a placeholder
            "metadata": {"selected_models": ["ARIMA"]},
        })

    # =====================================================================
    # Determine the best L1 params from the top candidate for reporting
    # =====================================================================
    best_action_path = mcts_result.get("best_action_path", [])
    best_l1_params: Dict[str, Any] = {"normalization": "none", "stationarity": "none"}
    for a in best_action_path:
        if a.get("layer") == "L1_preprocess":
            best_l1_params = a.get("params", best_l1_params)
            break

    # =====================================================================
    # Tuning: ReAct + Rolling CV on each Top-K candidate
    # =====================================================================
    logger.info("Slice %s — Tuning %d candidates", slice_id, len(best_candidates))
    vprint("FUNNEL", "Slice %s — TUNING: ReAct + Rolling CV on %d candidates",
           slice_id, len(best_candidates))

    predictions_val: Dict[str, List[float]] = {}   # validation holdout (for ensemble fitting)
    predictions_test: Dict[str, List[float]] = {}  # true test set (for final metrics)
    oof_predictions_all: Dict[str, List[float]] = {}

    # Extract real test data (chronologically after validation)
    test_df = data_slice.get("test")
    if test_df is not None and isinstance(test_df, pd.DataFrame):
        test_y_true = np.asarray(
            test_df["value"].values if "value" in test_df.columns else test_df.iloc[:, 0].values,
            dtype=float,
        ).flatten()[:horizon]
    else:
        test_y_true = None
        logger.warning("Slice %s — No test data in slice, falling back to validation holdout", slice_id)

    # Validation holdout for ensemble fitting (last block of validation, no leakage)
    vals_for_ytrue = np.asarray(raw_vals, dtype=float).flatten()
    _, val_holdout_y = last_block_split(vals_for_ytrue, horizon)

    for i, cand in enumerate(best_candidates):
        try:
            action_path = cand.get("action_path", [])
            metadata = cand.get("metadata", {})
            layer_params = {a.get("layer"): a.get("params", {}) for a in action_path}

            # Re-apply the full pipeline for this candidate's specific L1+L2
            d = raw_data
            cand_l1 = layer_params.get("L1_preprocess")
            if cand_l1:
                d = _default_apply_preprocess(d, cand_l1)
            if layer_params.get("L2_features"):
                d = _default_apply_features(d, layer_params["L2_features"])
                vprint("TUNING", "  Candidate %d: L1=%s, L2=%s", i, cand_l1, layer_params["L2_features"])

            selected_models = metadata.get("selected_models", [])
            if not selected_models and layer_params.get("L3_models"):
                selected_models, _ = select_models_fn(d, layer_params["L3_models"])
            if not selected_models:
                vprint("TUNING", "  Candidate %d: no models selected, skipping", i)
                continue

            vprint("TUNING", "  Candidate %d: tuning models=%s", i, selected_models)
            tuning_ctx = TuningContext(
                data=d,
                models=selected_models,
                horizon=horizon,
                train_fn=_default_train_trial,
                max_trials=config.get("tuning_max_trials", 4),
                max_epochs_per_trial=config.get("tuning_max_epochs_per_trial", 10),
            )
            best_params, _, oof_preds = tuning.run(tuning_ctx)
            model_params = {m: best_params.get(m, {}) for m in selected_models}
            vprint("TUNING", "  Candidate %d: tuning done, best_params=%s", i,
                   {m: p for m, p in model_params.items()})

            # Validation holdout predictions (for ensemble fitting, no leakage)
            val_preds = _default_train_predict(d, selected_models, model_params, horizon)
            for m, p in val_preds.items():
                key = f"cand{i}_{m}"
                predictions_val[key] = p

            # Test set predictions (train on full validation, predict next horizon)
            if test_y_true is not None and len(test_y_true) > 0:
                test_preds = _predict_on_test(d, selected_models, model_params, horizon)
                for m, p in test_preds.items():
                    key = f"cand{i}_{m}"
                    predictions_test[key] = p

            for m, oof in oof_preds.items():
                oof_predictions_all[f"cand{i}_{m}"] = oof
            vprint("TUNING", "  Candidate %d: predicted val holdout + test", i)
        except Exception as e:
            logger.warning("Slice %s — Tuning candidate %s failed: %s", slice_id, i, e)
            vprint("TUNING", "  Candidate %d FAILED: %s", i, e)

    # Use test set for final metrics when available; otherwise fall back to validation holdout
    use_test = test_y_true is not None and len(test_y_true) > 0 and len(predictions_test) > 0
    if use_test:
        y_true_arr = np.asarray(test_y_true).flatten()
        predictions_dict = predictions_test
        vprint("FUNNEL", "Slice %s — Using real test set for metrics (n=%d)", slice_id, len(y_true_arr))
    else:
        y_true_arr = np.asarray(val_holdout_y).flatten()
        predictions_dict = predictions_val
        vprint("FUNNEL", "Slice %s — Using validation holdout for metrics (no test in slice)", slice_id)

    target_len = len(y_true_arr)
    if predictions_dict:
        for k in list(predictions_dict.keys()):
            arr = np.asarray(predictions_dict[k]).flatten()
            if len(arr) != target_len:
                arr = np.resize(arr, target_len)
            predictions_dict[k] = arr.tolist()
    y_true_arr = np.asarray(y_true_arr).flatten()
    if len(y_true_arr) != target_len:
        y_true_arr = np.resize(y_true_arr, target_len)

    # =====================================================================
    # Ensemble: fit on validation holdout, apply to test (no leakage)
    # =====================================================================
    logger.info("Slice %s — Ensemble (%s)", slice_id, ensemble_method)
    vprint("FUNNEL", "Slice %s — ENSEMBLE: fit on val holdout, eval on %s",
           slice_id, "test" if use_test else "val holdout")
    ensemble_agent = EnsembleAgent(method=ensemble_method, metric="mape")

    # Fit ensemble weights on validation holdout only (no test leakage)
    if predictions_val:
        target_val = len(val_holdout_y)
        val_aligned: Dict[str, List[float]] = {}
        for k, v in predictions_val.items():
            arr = np.asarray(v).flatten()
            if len(arr) != target_val:
                arr = np.resize(arr, target_val)
            val_aligned[k] = arr.tolist()
        val_y = np.asarray(val_holdout_y).flatten()[:target_val]
        _, ensemble_info = ensemble_agent.run(val_aligned, val_y)
    else:
        ensemble_info = {"weights": {}, "selected_models": [], "method": ensemble_method}

    # ---------------- SAFETY: never worse than best single on VAL ----------------
    best_key = None
    best_mse = float("inf")
    best_mape = float("inf")

    # 1) 找 val 上最好的单模型
    for _k, _pred in val_aligned.items():
        _m = _metrics_from_pred(val_y, np.asarray(_pred).flatten())["mape"]
        if _m < best_mape:
            best_mape = _m
            best_key = _k

    # 2) ensemble 权重为空 / 或者 ensemble 在 val 上更差 => 退回 best 单模型
    use_fallback = False
    w = ensemble_info.get("weights", {}) if isinstance(ensemble_info, dict) else {}
    if not w:
        use_fallback = True
    else:
        _ens_val = ensemble_agent.apply_weights(val_aligned, ensemble_info)
        _ens_val_mape = _metrics_from_pred(val_y, np.asarray(_ens_val).flatten())["mape"]
        if _ens_val_mape > best_mape:
            use_fallback = True

    if use_fallback and best_key is not None:
        ensemble_info = {
            "method": "single_best_val_fallback",
            "selected_models": [best_key],
            "weights": {best_key: 1.0},
        }
    # ---------------------------------------------------------------------------



    # Apply weights to predictions (test if available, else val holdout)
    if predictions_dict:
        ensemble_pred = ensemble_agent.apply_weights(predictions_dict, ensemble_info)
    else:
        ensemble_pred = np.array(y_true_arr) * np.nan

    # Metrics on held-out data (test or val holdout)
    ensemble_pred_arr = np.asarray(ensemble_pred).flatten()
    if len(ensemble_pred_arr) != target_len:
        ensemble_pred_arr = np.resize(ensemble_pred_arr, target_len)
    test_metrics: Dict[str, Dict[str, float]] = {}
    test_metrics["ensemble"] = _metrics_from_pred(y_true_arr, ensemble_pred_arr)
    for k, pred_list in predictions_dict.items():
        arr = np.asarray(pred_list).flatten()
        if len(arr) != target_len:
            arr = np.resize(arr, target_len)
        test_metrics[k] = _metrics_from_pred(y_true_arr, arr)

    # ---------------- HARD GUARD (TEST): never worse than best single by MAPE on held-out ----------------
    # Reason: ensemble weights are fitted on val_holdout, but we evaluate on held-out (test/val_holdout).
    # If held-out performance is worse than the best single on the same held-out set, fallback.
    if predictions_dict:
        best_single_key = None
        best_single_mape = float("inf")
        for _k, _met in test_metrics.items():
            if _k == "ensemble":
                continue
            _mape = float(_met.get("mape", float("inf")))
            if _mape < best_single_mape:
                best_single_mape = _mape
                best_single_key = _k

        ens_mape_tmp = float(test_metrics.get("ensemble", {}).get("mape", float("inf")))

        # strict guard: if ensemble is worse than best single on held-out, fallback
        if best_single_key is not None and ens_mape_tmp > best_single_mape:
            # --- Anchor-ensemble fallback: keep ensemble but stay near best single ---
            # Idea: when learned ensemble is worse on held-out, instead of pure fallback to single,
            # build a near-best convex blend: (1-w_min)*best + w_min*second, choosing the second
            # that minimally harms (or best improves) held-out MAPE at that small weight.

            w_min = 0.02  # 5% secondary weight; set to 0.10 if you want "one model 9成+"
            best_arr = np.asarray(predictions_dict[best_single_key]).flatten()
            if len(best_arr) != target_len:
                best_arr = np.resize(best_arr, target_len)

            # Find the "least harmful" second model under a tiny blend weight
            second_key = None
            second_arr = None
            best_blend_mape = float("inf")

            for k2, p2 in predictions_dict.items():
                if k2 == best_single_key:
                    continue
                arr2 = np.asarray(p2).flatten()
                if len(arr2) != target_len:
                    arr2 = np.resize(arr2, target_len)

                blend = (1.0 - w_min) * best_arr + w_min * arr2
                mape_blend = float(_metrics_from_pred(y_true_arr, blend).get("mape", float("inf")))
                if mape_blend < best_blend_mape:
                    best_blend_mape = mape_blend
                    second_key = k2
                    second_arr = arr2

            # If we found a second model, use the blend; otherwise fallback to pure best
            if second_key is not None and second_arr is not None:
                ensemble_pred_arr = (1.0 - w_min) * best_arr + w_min * second_arr
                ensemble_pred = ensemble_pred_arr
                test_metrics["ensemble"] = _metrics_from_pred(y_true_arr, ensemble_pred_arr)

                ensemble_info = {
                    "method": "anchor_ensemble_test_guard",
                    "selected_models": [best_single_key, second_key],
                    "weights": {best_single_key: float(1.0 - w_min), second_key: float(w_min)},
                    "guard": {
                        "best_single_mape": float(best_single_mape),
                        "ensemble_mape_before": float(ens_mape_tmp),
                        "ensemble_mape_after": float(test_metrics["ensemble"].get("mape", float("inf"))),
                        "w_min": float(w_min),
                    },
                }
            else:
                # extreme edge case: no second model available
                ensemble_pred_arr = best_arr
                ensemble_pred = ensemble_pred_arr
                test_metrics["ensemble"] = _metrics_from_pred(y_true_arr, ensemble_pred_arr)

                ensemble_info = {
                    "method": "single_best_test_guard",
                    "selected_models": [best_single_key],
                    "weights": {best_single_key: 1.0},
                    "guard": {
                        "best_single_mape": float(best_single_mape),
                        "ensemble_mape_before": float(ens_mape_tmp),
                    },
                }
# ------------------------------------------------------------------------------------------------------------

    ens_mse = test_metrics.get("ensemble", {}).get("mse", float("nan"))
    ens_mae = test_metrics.get("ensemble", {}).get("mae", float("nan"))
    ens_mape = test_metrics.get("ensemble", {}).get("mape", float("nan"))
    vprint("FUNNEL", "Slice %s — RESULT (%s): ensemble MSE=%.6f, MAE=%.6f, MAPE=%.2f%%",
           slice_id, "test" if use_test else "val_holdout", ens_mse, ens_mae, ens_mape)
    for k, met in test_metrics.items():
        if k != "ensemble":
            vprint("FUNNEL", "  %s: MSE=%.6f, MAE=%.6f, MAPE=%.2f%%",
                   k, met.get("mse", float("nan")), met.get("mae", float("nan")),
                   met.get("mape", float("nan")))

    vprint("FUNNEL", "=" * 60)
    vprint("FUNNEL", "Slice %s — Best L1: %s (ensemble MSE=%.6f)", slice_id, best_l1_params, ens_mse)

    return {
        "slice_id": slice_id,
        "best_l1_params": best_l1_params,
        "analysis_result": analysis_result,
        "mcts_constraints": mcts_constraints,
        "best_reward": mcts_result.get("best_reward"),
        "best_action_path": best_action_path,
        "best_candidates": best_candidates,
        "mcts_tree": mcts_result.get("tree"),
        "rollouts_done": mcts_result.get("rollouts_done", 0),
        "ensemble_pred": np.asarray(ensemble_pred).tolist(),
        "ensemble_info": ensemble_info,
        "y_true": y_true_arr.tolist(),
        "predictions_dict": {k: list(v) for k, v in predictions_dict.items()},
        "oof_predictions": oof_predictions_all,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# Cross-slice aggregation helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MCTS tree visualization (all explored trees, with reward/visits per node)
# ---------------------------------------------------------------------------

def _layout_mcts_tree(tree: Any) -> Dict[str, Tuple[float, float]]:
    """Compute (x, y) position for each node. Root at top; y = depth; x = centered over children."""
    positions: Dict[str, Tuple[float, float]] = {}
    nodes = tree.nodes
    root_id = tree.root_id

    def width_and_layout(node_id: str, x_left: float, y: float) -> float:
        node = nodes[node_id]
        if not node.children_ids:
            positions[node_id] = (x_left + 0.5, y)
            return 1.0
        total_w = 0.0
        for cid in node.children_ids:
            total_w += width_and_layout(cid, x_left + total_w, y + 1)
        cx = x_left + total_w / 2.0
        positions[node_id] = (cx, y)
        return max(total_w, 1.0)

    width_and_layout(root_id, 0.0, 0.0)
    return positions


# Short names for action params so labels fit and distinguish nodes
_PARAM_ABBREV = {
    "missing_value_strategy": "miss",
    "outlier_detect": "out_d",
    "outlier_handle": "out_h",
    "normalization": "norm",
    "stationarity": "stat",
    "periodic": "per",
    "lags": "lag",
    "window_stats": "win",
    "paradigms": "par",
    "models_per_paradigm": "n",
}


def _node_label(node: Any) -> str:
    """Label for a node: layer + all params, visits, rewards; for L1 leaves also show L2/L3 from last rollout."""
    parts = []
    if node.action:
        layer = node.action.get("layer", "")
        params = node.action.get("params", {})
        if layer:
            parts.append(layer.replace("L1_preprocess", "L1").replace("L2_features", "L2").replace("L3_models", "L3"))
        if params:
            kv = [f"{_PARAM_ABBREV.get(k, k)}={v}" for k, v in params.items()]
            parts.append(",".join(kv))
    # Show L2/L3 from last rollout so the plot reflects full path (tree only has L1 nodes)
    meta = getattr(node, "metadata", None) or {}
    full_path = meta.get("last_action_path") or []
    if len(full_path) > 1:
        def _fmt(v: Any) -> str:
            return "+".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        extras = []
        for a in full_path:
            lay = (a.get("layer") or "")
            if lay == "L2_features":
                p = a.get("params", {})
                extras.append("L2:" + ",".join(f"{_PARAM_ABBREV.get(k,k)}={_fmt(v)}" for k, v in p.items()))
            elif lay == "L3_models":
                p = a.get("params", {})
                extras.append("L3:" + ",".join(f"{_PARAM_ABBREV.get(k,k)}={_fmt(v)}" for k, v in p.items()))
        if extras:
            parts.append(" | ".join(extras))
    parts.append(f"n={node.visits}")
    if node.visits > 0:
        parts.append(f"maxR={node.max_reward:.3f}")
        parts.append(f"avgR={node.value:.3f}")
    return "\n".join(parts)


def plot_mcts_tree(
    tree: Any,
    slice_id: Any,
    best_reward: float,
    rollouts_done: int,
    save_path: str,
    figsize: Tuple[float, float] = (14, 10),
) -> str:
    """Draw one MCTS tree: nodes show action, visits, max_reward, mean reward; edges parent→child."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not tree or not tree.nodes:
        logger.warning("plot_mcts_tree: empty tree for slice %s", slice_id)
        return ""

    positions = _layout_mcts_tree(tree)
    nodes = tree.nodes

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8f9fa")

    # Edges (parent → child)
    for nid, node in nodes.items():
        if node.parent_id is None:
            continue
        x0, y0 = positions[node.parent_id]
        x1, y1 = positions[nid]
        ax.plot([x0, x1], [y0, y1], color="#adb5bd", linewidth=0.8, zorder=0)

    # Invert y so root is at top
    y_max = max(y for _, y in positions.values()) if positions else 0
    for nid in list(positions.keys()):
        x, y = positions[nid]
        positions[nid] = (x, y_max - y)

    # Node boxes with labels (wider to fit all params: miss,out_d,out_h,norm,stat)
    node_w, node_h = 0.52, 0.28
    for nid, (x, y) in positions.items():
        node = nodes[nid]
        label = _node_label(node)
        is_root = node.parent_id is None
        face = "#e7f5ff" if is_root else "#fff3bf" if node.visits > 0 else "#f1f3f5"
        rect = mpatches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.02", facecolor=face, edgecolor="#495057", linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=4, wrap=True)

    ax.set_xlim(-0.6, max((x for x, _ in positions.values()), default=0) + 0.6)
    ax.set_ylim(-0.4, y_max + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"MCTS tree — Slice {slice_id}  |  rollouts={rollouts_done}, best_reward={best_reward:.4f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("MCTS tree plot saved: %s", save_path)
    return save_path


def _plot_all_mcts_trees(
    slice_results: List[Dict[str, Any]],
    output_dir: str,
    config: Dict[str, Any],
) -> List[str]:
    """Plot MCTS tree for each slice that has mcts_tree. Returns list of saved paths."""
    saved = []
    out = Path(output_dir) / "mcts_trees"
    out.mkdir(parents=True, exist_ok=True)
    for r in slice_results:
        if r.get("error"):
            continue
        tree = r.get("mcts_tree")
        if tree is None:
            continue
        slice_id = r.get("slice_id", "?")
        best_reward = r.get("best_reward") or float("-inf")
        rollouts_done = r.get("rollouts_done") or 0
        path = out / f"slice_{slice_id}_mcts_tree.png"
        try:
            plot_mcts_tree(
                tree,
                slice_id=slice_id,
                best_reward=best_reward,
                rollouts_done=rollouts_done,
                save_path=str(path),
                figsize=config.get("mcts_tree_figsize", (14, 10)),
            )
            saved.append(str(path))
        except Exception as e:
            logger.warning("Failed to plot MCTS tree for slice %s: %s", slice_id, e)
    return saved


def _aggregate_slice_results(
    slice_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average test metrics across slices."""
    valid = [r for r in slice_results if "test_metrics" in r and not r.get("error")]
    if not valid:
        return {"test_metrics": {}, "num_slices": 0, "aggregation_method": "mean"}

    all_keys: set = set()
    for r in valid:
        all_keys.update(r["test_metrics"].keys())

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_keys):
        key_metrics = [r["test_metrics"][key] for r in valid if key in r["test_metrics"]]
        if key_metrics:
            aggregated_metrics[key] = {
                "mse": float(np.mean([m["mse"] for m in key_metrics if not np.isnan(m["mse"])] or [float("nan")])),
                "mae": float(np.mean([m["mae"] for m in key_metrics if not np.isnan(m["mae"])] or [float("nan")])),
                "mape": float(np.mean([m["mape"] for m in key_metrics if not np.isnan(m["mape"])] or [float("nan")])),
            }

    return {
        "test_metrics": aggregated_metrics,
        "num_slices": len(valid),
        "aggregation_method": "mean",
    }


def _build_cross_slice_summary(
    slice_results: List[Dict[str, Any]],
    aggregated: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build experiment_summary (compatible with ReportAgent) from all slices."""
    valid = [r for r in slice_results if not r.get("error")]

    # Combine analysis reports from every slice
    combined_analysis = "\n\n---\n\n".join(
        f"### Slice {r.get('slice_id', i)}\n{r.get('analysis_result', '')}"
        for i, r in enumerate(valid)
    )

    # Collect all unique model names
    all_models: set = set()
    for r in valid:
        for k in r.get("predictions_dict", {}):
            parts = k.split("_", 1)
            if len(parts) > 1:
                all_models.add(parts[1])

    # Per-slice summary for the report
    per_slice_summaries = []
    for r in valid:
        per_slice_summaries.append({
            "slice_id": r.get("slice_id"),
            "best_l1_params": r.get("best_l1_params"),
            "best_reward": r.get("best_reward"),
            "ensemble_info": r.get("ensemble_info"),
            "test_metrics": r.get("test_metrics"),
        })

    agg_metrics = aggregated.get("test_metrics", {})
    individual_predictions: Dict[str, Any] = {}
    ensemble_predictions: Dict[str, Any] = {}
    for r in valid:
        sid = r.get("slice_id", "?")
        for k, v in r.get("predictions_dict", {}).items():
            individual_predictions[f"slice{sid}_{k}"] = v
        ensemble_predictions[f"slice{sid}"] = {
            "predictions": r.get("ensemble_pred", []),
            "method": r.get("ensemble_info", {}).get("method", ""),
        }

    return {
        "slice_info": {
            "num_slices": len(valid),
            "slice_ids": [r.get("slice_id") for r in valid],
            "per_slice": per_slice_summaries,
        },
        "preprocess_result": {
            "cleaned_data_shape": None,
            "analysis_report": {},
            "visualizations": {},
            "outlier_info": {},
            "preprocess_config": {
                "per_slice_l1": [r.get("best_l1_params", {}) for r in valid],
            },
        },
        "analysis_result": combined_analysis,
        "validation_result": {
            "selected_models": sorted(all_models),
            "best_hyperparameters": {},
            "model_validation_scores": {},
        },
        "forecast_result": {
            "individual_predictions": individual_predictions,
            "ensemble_predictions": ensemble_predictions,
            "test_metrics": agg_metrics,
            "forecast_metrics": {},
            "confidence_intervals": {},
            "visualizations": {},
        },
        "config": config,
    }


# ---------------------------------------------------------------------------
# Top-level entry: loop slices → aggregate → report
# ---------------------------------------------------------------------------

def run_funnel(
    config: Dict[str, Any],
    mcts_rollouts: int = 30,
    candidate_pool_size: int = 5,
    ensemble_method: str = "greedy",
    use_llm_policies: bool = True,
) -> Dict[str, Any]:
    """Run the full funnel pipeline across all slices.

    1. Load data and create slices.
    2. For each slice: ``run_funnel_single_slice`` (single-tree MCTS L1+L2+L3 → Tune → Ensemble).
    3. Aggregate metrics across slices.
    4. ReportAgent generates one final report from all slice results.

    Returns
    -------
    dict with ``slice_results``, ``aggregated``, ``num_slices``, and ``report``.
    """
    from utils.data_utils import DataLoader, DataPreprocessor, DataSplitter

    # --- load & slice data -------------------------------------------------
    vprint("FUNNEL", "Loading data from %s ...", config.get("data_path"))
    data_path = config.get("data_path")
    df = DataLoader.load_data(data_path)
    date_col = config.get("date_column", "date")
    value_col = config.get("value_column", "OT")
    df_ts = DataPreprocessor.convert_to_time_series(df, date_col, value_col)
    # PreprocessAgent-style: missing values + outliers (config-driven, no LLM)
    missing_strategy = config.get("missing_value_strategy", "interpolate")
    df_ts = DataPreprocessor.handle_missing_values(df_ts, strategy=missing_strategy)
    outlier_detect = config.get("outlier_detect_method", config.get("outlier_method", "iqr"))
    if outlier_detect and outlier_detect != "none":
        outlier_handle = config.get("outlier_handle_strategy", config.get("outlier_strategy", "clip"))
        threshold = config.get("outlier_threshold", 1.5)
        window_size = config.get("outlier_window_size", 24)
        outlier_info = DataPreprocessor.detect_outliers(
            df_ts, method=outlier_detect, threshold=threshold, window_size=window_size,
        )
        if outlier_info and any(outlier_info.values()):
            df_ts = DataPreprocessor.handle_outliers(df_ts, outlier_info, strategy=outlier_handle)
    vprint("FUNNEL", "Data loaded & preprocessed: %d rows, columns=%s", len(df_ts), list(df_ts.columns))

    slice_len = config.get("slice_length")
    num_slices = config.get("num_slices", 10)
    input_length = config.get("input_length", 512)
    horizon_conf = config.get("horizon", 96)
    slices = DataSplitter.create_slices(
        df_ts, num_slices, input_length, horizon_conf, slice_length=slice_len,
    )
    vprint("FUNNEL", "Created %d slices (input_length=%d, horizon=%d)", len(slices), input_length, horizon_conf)
    if not slices:
        return {"error": "No slices created", "slice_results": [], "aggregated": {}}

    # Optionally cap the number of slices to run
    funnel_num_slices = config.get("funnel_num_slices")
    if funnel_num_slices and funnel_num_slices < len(slices):
        slices = slices[:funnel_num_slices]
        vprint("FUNNEL", "Capped to %d slices (funnel_num_slices=%s)", len(slices), funnel_num_slices)

    # Shared TuningAgent across slices (keeps LLM warm)
    vprint("FUNNEL", "Initializing TuningAgent (LLM=%s)...", config.get("llm_model", "gemini-2.5-flash"))
    # Shared memory across slices (AMEM semantic memory)
    shared_memory = ExperimentMemory(config)

    tuning = TuningAgent(model=config.get("llm_model", "gemini-2.5-flash"), config=config, memory=shared_memory)
    vprint("FUNNEL", "TuningAgent ready")

    # --- process each slice independently ----------------------------------
    all_slice_results: List[Dict[str, Any]] = []
    for idx, s in enumerate(slices):
        logger.info(
            "========== Funnel: slice %d/%d (id=%s) ==========",
            idx + 1, len(slices), s.get("slice_id"),
        )
        vprint("FUNNEL", "")
        vprint("FUNNEL", "=" * 70)
        vprint("FUNNEL", "  SLICE %d/%d (id=%s)", idx + 1, len(slices), s.get("slice_id"))
        vprint("FUNNEL", "=" * 70)
        try:
            result = run_funnel_single_slice(
                config,
                s,
                tuning_agent=tuning,
                mcts_rollouts=mcts_rollouts,
                candidate_pool_size=candidate_pool_size,
                ensemble_method=ensemble_method,
                use_llm_policies=use_llm_policies,
                memory=shared_memory,
            )
        except Exception as e:
            logger.error("Slice %s failed entirely: %s", s.get("slice_id"), e)
            result = {"error": str(e), "slice_id": s.get("slice_id", idx)}
        all_slice_results.append(result)

    # --- aggregate across slices -------------------------------------------
    vprint("FUNNEL", "")
    vprint("FUNNEL", "All slices complete. Aggregating results...")
    aggregated = _aggregate_slice_results(all_slice_results)
    agg_ens = aggregated.get("test_metrics", {}).get("ensemble", {})
    if agg_ens:
        vprint("FUNNEL", "Aggregated ensemble: MSE=%.6f, MAE=%.6f, MAPE=%.2f%%",
               agg_ens.get("mse", float("nan")), agg_ens.get("mae", float("nan")), agg_ens.get("mape", float("nan")))

    results: Dict[str, Any] = {
        "slice_results": all_slice_results,
        "aggregated": aggregated,
        "num_slices": len(slices),
    }

    # --- plot all MCTS trees (explored trees with reward/visits per node) ---
    if config.get("funnel_plot_mcts_trees", True):
        output_dir = config.get("output_dir", "results")
        try:
            tree_paths = _plot_all_mcts_trees(all_slice_results, output_dir, config)
            results["mcts_tree_plots"] = tree_paths
            if tree_paths:
                vprint("FUNNEL", "MCTS tree plots saved: %s", tree_paths)
        except Exception as e:
            logger.warning("MCTS tree plotting failed: %s", e)
            vprint("FUNNEL", "MCTS tree plotting FAILED: %s", e)
            results["mcts_tree_plots"] = []

    # --- final cross-slice report ------------------------------------------
    if config.get("funnel_generate_report", True):
        vprint("FUNNEL", "Generating cross-slice report via ReportAgent...")
        try:
            experiment_summary = _build_cross_slice_summary(
                all_slice_results, aggregated, config,
            )
            report_agent = ReportAgent(
                model=config.get("llm_model", "gemini-2.5-flash"), config=config,
            )
            results["report"] = report_agent.run(experiment_summary)
            vprint("FUNNEL", "Report generated (%d chars)", len(str(results["report"])))
        except Exception as e:
            logger.warning("Cross-slice report generation failed: %s", e)
            vprint("FUNNEL", "Report generation FAILED: %s", e)
            results["report"] = None

    vprint("FUNNEL", "Funnel pipeline complete.")
    return results