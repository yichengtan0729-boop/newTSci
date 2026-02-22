"""
Hyperparameter Tuning Agent (ReAct)

Called after MCTS determines L1-L4. Uses ReAct loop:
  Observation -> Thought -> Action -> Execution
to tune hyperparameters by analyzing train/val loss curves and proposing next params.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.llm_factory import get_llm
from utils.progress import vprint
from agents.memory import ExperimentMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complete per-model hyperparameter catalog
# Keys match model names in MODEL_FUNCTIONS; values describe every param
# that the model's predict_* function reads via params.get(...).
# ---------------------------------------------------------------------------
MODEL_PARAM_CATALOG: Dict[str, Dict[str, str]] = {
    "ARIMA": {
        "p": "AR order (int, default 1). Typical: 1-5.",
        "d": "Differencing order (int, default 1). Typical: 0-2.",
        "q": "MA order (int, default 1). Typical: 1-5.",
    },
    "ExponentialSmoothing": {
        "trend": "'add', 'mul', or None. Additive vs multiplicative trend.",
        "seasonal": "'add', 'mul', or None. Additive vs multiplicative seasonality.",
        "seasonal_periods": "int, e.g. 12, 24, 48. Seasonal cycle length.",
    },
    "Prophet": {
        "yearly_seasonality": "bool (default False). Enable yearly seasonality.",
        "weekly_seasonality": "bool (default False). Enable weekly seasonality.",
        "daily_seasonality": "bool (default False). Enable daily seasonality.",
        "seasonality_mode": "'additive' or 'multiplicative' (default 'additive').",
    },
    "TBATS": {
        "seasonal_periods": "list[int] or None. Seasonal periods to model.",
        "use_arma_errors": "bool. Whether to use ARMA errors.",
        "use_box_cox": "bool or None. Whether to apply Box-Cox transform.",
    },
    "Theta": {
        "seasonal_period": "int (default 12). The dominant seasonal cycle length.",
    },
    "Croston": {
        "alpha": "float (default 0.4). Smoothing parameter, 0-1.",
    },
    "RandomWalk": {},
    "MovingAverage": {
        "window_size": "int (default 10). Rolling window length.",
    },
    "LinearRegression": {},
    "PolynomialRegression": {
        "degree": "int (default 2). Polynomial degree, e.g. 2 or 3.",
    },
    "RidgeRegression": {
        "alpha": "float (default 1.0). L2 regularization strength. Typical: 0.01-100.",
    },
    "LassoRegression": {
        "alpha": "float (default 1.0). L1 regularization strength. Typical: 0.001-10.",
    },
    "ElasticNet": {
        "alpha": "float (default 1.0). Overall regularization strength.",
        "l1_ratio": "float (default 0.5). Mix of L1 vs L2, 0-1. 1.0 = pure Lasso.",
    },
    "SVR": {
        "C": "float (default 1.0). Regularization. Typical: 0.1, 1, 10, 100.",
        "gamma": "'scale' or 'auto' or float. Kernel coefficient.",
        "kernel": "'rbf', 'linear', 'poly'. Kernel type.",
        "epsilon": "float (default 0.1). Epsilon-tube width.",
    },
    "RandomForest": {
        "n_estimators": "int (default 100). Number of trees. Typical: 50-500.",
        "max_depth": "int or None (default None). Max tree depth. Typical: 5-30 or None.",
        "min_samples_split": "int (default 2). Min samples to split a node.",
        "min_samples_leaf": "int (default 1). Min samples per leaf.",
    },
    "GradientBoosting": {
        "n_estimators": "int (default 100). Boosting rounds. Typical: 100-500.",
        "max_depth": "int (default 3). Max tree depth. Typical: 3-8.",
        "learning_rate": "float (default 0.1). Shrinkage rate. Typical: 0.01-0.3.",
        "subsample": "float (default 1.0). Row subsampling ratio, 0-1.",
    },
    "XGBoost": {
        "n_estimators": "int (default 100). Boosting rounds. Typical: 100-500.",
        "max_depth": "int (default 6). Max tree depth. Typical: 3-10.",
        "learning_rate": "float (default 0.3). Eta / shrinkage. Typical: 0.01-0.3.",
        "subsample": "float (default 1.0). Row subsampling ratio.",
        "colsample_bytree": "float (default 1.0). Feature subsampling ratio.",
        "reg_alpha": "float (default 0). L1 regularization on weights.",
        "reg_lambda": "float (default 1). L2 regularization on weights.",
    },
    "LightGBM": {
        "n_estimators": "int (default 100). Boosting rounds.",
        "max_depth": "int (default -1, unlimited). Typical: 3-12 or -1.",
        "learning_rate": "float (default 0.1). Typical: 0.01-0.3.",
        "num_leaves": "int (default 31). Max leaves per tree. Typical: 15-127.",
        "subsample": "float (default 1.0). Bagging fraction.",
        "colsample_bytree": "float (default 1.0). Feature fraction.",
        "reg_alpha": "float (default 0). L1 regularization.",
        "reg_lambda": "float (default 0). L2 regularization.",
    },
    "NeuralNetwork": {
        "hidden_layer_sizes": "tuple, e.g. (64,) or (128, 64). MLP layer sizes.",
        "max_iter": "int (default 200). Max training iterations.",
        "learning_rate_init": "float (default 0.001). Initial learning rate.",
        "activation": "'relu', 'tanh', 'logistic'. Activation function.",
        "alpha": "float (default 0.0001). L2 regularization.",
        "batch_size": "int or 'auto' (default 'auto'). Mini-batch size.",
        "early_stopping": "bool (default False). Use validation-based early stopping.",
    },
    "LSTM": {
        "lookback": "int (default 20). Input sequence window length.",
        "units": "int (default 64). LSTM hidden size.",
        "layers": "int (default 2). Number of stacked LSTM layers.",
        "dropout": "float (default 0.1). Dropout between LSTM layers, 0-0.5.",
        "epochs": "int (default 50). Training epochs.",
        "batch_size": "int (default 32). Mini-batch size.",
        "learning_rate": "float (default 0.001). Adam learning rate.",
    },
    "Transformer": {
        "lookback": "int (default 20). Input sequence window length.",
        "d_model": "int (default 64). Embedding dimension (must be divisible by nhead).",
        "nhead": "int (default 4). Number of attention heads.",
        "num_layers": "int (default 2). Number of TransformerEncoder layers.",
        "dropout": "float (default 0.1). Dropout rate, 0-0.5.",
        "epochs": "int (default 50). Training epochs.",
        "batch_size": "int (default 32). Mini-batch size.",
        "learning_rate": "float (default 0.001). Adam learning rate.",
    },
}


def _format_param_catalog_for_models(models: List[str]) -> str:
    """Build a human-readable parameter reference for the given model names."""
    lines: List[str] = []
    for m in models:
        cat = MODEL_PARAM_CATALOG.get(m, {})
        if not cat:
            lines.append(f"- **{m}**: no tunable hyperparameters (uses defaults).")
            continue
        lines.append(f"- **{m}**:")
        for k, desc in cat.items():
            lines.append(f"    - `{k}`: {desc}")
    return "\n".join(lines)


FAST_SIM_SUGGEST_PROMPT = """You suggest hyperparameters for a quick validation run (no ReAct loop).
Given a list of time series model names, their tunable parameters, and max_epochs for neural models,
output exactly one JSON object. Keys are model names, values are hyperparameter dicts.
Use only numeric/string/bool values.

Constraints:
- For LSTM/NeuralNetwork/Transformer set "epochs" to at most max_epochs.
- Choose reasonable defaults; prefer conservative values for a fast run.
- Output ONLY the JSON object, no markdown or explanation."""

TUNING_SYSTEM_PROMPT = """You are a Hyperparameter Tuning Agent for time series forecasting.
You reason like an algorithm engineer: observe training logs, diagnose issues, propose changes.

## ReAct Loop
1. **Observation**: Review the training output (metrics: MSE, MAE, MAPE from Rolling CV).
2. **Thought**: Diagnose the situation.
   - High MSE + High MAE → model under-fits → increase capacity (more estimators/units/layers, lower regularization).
   - MAE improves but MAPE worsens → scale-sensitive issues → try different normalization or regularization.
   - Metrics barely change between trials → may be near optimum → try a different axis (e.g. switch from tuning learning_rate to tuning depth).
   - For tree models: if MAE is good but MAPE is high → try reducing max_depth or increasing min_samples_leaf (overfitting to outliers).
   - For neural models (LSTM/Transformer): if metrics oscillate → reduce learning_rate; if metrics plateau → increase units/layers or reduce dropout.
3. **Action**: Call `train_trial_model` with the next param set to test, OR call `finish` to return the best so far.
4. **Execution**: The tool runs; you get new Observation. Repeat until you call `finish`.

## Important
- You MUST use the exact parameter names listed in the model's parameter catalog below.
- Only include parameters relevant to the model being tuned.
- Propose ONE new param set per `train_trial_model` call.
- After 3–5 trials or when validation metric improves little, call `finish`.
- Always include a brief Thought before each Action.
"""


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


@dataclass
class TuningContext:
    """Context for a tuning run: fixed L1-L4 pipeline, data, models."""

    data: Any  # pd.DataFrame or dict with 'value'
    models: List[str]
    horizon: int
    train_fn: Callable[
        [Any, str, Dict[str, Any], int, Optional[int]],
        Tuple[Dict[str, float], Optional[Dict[str, List[float]]]],
    ]
    max_trials: int = 8
    max_epochs_per_trial: int = 20


_TUNING_CV_SPLITS = 3  # default n_splits for Rolling Time-Series CV


def _default_train_trial(
    data: Any,
    model_name: str,
    params: Dict[str, Any],
    horizon: int,
    max_epochs: Optional[int],
) -> Tuple[Dict[str, float], Optional[Dict[str, List[float]]]]:
    """Evaluate one param set via **Rolling Time-Series CV** (Stage 2).

    Uses ``sklearn.model_selection.TimeSeriesSplit(n_splits)`` through
    :func:`utils.validation.rolling_cv_with_oof`.

    Supports enriched data dicts (with L2 feature columns) — extra columns
    are passed through to each CV fold so tree/regression models can use them.

    Returns
    -------
    avg_metrics : dict
        ``{"mse": ..., "mae": ..., "mape": ...}`` averaged across folds.
    extra : dict or None
        ``{"oof_predictions": list[float]}`` — full-length OOF array with
        ``NaN`` at non-validation positions, or ``None`` on failure.
    """
    from utils.model_library import get_model_function
    from utils.validation import rolling_cv_with_oof
    import pandas as pd
    import numpy as np

    enriched_data = None
    scaler = None
    series_original = None
    if isinstance(data, pd.DataFrame):
        series = data["value"].dropna().values if "value" in data.columns else data.iloc[:, 0].values
    elif isinstance(data, dict):
        series = np.array(data.get("value", list(data.values())[0] if data else []))
        if len(data) > 1:
            enriched_data = {k: v for k, v in data.items() if k not in ("value_original", "scaler")}
        scaler = data.get("scaler")
        if data.get("value_original") is not None:
            series_original = np.asarray(data["value_original"]).flatten()
    else:
        series = np.array(data)

    trial_params = dict(params)
    if max_epochs and model_name in ("LSTM", "NeuralNetwork", "Transformer"):
        trial_params["epochs"] = min(trial_params.get("epochs", 50), max_epochs)

    try:
        model_func = get_model_function(model_name)
        avg_metrics, oof_preds = rolling_cv_with_oof(
            series, horizon, model_func, trial_params,
            n_splits=_TUNING_CV_SPLITS, enriched_data=enriched_data,
            scaler=scaler, series_original=series_original,
        )
        return avg_metrics, {"oof_predictions": oof_preds.tolist()}
    except Exception as e:
        logger.warning(f"Trial failed for {model_name}: {e}")
        return {"mse": float("inf"), "mae": float("inf"), "mape": float("inf")}, None


class TuningAgent:
    """
    ReAct-based hyperparameter tuning agent.
    Uses LLM to reason over training logs and propose next params.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        config: Optional[Dict[str, Any]] = None,
        memory: Optional[ExperimentMemory] = None
    ):
        self.config = config or {}
        cfg = {**self.config, "llm_model": self.config.get("llm_model", model)}
        cfg.setdefault("llm_max_tokens", 1500)
        self.llm = get_llm(cfg)
        self.trial_history: List[Dict[str, Any]] = []
        self.memory = memory or ExperimentMemory(self.config)

    def suggest_params_for_fast_sim(
        self,
        models: List[str],
        horizon: int,
        max_epochs: int = 5,
        data_info: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        One-shot LLM call to suggest hyperparameters for fast simulation (no ReAct).
        Used in MCTS rollouts to let the LLM choose params for quick validation.
        """
        if not models:
            return {}
        vprint("LLM", "Fast sim param suggestion: models=%s, horizon=%d, max_epochs=%d", models, horizon, max_epochs)
        catalog_text = _format_param_catalog_for_models(models)

        # AMEM: retrieve prior tuning notes (optional)
        mem_block = ""
        try:
            hits = self.memory.retrieve_semantic(
                query="hyperparameter tuning for time series models",
                top_k=int(self.config.get("amem", {}).get("top_k", 5)),
            )
            if hits:
                mem_block = "\n\n[AMEM — Relevant prior notes]\n" + "\n".join([f"- ({h['score']:.3f}) {h['text']}" for h in hits])
        except Exception:
            pass

        prompt = f"""{FAST_SIM_SUGGEST_PROMPT}{mem_block}

## Tunable parameters per model
{catalog_text}

Models: {models}. Horizon: {horizon}. max_epochs: {max_epochs}. {data_info}
Output only one JSON object: keys = model names, values = hyperparameter dicts. For LSTM/NeuralNetwork/Transformer set epochs <= {max_epochs}."""
        try:
            msg = self.llm.invoke([HumanMessage(content=prompt)])
            text = getattr(msg, "content", "") or str(msg)
            # strip markdown code block if present
            if "```" in text:
                for part in text.split("```"):
                    if part.strip().startswith("{"):
                        text = part.strip()
                        break
            obj = _extract_json(text)
            if not obj:
                vprint("LLM", "Fast sim suggestion: no valid JSON returned, using defaults")
                return {m: {} for m in models}
            out = {}
            for m in models:
                out[m] = obj.get(m, {})
                if m in ("LSTM", "NeuralNetwork", "Transformer") and "epochs" in out[m]:
                    out[m]["epochs"] = min(int(out[m]["epochs"]), max_epochs)
            vprint("LLM", "Fast sim suggestion result: %s", out)
            return out
        except Exception as e:
            logger.warning(f"suggest_params_for_fast_sim failed: {e}")
            vprint("LLM", "Fast sim suggestion FAILED: %s", e)
            return {m: {} for m in models}

    def _create_tools(self, ctx: TuningContext):
        """Create tools bound to this run's context."""
        agent_self = self

        # IMPORTANT: function names must match the tc["name"] checks in run().
        # LangChain's @tool decorator uses __name__ as the tool name exposed
        # to the LLM, so the function name IS the tool name.

        def train_trial_model(params_json: str) -> str:
            """Train a trial with the given hyperparameters (JSON string) and return metrics."""
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except Exception:
                return json.dumps({"error": "Invalid JSON params", "metrics": None})

            all_metrics = {}
            oof_per_model = {}
            for model_name in ctx.models:
                metrics, extra = ctx.train_fn(
                    ctx.data,
                    model_name,
                    params,
                    ctx.horizon,
                    ctx.max_epochs_per_trial,
                )
                all_metrics[model_name] = metrics
                if extra and isinstance(extra, dict):
                    oof_vals = extra.get("oof_predictions")
                    if oof_vals is not None:
                        oof_per_model[model_name] = oof_vals

            trial = {"params": params, "metrics": all_metrics, "oof": oof_per_model}
            agent_self.trial_history.append(trial)
            # Do NOT include OOF arrays in LLM output (too large / noisy)
            return json.dumps({"trial": len(agent_self.trial_history), "params": params, "metrics": all_metrics})

        train_trial_model_tool = tool(train_trial_model)

        def finish(best_params_json: str, reason: str = "") -> str:
            """Finish tuning and return the best hyperparameters found so far."""
            return json.dumps({"action": "finish", "best_params": best_params_json, "reason": reason})

        finish_tool = tool(finish)
        return {"train_trial_model": train_trial_model_tool, "finish": finish_tool}

    def run(
        self,
        context: TuningContext,
    ) -> Tuple[Dict[str, Dict[str, Any]], float, Dict[str, List[float]]]:
        """Run ReAct tuning loop with Rolling CV.

        Returns
        -------
        best_params_per_model : dict[str, dict]
            Best hyper-parameters (same dict applied to every model).
        best_metric : float
            Average MAE of the best trial.
        oof_predictions : dict[str, list[float]]
            Per-model Out-of-Fold prediction arrays from the best trial.
            Keys are model names, values are lists of length *N* (full
            training series) with ``NaN`` at non-validation positions.
        """
        self.trial_history = []
        tools_dict = self._create_tools(context)
        tools_list = list(tools_dict.values())
        bound_llm = self.llm.bind_tools(tools_list)

        catalog_text = _format_param_catalog_for_models(context.models)

        # Compute data length for context
        _data_len = "unknown"
        if isinstance(context.data, dict):
            _v = context.data.get("value")
            if _v is not None:
                import numpy as _np
                _data_len = str(len(_np.asarray(_v).flatten()))
        elif hasattr(context.data, "shape"):
            _data_len = str(context.data.shape)

        prompt = f"""
Start hyperparameter tuning for models: {context.models}.
Data length: {_data_len}, horizon: {context.horizon}.

## Tunable parameters for the current model(s)
{catalog_text}

**Instructions:**
1. Call `train_trial_model` with an initial param set (as JSON string) using parameter names from the catalog above.
2. Observe the returned metrics (MSE, MAE, MAPE from Rolling CV).
3. Reason about the results and propose a better param set.
4. Repeat for 3-5 trials, then call `finish` with the best params.

Start with reasonable defaults from the catalog, then iterate.
"""
        messages = [
            SystemMessage(content=TUNING_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        def _pick_best():
            """Return (best_params, best_metric, oof) from trial_history."""
            if not self.trial_history:
                return {m: {} for m in context.models}, float("inf"), {}
            best = min(
                self.trial_history,
                key=lambda t: sum(
                    m.get("mae", float("inf")) for m in t["metrics"].values()
                ) / max(len(t["metrics"]), 1),
            )
            best_metric = sum(m["mae"] for m in best["metrics"].values()) / len(best["metrics"])
            best_params = best["params"]
            oof = best.get("oof", {})
            return {m: best_params for m in context.models}, best_metric, oof

        vprint("TUNING", "ReAct loop starting: models=%s, max_trials=%d, horizon=%d",
               context.models, context.max_trials, context.horizon)
        for trial_round in range(context.max_trials):
            vprint("TUNING", "  ReAct round %d/%d: calling LLM...", trial_round + 1, context.max_trials)
            try:
                response = bound_llm.invoke(messages)
            except Exception as e:
                logger.warning("TuningAgent LLM call failed (round %d): %s", trial_round, e)
                vprint("TUNING", "  ReAct round %d: LLM call FAILED: %s", trial_round + 1, e)
                break
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                vprint("TUNING", "  ReAct round %d: LLM returned no tool calls, stopping", trial_round + 1)
                break

            for tc in tool_calls:
                if tc["name"] == "finish":
                    vprint("TUNING", "  ReAct round %d: LLM called finish()", trial_round + 1)
                    return _pick_best()

                if tc["name"] == "train_trial_model":
                    args = tc.get("args", {})
                    vprint("TUNING", "  ReAct round %d: train_trial_model(%s)", trial_round + 1,
                           str(args)[:200])
                    try:
                        tool_output = tools_dict["train_trial_model"].invoke(args)
                        if isinstance(tool_output, dict):
                            tool_output = json.dumps(tool_output)
                        vprint("TUNING", "  ReAct round %d: trial result=%s", trial_round + 1,
                               str(tool_output)[:200])
                    except Exception as e:
                        logger.warning("train_trial_model tool failed: %s", e)
                        vprint("TUNING", "  ReAct round %d: trial FAILED: %s", trial_round + 1, e)
                        tool_output = json.dumps({"error": str(e), "metrics": None})
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tc["id"]))

        best_params, best_metric, oof = _pick_best()
        vprint("TUNING", "ReAct loop done: best_metric(MAE)=%.6f, %d trials completed", best_metric, len(self.trial_history))

        # AMEM: store tuning outcome
        try:
            import json as _json
            self.memory.store_semantic(
                text=f"Tuning outcome: best_metric_MAE={best_metric:.6f} best_params={_json.dumps(best_params, ensure_ascii=False)}",
                meta={"agent": "TuningAgent"},
            )
        except Exception:
            pass
        return best_params, best_metric, oof
