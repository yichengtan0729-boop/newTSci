"""
Three-layer Action Space for MCTS time series pipeline search.

MCTS handles only L1-L3 (structural decisions).
L4 (ensemble) is delegated to EnsembleAgent after Tuning.
L5 (hyperparameter tuning) is delegated to TuningAgent (ReAct).

Compact spec format: each param -> [options]. LLM picks one per param.
L3 paradigms: single-select — each MCTS path picks ONE paradigm + ONE model.
Model diversity comes from running multiple MCTS rollouts; ensemble fuses them.
"""

from itertools import combinations
from typing import Any, Dict, List

# 模型范式（L3 用）
MODEL_PARADIGM = {
    "statistical": [
        "ARIMA", "ExponentialSmoothing", "TBATS", "Prophet",
        "Theta", "Croston", "RandomWalk", "MovingAverage",
    ],
    "regression": [
        "LinearRegression", "PolynomialRegression", "RidgeRegression",
        "LassoRegression", "ElasticNet", "SVR",
    ],
    "tree": ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"],
    "deep": ["LSTM", "NeuralNetwork", "Transformer"],
    "foundation": ["TTM"],
}


# PreprocessAgent-style strategies (no LLM agent; same capabilities as data_utils)
MISSING_VALUE_STRATEGIES = [
    "interpolate", "forward_fill", "backward_fill", "mean", "median", "drop", "zero", "none",
]
OUTLIER_DETECT_METHODS = ["iqr", "zscore", "percentile", "none"]
# Time-series safe only: no 'drop' (breaks lag/calendar continuity)
OUTLIER_HANDLE_STRATEGIES = [
    "clip", "interpolate", "ffill", "bfill", "mean", "median", "smooth", "none",
]

# 三层 Action Space（L1-L3，紧凑：每参数 → 选项列表）
# L4 融合由 EnsembleAgent 在 Tuning 之后独立处理；L5 超参由 TuningAgent (ReAct) 处理
ACTION_SPACE: Dict[str, Dict[str, Any]] = {
    "L1_preprocess": {
        "actions": {
            "missing_value_strategy": {"options": MISSING_VALUE_STRATEGIES},
            "outlier_detect": {"options": OUTLIER_DETECT_METHODS},
            "outlier_handle": {"options": OUTLIER_HANDLE_STRATEGIES},
            "normalization": {"options": ["minmax", "zscore", "none"]},
            "stationarity": {"options": ["diff", "log", "none"]},
        }
    },
    "L2_features": {
        "actions": {
            "periodic": {"options": ["fourier", "none"]},
            "lags": {"options": [5, 10, 20, 50]},
            "window_stats": {"options": ["mean", "std", "min_max", "none"]},
        }
    },
    "L3_models": {
        "actions": {
            # Single-select: each MCTS path picks exactly ONE paradigm, ONE model.
            # Ensemble happens only after Tuning, fusing results from different paths.
            "paradigms": {"options": ["statistical", "regression", "tree", "deep", "foundation"]},
            # Fixed to 1 — one model per path, diversity comes from multiple MCTS paths.
            "models_per_paradigm": {"options": [1]},
        }
    },
}


def get_action_space(layer: str = None) -> Dict[str, Any]:
    """Get action space for a layer or all."""
    if layer:
        return ACTION_SPACE.get(layer, {})
    return ACTION_SPACE


def get_layer_action_spec(layer: str, action_space: Dict[str, Any] = None) -> Dict[str, List[Any]]:
    """
    Return compact spec: {param_name: [options]}.
    LLM sees this and picks one value per param. No Cartesian product.

    L3 paradigms are single-select: each MCTS path picks exactly one paradigm.
    """
    space = action_space or ACTION_SPACE
    layer_spec = space.get(layer, {})
    actions_spec = layer_spec.get("actions", {})
    if not actions_spec:
        return {}

    spec: Dict[str, List[Any]] = {}
    for key, meta in actions_spec.items():
        options = meta.get("options", [])
        if meta.get("combinable") and key == "paradigms":
            combos = []
            for r in range(1, len(options) + 1):
                for c in combinations(options, r):
                    combos.append(list(c))
            spec[key] = combos
        else:
            spec[key] = list(options)
    return spec


def sample_action(layer: str, use_random: bool = True) -> Dict[str, Any]:
    """Sample one action (one value per param)."""
    import random
    spec = get_layer_action_spec(layer)
    if not spec:
        return {}
    action = {}
    for key, opts in spec.items():
        action[key] = random.choice(opts) if use_random else opts[0]
    return action
