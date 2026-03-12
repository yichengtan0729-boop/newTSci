"""
Model-First Three-layer Action Space for MCTS time series pipeline search.

Search order:  L1_model → L2_preprocess → L3_features
Execution order: apply_preprocess(L2) → apply_features(L3) → train_model(L1)

L2/L3 options are conditioned on the chosen L1 model via MODEL_ACTION_SPACE.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model taxonomy
# ---------------------------------------------------------------------------
MODEL_PARADIGM = {
    "statistical": [
        "ARIMA", "SARIMA", "ExponentialSmoothing", "TBATS", "Prophet",
        "Theta", "Croston", "RandomWalk", "MovingAverage",
    ],
    "regression": [
        "LinearRegression", "PolynomialRegression", "RidgeRegression",
        "LassoRegression", "ElasticNet", "SVR",
    ],
    "tree": ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"],
    "deep": ["LSTM", "NeuralNetwork", "Transformer"],
}

ALL_MODEL_NAMES: List[str] = sorted(m for ms in MODEL_PARADIGM.values() for m in ms)

# ---------------------------------------------------------------------------
# Legacy option constants (kept as references)
# ---------------------------------------------------------------------------
TIME_INDEX_REGULARIZE = ["on", "off"]

MISSING_VALUE_STRATEGIES = [
    "interpolate_time", "interpolate_linear",
    "forward_fill", "backward_fill", "seasonal_fill",
    "mean", "median", "drop", "zero", "none",
]

OUTLIER_DETECT_METHODS = ["none", "iqr", "rolling_z", "percentile"]

OUTLIER_HANDLE_STRATEGIES = [
    "none", "clip_1_99", "winsorize", "mask_as_missing", "smooth",
]

NORMALIZATION_METHODS = ["none", "zscore", "minmax", "robust"]

TARGET_TRANSFORMS = [
    "none",
    "log1p", "yeo_johnson",
    "diff1", "diff2",
    "seasonal7", "seasonal24",
    "diff1_seasonal7", "diff1_seasonal24",
]

RECIPE_LEVELS = ["none", "time_only", "tree_light", "tree_heavy", "tree_extreme"]

# ---------------------------------------------------------------------------
# ACTION_SPACE — Model-First
# ---------------------------------------------------------------------------
ACTION_SPACE: Dict[str, Dict[str, Any]] = {
    "L1_model": {
        "actions": {
            "model": {"options": ALL_MODEL_NAMES},
        }
    },
}

# ---------------------------------------------------------------------------
# MODEL_ACTION_SPACE — per-model legal L2 + L3 options
# ---------------------------------------------------------------------------
MODEL_ACTION_SPACE: Dict[str, Dict[str, Dict[str, List]]] = {
    "XGBoost": {
        "L2_preprocess": {
            "missing_value_strategy": ["none", "interpolate_time", "forward_fill"],
            "outlier_detect": ["none", "iqr"],
            "outlier_handle": ["none", "clip_1_99"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["tree_light", "tree_heavy", "tree_extreme"]},
    },
    "LightGBM": {
        "L2_preprocess": {
            "missing_value_strategy": ["none", "interpolate_time", "forward_fill"],
            "normalization": ["none"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["tree_light", "tree_heavy", "tree_extreme"]},
    },
    "RandomForest": {
        "L2_preprocess": {
            "missing_value_strategy": ["interpolate_time", "forward_fill"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["tree_light", "tree_heavy"]},
    },
    "GradientBoosting": {
        "L2_preprocess": {
            "missing_value_strategy": ["interpolate_time", "forward_fill", "seasonal_fill"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["tree_light", "tree_heavy"]},
    },

    "ExponentialSmoothing": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_time", "forward_fill"],
            "outlier_detect": ["iqr", "rolling_z"],
            "outlier_handle": ["clip_1_99", "winsorize"],
            "normalization": ["none"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "SARIMA": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_linear", "seasonal_fill"],
            "normalization": ["none"],
            "target_transform": ["none", "log1p", "yeo_johnson"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "TBATS": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_time"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "Prophet": {
        "L2_preprocess": {
            "time_index_regularize": ["off"],
            "missing_value_strategy": ["none"],
            "outlier_detect": ["none"],
            "outlier_handle": ["none"],
            "normalization": ["none"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "Theta": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_linear"],
            "normalization": ["none"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "RandomWalk": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["forward_fill"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "MovingAverage": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_linear", "forward_fill"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["none"]},
    },
    "Croston": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["zero", "forward_fill"],
            "normalization": ["none"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["none"]},
    },

    "SVR": {
        "L2_preprocess": {
            "missing_value_strategy": ["interpolate_linear", "forward_fill"],
            "outlier_detect": ["iqr", "percentile"],
            "outlier_handle": ["clip_1_99", "winsorize"],
            "normalization": ["zscore", "minmax", "robust"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["time_only", "tree_light"]},
    },
    "ElasticNet": {
        "L2_preprocess": {
            "missing_value_strategy": ["interpolate_time", "mean"],
            "outlier_detect": ["iqr"],
            "outlier_handle": ["clip_1_99"],
            "normalization": ["zscore", "robust"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["tree_light", "tree_heavy"]},
    },

    "Transformer": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_linear"],
            "normalization": ["zscore"],
            "target_transform": ["none"],
        },
        "L3_features": {"recipe": ["time_only", "tree_light"]},
    },
    "LSTM": {
        "L2_preprocess": {
            "time_index_regularize": ["on"],
            "missing_value_strategy": ["interpolate_linear", "forward_fill"],
            "normalization": ["zscore"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["time_only", "tree_light"]},
    },
    "NeuralNetwork": {
        "L2_preprocess": {
            "missing_value_strategy": ["interpolate_linear", "mean"],
            "normalization": ["zscore", "robust"],
            "target_transform": ["none", "log1p"],
        },
        "L3_features": {"recipe": ["tree_light", "tree_heavy"]},
    },
}

# Reuse
MODEL_ACTION_SPACE["ARIMA"] = MODEL_ACTION_SPACE["SARIMA"]
for _m in ["LinearRegression", "PolynomialRegression", "RidgeRegression", "LassoRegression"]:
    MODEL_ACTION_SPACE[_m] = MODEL_ACTION_SPACE["ElasticNet"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model_conditioned_spec(model_name: str, layer: str) -> Dict[str, List]:
    model_spec = MODEL_ACTION_SPACE.get(model_name, {})
    return dict(model_spec.get(layer, {}))

def _extract_model_from_path(action_path: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not action_path:
        return None
    for action in action_path:
        if action.get("layer") == "L1_model":
            return (action.get("params") or {}).get("model")
    return None

def get_conditioned_action_spec(
    layer: str,
    action_path: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List]:
    if layer == "L1_model":
        return {"model": list(ALL_MODEL_NAMES)}

    model_name = _extract_model_from_path(action_path)
    if model_name is None:
        return {}
    return get_model_conditioned_spec(model_name, layer)

def get_action_space(layer: str = None) -> Dict[str, Any]:
    if layer:
        return ACTION_SPACE.get(layer, {})
    return ACTION_SPACE

def get_layer_action_spec(layer: str, action_space: Dict[str, Any] = None) -> Dict[str, List[Any]]:
    space = action_space or ACTION_SPACE
    layer_spec = space.get(layer, {})
    actions_spec = layer_spec.get("actions", {})
    if not actions_spec:
        return {}

    spec: Dict[str, List[Any]] = {}
    for key, meta in actions_spec.items():
        spec[key] = list(meta.get("options", []))
    return spec

def sample_action(layer: str, use_random: bool = True, action_path: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    import random as _rnd

    spec = get_conditioned_action_spec(layer, action_path)
    if not spec:
        spec = get_layer_action_spec(layer)
    if not spec:
        return {}

    action = {}
    for key, opts in spec.items():
        action[key] = _rnd.choice(opts) if (use_random and opts) else (opts[0] if opts else None)
    return action

# ---------------------------------------------------------------------------
# Dynamic feature recipe factory
# ---------------------------------------------------------------------------
_FREQ_CYCLES = {
    "hourly":  (24, 168),
    "15min":   (96, 672),
    "daily":   (1, 7),
    "weekly":  (1, 4),
    "monthly": (1, 12),
}

def generate_dynamic_recipe(
    freq_hint_std: str,
    recipe_level: str,
    suspected_seasonal_periods: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Generate a concrete feature configuration from a recipe level and frequency."""
    if recipe_level == "none" or not recipe_level:
        return {}

    day_cycle, week_cycle = _FREQ_CYCLES.get(freq_hint_std, (24, 168))
    periods = list(suspected_seasonal_periods or [])
    if not periods:
        periods = [day_cycle]
        if week_cycle != day_cycle:
            periods.append(week_cycle)

    config: Dict[str, Any] = {
        "lags": [],
        "rolling_windows": [],
        "rolling_stats": [],
        "fourier_periods": [],
        "time_features": False,
        "ewma_spans": [],
        "momentum": False,
    }

    if recipe_level == "time_only":
        config["time_features"] = True
        config["fourier_periods"] = periods[:2]
        return config

    if recipe_level in ("tree_light", "tree_heavy", "tree_extreme"):
        config["time_features"] = True
        config["lags"] = sorted(set([1, 2, 3, day_cycle]))
        config["rolling_windows"] = [day_cycle]
        config["rolling_stats"] = ["mean", "std"]
        config["fourier_periods"] = periods[:2]

    if recipe_level in ("tree_heavy", "tree_extreme"):
        extra_lags = [week_cycle] if week_cycle != day_cycle else []
        config["lags"] = sorted(set(config["lags"] + extra_lags))
        half_day = max(1, day_cycle // 2)
        config["rolling_windows"] = sorted(set([half_day, day_cycle, week_cycle]))
        config["rolling_stats"] = ["mean", "std", "min", "max"]
        config["ewma_spans"] = [day_cycle]

    if recipe_level == "tree_extreme":
        extra_lags = [day_cycle * 2, week_cycle * 2] if week_cycle != day_cycle else [day_cycle * 2]
        config["lags"] = sorted(set(config["lags"] + extra_lags))
        config["rolling_windows"] = sorted(set(config["rolling_windows"] + [week_cycle]))
        config["rolling_stats"] = ["mean", "std", "min", "max", "skew"]
        config["momentum"] = True

    return config