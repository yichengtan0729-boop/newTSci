"""
MCTS search implementation for time series pipeline selection.

Current funnel layer semantics:
    L1_model       -> choose forecasting model
    L2_preprocess  -> choose preprocessing recipe
    L3_features    -> choose feature engineering recipe

LangGraph is used ONLY as the control-flow loop:
Select -> Expand -> Simulate -> Backpropagate
The dynamic tree (nodes/edges) is maintained in Python.
"""

from __future__ import annotations

import math
import time
import random
import uuid
import numpy as np
from dataclasses import dataclass, field
from itertools import product, combinations
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, END

from .action_space import ACTION_SPACE, get_layer_action_spec
from utils.progress import vprint


Action = Dict[str, Any]
ActionPath = List[Action]


@dataclass
class Node:
    node_id: str
    parent_id: Optional[str]
    depth: int
    action: Optional[Action] = None
    action_path: ActionPath = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    max_reward: float = -float("inf")
    untried_actions: List[Action] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Mean reward, kept mainly for diagnostics."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def is_fully_expanded(self, max_children: int) -> bool:
        return len(self.children_ids) >= max_children


@dataclass
class Tree:
    nodes: Dict[str, Node]
    root_id: str

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def add_child(self, parent_id: str, action: Action) -> Node:
        parent = self.get(parent_id)
        node_id = str(uuid.uuid4())
        node = Node(
            node_id=node_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            action=action,
            action_path=parent.action_path + [action],
        )
        self.nodes[node_id] = node
        parent.children_ids.append(node_id)
        return node

    def backpropagate(self, node_id: str, reward: float) -> None:
        """
        Backpropagate reward up the tree.

        Updates:
        - visits
        - value_sum
        - max_reward
        """
        current_id = node_id
        while current_id is not None:
            node = self.get(current_id)
            node.visits += 1
            node.value_sum += reward
            if reward > node.max_reward:
                node.max_reward = reward
            current_id = node.parent_id


class MCTSState(TypedDict):
    tree: Tree
    current_node_id: str
    rollouts_done: int
    max_rollouts: int
    layers: List[str]
    best_reward: float
    best_action_path: ActionPath
    pending_reward: float
    last_action_path: ActionPath
    last_metadata: Dict[str, Any]
    best_candidates: List[Dict[str, Any]]


@dataclass
class MCTSConfig:
    max_rollouts: int = 100
    exploration_weight: float = 1.4
    max_children_per_node: int = 20
    candidate_pool_size: int = 10

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass
class MCTSCallbacks:
    """
    LLM policies use compact spec: {param: [options]}.
    expand_policy(node, tree, layer, spec, expanded_params, context) -> params dict
    rollout_policy(action_path, remaining_layers, get_spec_fn, context) -> action_path
    """

    expand_policy: Optional[
        Callable[[Node, Tree, str, Dict[str, List[Any]], List[Dict[str, Any]], Dict[str, Any]], Dict[str, Any]]
    ] = None
    rollout_policy: Optional[
        Callable[[ActionPath, List[str], Callable[[str], Dict[str, List[Any]]], Dict[str, Any]], ActionPath]
    ] = None


def uct_score(parent_visits: int, child: Node, exploration_weight: float) -> float:
    """
    Max-UCB score for AutoML pipeline search.

    Use max_reward rather than mean_reward because we want the single best
    pipeline, not the best average policy.
    """
    if child.visits == 0:
        return float("inf")
    exploitation = child.max_reward
    exploration = exploration_weight * math.sqrt(math.log(parent_visits + 1) / child.visits)
    return exploitation + exploration


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

# Current funnel naming:
#   L1_model       -> forbid model choices
#   L2_preprocess  -> forbid preprocess actions
#   L3_features    -> forbid feature actions
_LAYER_CONSTRAINT_KEY = {
    "L1_model": "forbidden_L3_models",
    "L2_preprocess": "forbidden_L1_actions",
    "L3_features": "forbidden_L2_actions",
}


def _normalize_constraint_value(v: Any) -> str:
    if isinstance(v, (list, tuple, set)):
        return "|".join(str(x) for x in v)
    return str(v)


def _apply_constraints(
    spec: Dict[str, List[Any]],
    layer: str,
    constraints: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """
    Filter a layer's action spec by removing forbidden values.
    """
    if not constraints:
        return spec

    forbidden_key = _LAYER_CONSTRAINT_KEY.get(layer)
    if not forbidden_key:
        return spec

    forbidden: List[Any] = constraints.get(forbidden_key, [])
    if not forbidden:
        return spec

    forbidden_set = set(_normalize_constraint_value(v) for v in forbidden)

    filtered: Dict[str, List[Any]] = {}
    for param, opts in spec.items():
        clean = [v for v in opts if _normalize_constraint_value(v) not in forbidden_set]

        if not clean:
            if "none" in opts:
                clean = ["none"]
            else:
                clean = [opts[0]] if opts else []

            vprint(
                "MCTS",
                "  Constraint: layer=%s param=%s — ALL options forbidden, fallback=%s",
                layer,
                param,
                clean,
            )
        elif len(clean) < len(opts):
            removed = [v for v in opts if _normalize_constraint_value(v) in forbidden_set]
            vprint(
                "MCTS",
                "  Constraint: layer=%s param=%s — removed %s, remaining=%s",
                layer,
                param,
                removed,
                clean,
            )

        filtered[param] = clean

    return filtered


def default_rollout_policy(
    action_path: ActionPath,
    remaining_layers: List[str],
    get_layer_action_spec_fn: Callable[[str], Dict[str, List[Any]]],
    context: Dict[str, Any],
) -> ActionPath:
    constraints = context.get("mcts_constraints", {}) if context else {}

    for layer in remaining_layers:
        spec = get_layer_action_spec_fn(layer)
        if not spec:
            continue
        spec = _apply_constraints(spec, layer, constraints)
        params = {k: random.choice(opts) for k, opts in spec.items()}
        action_path.append({"layer": layer, "params": params})

    return action_path


class MCTSRunner:
    """
    MCTS runner that uses LangGraph for control flow.

    Provide:
    - get_layer_action_spec(layer) -> {param: [options]}
    - simulate(action_path) -> (reward, metadata)
    """

    def __init__(
        self,
        layers: List[str],
        simulate: Callable[[ActionPath], Tuple[float, Dict[str, Any]]],
        get_layer_action_spec_fn: Optional[Callable[[str], Dict[str, List[Any]]]] = None,
        config: Optional[MCTSConfig] = None,
        callbacks: Optional[MCTSCallbacks] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.layers = layers
        self.get_layer_action_spec = get_layer_action_spec_fn or get_layer_action_spec
        self.simulate_fn = simulate
        self.config = config or MCTSConfig()
        self.callbacks = callbacks or MCTSCallbacks()
        self.context = context or {}

        self.context.setdefault("mcts_constraints", {})
        self.context.setdefault("hard_constraint_penalty", -1e6)
        self.context.setdefault("backprop_reward_floor", -1000.0)

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MCTSState)
        workflow.add_node("select", self._select_node)
        workflow.add_node("expand", self._expand_node)
        workflow.add_node("simulate", self._simulate_node)
        workflow.add_node("backpropagate", self._backpropagate)

        workflow.add_edge("select", "expand")
        workflow.add_edge("expand", "simulate")
        workflow.add_edge("simulate", "backpropagate")
        workflow.add_conditional_edges(
            "backpropagate",
            self._rollout_condition,
            {
                "loop": "select",
                "end": END,
            },
        )
        workflow.set_entry_point("select")
        return workflow.compile()

    def _rollout_condition(self, state: MCTSState) -> str:
        if state["rollouts_done"] < state["max_rollouts"]:
            return "loop"
        return "end"

    def _select_node(self, state: MCTSState) -> Dict[str, Any]:
        tree = state["tree"]
        current_id = tree.root_id
        rollout_num = state.get("rollouts_done", 0) + 1

        while True:
            node = tree.get(current_id)

            if node.depth >= len(state["layers"]):
                break

            if not node.is_fully_expanded(self.config.max_children_per_node) or node.is_leaf():
                break

            parent_visits = node.visits
            best_child_id = None
            best_score = float("-inf")

            for child_id in node.children_ids:
                child = tree.get(child_id)
                score = uct_score(parent_visits, child, self.config.exploration_weight)
                if score > best_score:
                    best_score = score
                    best_child_id = child_id

            if best_child_id is None:
                break

            current_id = best_child_id

        sel_node = tree.get(current_id)
        vprint(
            "MCTS",
            "Rollout %d/%d | Select: depth=%d, visits=%d, children=%d",
            rollout_num,
            state["max_rollouts"],
            sel_node.depth,
            sel_node.visits,
            len(sel_node.children_ids),
        )
        return {"current_node_id": current_id}

    def _expand_node(self, state: MCTSState) -> Dict[str, Any]:
        tree = state["tree"]
        node = tree.get(state["current_node_id"])

        if node.depth >= len(state["layers"]):
            return {}

        if node.is_fully_expanded(self.config.max_children_per_node):
            return {}

        layer = state["layers"][node.depth]
        spec = self.get_layer_action_spec(layer)
        if not spec:
            return {}

        constraints = self.context.get("mcts_constraints", {})
        if constraints:
            spec = _apply_constraints(spec, layer, constraints)

        expanded_params = [
            tree.get(cid).action.get("params", {}) for cid in node.children_ids
            if tree.get(cid).action
        ]

        use_llm = bool(self.callbacks.expand_policy)
        if self.callbacks.expand_policy:
            vprint("MCTS", "  Expand: layer=%s via LLM (already expanded=%d)", layer, len(expanded_params))
            params = self.callbacks.expand_policy(
                node, tree, layer, spec, expanded_params, self.context
            )
        else:
            params = {k: random.choice(opts) for k, opts in spec.items()}

        action = {"layer": layer, "params": params}

        if any(p == params for p in expanded_params):
            vprint("MCTS", "  Expand: layer=%s -> DUPLICATE params, skip", layer)
            return {}

        child = tree.add_child(node.node_id, action)
        vprint("MCTS", "  Expand: layer=%s -> %s (llm=%s)", layer, params, use_llm)
        return {"current_node_id": child.node_id}

    def _simulate_node(self, state: MCTSState) -> Dict[str, Any]:
        tree = state["tree"]
        node = tree.get(state["current_node_id"])
        action_path = list(node.action_path)

        remaining_layers = state["layers"][node.depth:]
        use_llm_rollout = bool(self.callbacks.rollout_policy)
        if remaining_layers:
            vprint(
                "MCTS",
                "  Rollout: completing %d remaining layers %s (llm=%s)",
                len(remaining_layers),
                remaining_layers,
                use_llm_rollout,
            )

        if self.callbacks.rollout_policy:
            action_path = self.callbacks.rollout_policy(
                action_path, remaining_layers, self.get_layer_action_spec, self.context
            )
        else:
            action_path = default_rollout_policy(
                action_path, remaining_layers, self.get_layer_action_spec, self.context
            )

        path_summary = {a.get("layer"): a.get("params", {}) for a in action_path}
        vprint("MCTS", "  Simulate: action_path=%s", path_summary)

        constraints = self.context.setdefault("mcts_constraints", {})
        forbidden_models = set(str(m) for m in constraints.get("forbidden_L3_models", []))
        target_mape = self.context.get("target_mape", None)

        start = time.time()
        reward, metadata = self.simulate_fn(action_path)
        elapsed = time.time() - start

        metadata = dict(metadata)
        models_used = metadata.get("selected_models", [])
        mape = metadata.get("mape", None)

        # -----------------------------
        # Soft constraint 1:
        # reused model already marked as undesirable
        # -----------------------------
        hit_forbidden_model = False
        if models_used and forbidden_models:
            for m in models_used:
                if str(m) in forbidden_models:
                    hit_forbidden_model = True
                    break

        if hit_forbidden_model:
            metadata["soft_constraint_triggered"] = True
            metadata["soft_constraint_reason"] = "forbidden_model_reused"
            reward = float(reward) - 20.0

            vprint(
                "MCTS",
                "  SOFT CONSTRAINT | reused undesirable model(s)=%s | reward=%.6f",
                models_used,
                reward,
            )

        # -----------------------------
        # Soft constraint 2:
        # MAPE threshold penalty
        # -----------------------------
        elif mape is not None and target_mape is not None and np.isfinite(mape):
            excess = max(0.0, float(mape) - float(target_mape))
            if excess > 0.0:
                metadata["soft_constraint_triggered"] = True
                metadata["soft_constraint_reason"] = "mape_threshold"
                metadata["mape_excess"] = excess

                # soft penalty only; do NOT permanently forbid the model
                reward = float(reward) - 10.0 * excess

                vprint(
                    "MCTS",
                    "  SOFT CONSTRAINT | mape=%.6f > %.6f | excess=%.6f | reward=%.6f",
                    float(mape),
                    float(target_mape),
                    float(excess),
                    reward,
                )

        vprint(
            "MCTS",
            "  Simulate: reward=%.6f, mape=%s, models=%s, time=%.1fs",
            reward,
            f"{mape:.6f}" if isinstance(mape, (int, float, np.floating)) and np.isfinite(mape) else str(mape),
            models_used,
            elapsed,
        )

        node.metadata["last_reward"] = reward
        node.metadata["last_metadata"] = metadata
        node.metadata["last_action_path"] = list(action_path)

        return {
            "pending_reward": reward,
            "last_simulation_time": elapsed,
            "last_action_path": action_path,
            "last_metadata": metadata,
        }

    def _model_type_key(self, action_path: ActionPath, metadata: Dict[str, Any]) -> str:
        """
        Diversity key aligned to current funnel naming.

        Keep candidates distinct by:
        - chosen model branch (L1_model)
        - preprocess recipe (L2_preprocess)
        - feature recipe (L3_features)
        - actual selected_models returned by simulation

        This avoids collapsing many candidates into only one entry.
        """
        layer_params = {a.get("layer"): a.get("params", {}) for a in action_path}

        l1 = layer_params.get("L1_model", {})
        l2 = layer_params.get("L2_preprocess", {})
        l3 = layer_params.get("L3_features", {})

        l1_key = tuple(sorted((str(k), str(v)) for k, v in l1.items()))
        l2_key = tuple(sorted((str(k), str(v)) for k, v in l2.items()))
        l3_key = tuple(sorted((str(k), str(v)) for k, v in l3.items()))
        selected_models = tuple(sorted(str(m) for m in metadata.get("selected_models", [])))

        return f"L1={l1_key}|L2={l2_key}|L3={l3_key}|SEL={selected_models}"

    def _update_candidate_pool(
        self,
        best_candidates: List[Dict[str, Any]],
        action_path: ActionPath,
        reward: float,
        metadata: Dict[str, Any],
        pool_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Keep top-K diverse candidates.

        Only replace an existing candidate when the diversity key is the same.
        """
        model_key = self._model_type_key(action_path, metadata)
        candidate = {
            "action_path": list(action_path),
            "reward": reward,
            "model_type_key": model_key,
            "metadata": metadata,
        }

        new_pool = list(best_candidates)
        replaced = False

        for i, c in enumerate(new_pool):
            if c.get("model_type_key") == model_key:
                if reward > c.get("reward", float("-inf")):
                    new_pool[i] = candidate
                replaced = True
                break

        if not replaced:
            new_pool.append(candidate)

        new_pool.sort(key=lambda x: x.get("reward", float("-inf")), reverse=True)
        return new_pool[:pool_size]

    def _backpropagate(self, state: MCTSState) -> Dict[str, Any]:
        tree = state["tree"]
        reward = state["pending_reward"]
        node_id = state["current_node_id"]

        reward_floor = float(self.context.get("backprop_reward_floor", -1000.0))
        reward = max(float(reward), reward_floor)

        tree.backpropagate(node_id, reward)

        best_reward = state.get("best_reward", float("-inf"))
        is_new_best = reward > best_reward
        if is_new_best:
            best_reward = reward
            best_action_path = state.get("last_action_path", [])
        else:
            best_action_path = state.get("best_action_path", [])

        best_candidates = state.get("best_candidates", [])
        last_metadata = state.get("last_metadata", {})
        last_action_path = state.get("last_action_path", [])

        best_candidates = self._update_candidate_pool(
            best_candidates,
            last_action_path,
            reward,
            last_metadata,
            self.config.candidate_pool_size,
        )

        done = state["rollouts_done"] + 1
        new_best_tag = " *** NEW BEST ***" if is_new_best else ""

        vprint(
            "MCTS",
            "  Backprop: rollout %d/%d done, reward=%.6f, best=%.6f, pool=%d%s",
            done,
            state["max_rollouts"],
            reward,
            best_reward,
            len(best_candidates),
            new_best_tag,
        )

        return {
            "rollouts_done": done,
            "best_reward": best_reward,
            "best_action_path": best_action_path,
            "best_candidates": best_candidates,
        }

    def run(self) -> Dict[str, Any]:
        constraints = self.context.get("mcts_constraints", {})
        has_constraints = any(
            constraints.get(k)
            for k in ("forbidden_L1_actions", "forbidden_L2_actions", "forbidden_L3_models")
        )

        vprint(
            "MCTS",
            "Starting MCTS: layers=%s, max_rollouts=%d, pool_size=%d, constraints=%s, target_mape=%s",
            self.layers,
            self.config.max_rollouts,
            self.config.candidate_pool_size,
            constraints if has_constraints else "none",
            self.context.get("target_mape", None),
        )

        root = Node(node_id=str(uuid.uuid4()), parent_id=None, depth=0)
        tree = Tree(nodes={root.node_id: root}, root_id=root.node_id)

        init_state: MCTSState = {
            "tree": tree,
            "current_node_id": root.node_id,
            "rollouts_done": 0,
            "max_rollouts": self.config.max_rollouts,
            "layers": self.layers,
            "best_reward": float("-inf"),
            "best_action_path": [],
            "pending_reward": 0.0,
            "last_action_path": [],
            "last_metadata": {},
            "best_candidates": [],
        }

        t0 = time.time()
        final_state = self.graph.invoke(
            init_state,
            config={"recursion_limit": self.config.get("recursion_limit", 2000)},
        )
        elapsed = time.time() - t0

        vprint(
            "MCTS",
            "MCTS complete: best_reward=%.6f, rollouts=%d, tree_nodes=%d, time=%.1fs",
            final_state["best_reward"],
            final_state["rollouts_done"],
            len(final_state["tree"].nodes),
            elapsed,
        )

        return {
            "best_reward": final_state["best_reward"],
            "best_action_path": final_state["best_action_path"],
            "best_candidates": final_state.get("best_candidates", []),
            "rollouts_done": final_state["rollouts_done"],
            "tree": final_state["tree"],
        }


def _expand_action_space_options(actions_spec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = list(actions_spec.keys())
    options_list: List[Iterable[Any]] = []

    for key in keys:
        options = actions_spec[key].get("options", [])
        if key == "paradigms":
            combos = []
            for r in range(1, len(options) + 1):
                combos.extend(combinations(options, r))
            options_list.append(combos)
        else:
            options_list.append(options)

    actions = []
    for combo in product(*options_list):
        params = {k: v for k, v in zip(keys, combo)}
        actions.append(params)
    return actions


def get_layer_actions(layer: str, action_space: Dict[str, Any] = ACTION_SPACE) -> List[Action]:
    """
    Build all candidate actions for a given layer.

    Returns:
        {"layer": layer, "params": {...}}
    """
    layer_spec = action_space.get(layer, {})
    actions_spec = layer_spec.get("actions", {})
    if not actions_spec:
        return []
    params_list = _expand_action_space_options(actions_spec)
    return [{"layer": layer, "params": params} for params in params_list]


_DEEP_MODELS = ("LSTM", "NeuralNetwork", "Transformer")


# ---------------------------------------------------------------------------
# Stage-1 fast reward: last-block validation
# ---------------------------------------------------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Return MAPE in PERCENT units, consistent with funnel report output.
    Example: 9.11 means 9.11%
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    n = min(len(y_true), len(y_pred))
    if n == 0:
        return float("inf")

    y_true = y_true[:n]
    y_pred = y_pred[:n]

    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _get_fast_reward(
    data: Any,
    horizon: int,
    selected_models: List[str],
    model_params: Dict[str, Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Last-block validation reward for MCTS simulation (single-model).

    sub_train = train_data[:N-horizon]
    sub_val   = train_data[N-horizon:]

    Reward basis = -MSE
    Metadata includes mse/mae/mape (mape in percent units).
    """
    from utils.validation import last_block_split
    from utils.model_library import get_model_function

    if isinstance(data, dict):
        series = np.asarray(data.get("value", [])).flatten()
        series_original = data.get("value_original")
        if series_original is not None:
            series_original = np.asarray(series_original).flatten()
        scaler = data.get("scaler")
    else:
        series = np.asarray(data).flatten()
        series_original = None
        scaler = None

    if hasattr(data, "values") and not isinstance(data, dict):
        series = np.asarray(data.values).flatten()
        series_original = None
        scaler = None

    sub_train, sub_val = last_block_split(series, horizon)
    pred_len = len(sub_val)
    cut = len(sub_train)

    if series_original is not None and len(series_original) == len(series):
        _, sub_val_true = last_block_split(series_original, horizon)
        sub_val_true = sub_val_true[:pred_len]
    else:
        sub_val_true = sub_val[:pred_len]

    train_dict: Dict[str, Any] = {"value": sub_train}
    if isinstance(data, dict):
        for k, v in data.items():
            if k in ("value", "value_original", "scaler"):
                continue
            arr = np.asarray(v).flatten()
            if len(arr) == len(series):
                train_dict[k] = arr[:cut]

    predictions: Dict[str, List[float]] = {}

    for m in selected_models:
        try:
            fn = get_model_function(m)
            preds = fn(train_dict, model_params.get(m, {}), pred_len)
            preds = np.asarray(preds).flatten()[:pred_len]

            if scaler is not None and hasattr(scaler, "inverse_transform"):
                preds = scaler.inverse_transform(preds)

            predictions[m] = preds.tolist()
        except Exception:
            fallback = float(sub_train[-1]) if len(sub_train) > 0 else 0.0
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                fallback = float(scaler.inverse_transform(np.array([fallback]))[0])
            predictions[m] = [fallback] * pred_len

    if predictions:
        single_pred = np.asarray(list(predictions.values())[0]).flatten()[:pred_len]
        n = min(len(sub_val_true), len(single_pred))

        y_true = np.asarray(sub_val_true[:n], dtype=float)
        y_pred = np.asarray(single_pred[:n], dtype=float)

        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = _safe_mape(y_true, y_pred)
        accuracy_score = -mse
    else:
        mse = float("inf")
        mae = float("inf")
        mape = float("inf")
        accuracy_score = float("-inf")

    return accuracy_score, {
        "predictions": predictions,
        "sub_val": sub_val_true.tolist() if series_original is not None else sub_val.tolist(),
        "mse": mse,
        "mae": mae,
        "mape": mape,
    }


@dataclass
class SimulationContext:
    """
    Concrete pipeline simulation context for current funnel:

    L1_model       -> select forecasting model
    L2_preprocess  -> preprocess data
    L3_features    -> build features
    """

    data: Any
    horizon: int
    apply_preprocess: Optional[Callable[[Any, Dict[str, Any]], Any]] = None
    apply_features: Optional[Callable[[Any, Dict[str, Any]], Any]] = None
    select_models: Optional[
        Callable[[Any, Dict[str, Any]], Tuple[List[str], Dict[str, Dict[str, Any]]]]
    ] = None
    train_predict: Optional[
        Callable[[Any, List[str], Dict[str, Dict[str, Any]], int], Dict[str, List[float]]]
    ] = None
    score: Optional[Callable[[Any, Dict[str, Any]], float]] = None
    diversity_bonus: Optional[Callable[[Dict[str, List[float]]], float]] = None
    time_penalty_alpha: float = 0.01
    tuning_agent: Any = None
    tuning_agent_config: Optional[Dict[str, Any]] = None
    use_tuning_in_simulation: bool = False
    fast_simulation_max_epochs: int = 5

    # Deprecated
    analysis_fn: Optional[Callable[[Dict[str, Any], Any], str]] = None


def simulate_action_path(
    action_path: ActionPath,
    context: SimulationContext,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a concrete pipeline simulation for one action path.

    Current layer semantics:
        L1_model       -> model
        L2_preprocess  -> preprocess
        L3_features    -> features
    """
    layer_params = {a.get("layer"): a.get("params", {}) for a in action_path}

    start_time = time.time()
    data = context.data

    try:
        import json
        from pathlib import Path
        _p = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
        with open(_p, "a") as _f:
            _f.write(json.dumps({
                "timestamp": int(time.time() * 1000),
                "location": "mcts_search.py:simulate",
                "message": "simulate_entry",
                "data": {"path_layers": [a.get("layer") for a in action_path]},
                "hypothesisId": "H5"
            }) + "\n")
    except Exception:
        pass

    # L2_preprocess
    if context.apply_preprocess and layer_params.get("L2_preprocess"):
        try:
            import json
            from pathlib import Path
            _p = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
            with open(_p, "a") as _f:
                _f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "mcts_search.py:simulate_L2_preprocess",
                    "message": "simulate_L2_preprocess_start",
                    "data": {},
                    "hypothesisId": "H5"
                }) + "\n")
        except Exception:
            pass

        vprint("SIM", "    L2 preprocess: %s", layer_params["L2_preprocess"])
        data = context.apply_preprocess(data, layer_params["L2_preprocess"])

    # L3_features
    if context.apply_features and layer_params.get("L3_features"):
        try:
            import json
            from pathlib import Path
            _p = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
            with open(_p, "a") as _f:
                _f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "mcts_search.py:simulate_L3_features",
                    "message": "simulate_L3_features_start",
                    "data": {},
                    "hypothesisId": "H5"
                }) + "\n")
        except Exception:
            pass

        vprint("SIM", "    L3 features: %s", layer_params["L3_features"])
        data = context.apply_features(data, layer_params["L3_features"])

    # L1_model
    selected_models: List[str] = []
    model_params: Dict[str, Dict[str, Any]] = {}

    if context.select_models and layer_params.get("L1_model"):
        try:
            import json
            from pathlib import Path
            _p = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
            with open(_p, "a") as _f:
                _f.write(json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "location": "mcts_search.py:simulate_L1_model",
                    "message": "simulate_L1_model_start",
                    "data": {},
                    "hypothesisId": "H5"
                }) + "\n")
        except Exception:
            pass

        selected_models, model_params = context.select_models(data, layer_params["L1_model"])
        vprint(
            "SIM",
            "    L1 model: %s -> selected=%s",
            layer_params["L1_model"],
            selected_models,
        )

    tuning_used = False
    if context.use_tuning_in_simulation and context.tuning_agent and selected_models:
        from agents.tuning_agent import TuningContext as TuningCtx, _default_train_trial

        tuning_cfg = context.tuning_agent_config or {}
        tuning_ctx = TuningCtx(
            data=data,
            models=selected_models,
            horizon=context.horizon,
            train_fn=_default_train_trial,
            max_trials=tuning_cfg.get("max_trials", 6),
            max_epochs_per_trial=tuning_cfg.get("max_epochs_per_trial", 15),
        )
        best_params_per_model, _, _ = context.tuning_agent.run(tuning_ctx)
        for m in selected_models:
            model_params[m] = best_params_per_model.get(m, model_params.get(m, {}))
        tuning_used = True
    else:
        max_epochs = getattr(context, "fast_simulation_max_epochs", 5)

        if context.tuning_agent and hasattr(context.tuning_agent, "suggest_params_for_fast_sim"):
            try:
                suggested = context.tuning_agent.suggest_params_for_fast_sim(
                    selected_models,
                    context.horizon,
                    max_epochs=max_epochs,
                )
                for m in selected_models:
                    model_params[m] = suggested.get(m, model_params.get(m, {}))
            except Exception:
                pass

        for m in selected_models:
            p = model_params.get(m, {})
            if m in _DEEP_MODELS:
                p = {**p, "epochs": min(p.get("epochs", max_epochs), max_epochs)}
            model_params[m] = p

    accuracy_score, fast_meta = _get_fast_reward(
        data,
        context.horizon,
        selected_models,
        model_params,
    )

    predictions: Dict[str, List[float]] = fast_meta.get("predictions", {})
    ensemble_output: Dict[str, Any] = fast_meta.get("ensemble_output", {})
    elapsed = time.time() - start_time

    diversity = 0.0
    if context.diversity_bonus and predictions:
        try:
            diversity = float(context.diversity_bonus(predictions))
        except Exception:
            diversity = 0.0

    reward = accuracy_score - context.time_penalty_alpha * math.log(elapsed + 1.0) + diversity

    metadata = {
        "accuracy_score": accuracy_score,
        "time_cost": elapsed,
        "diversity_bonus": diversity,
        "ensemble_output": ensemble_output,
        "selected_models": selected_models,
        "model_params": model_params,
        "tuning_agent_used": tuning_used,
        "predictions": predictions,
        "mse": fast_meta.get("mse", float("inf")),
        "mae": fast_meta.get("mae", float("inf")),
        "mape": fast_meta.get("mape", float("inf")),
        "sub_val": fast_meta.get("sub_val", []),
    }
    return reward, metadata