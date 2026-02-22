"""
LLM-guided policies for MCTS action selection.

Uses compact spec {param: [options]}. LLM picks one value per param.
- expand_policy: pick params using sibling performance
- rollout_policy: complete remaining layers with LLM

Uses LangChain with_structured_output and dynamically generated Pydantic
models with a required reasoning (CoT) field.
"""

from __future__ import annotations
import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field, create_model
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_factory import get_llm
from .action_space import MODEL_PARADIGM
from .mcts_search import ActionPath, Node, Tree
from utils.progress import vprint

logger = logging.getLogger(__name__)
def _invoke_with_backoff(runnable, messages, *, max_retries: int = 8):
    """
    Retry LLM calls on 429 / rate limit with exponential backoff.
    Works for LangChain chat models (invoke([...])).
    """
    for i in range(max_retries):
        try:
            return runnable.invoke(messages)
        except Exception as e:
            msg = str(e)
            is_429 = ("429" in msg) or ("Too Many Requests" in msg) or ("rate limit" in msg.lower())
            if not is_429:
                raise
            sleep_s = min(60.0, (2 ** i) + random.random())
            logger.warning("LLM rate-limited (429). Sleeping %.1fs then retrying (%d/%d)...", sleep_s, i + 1, max_retries)
            time.sleep(sleep_s)
    raise RuntimeError("LLM rate-limited too many times (429).")



def _format_spec(spec: Dict[str, List]) -> str:
    """Compact format: param -> [options]. LLM picks one per param."""
    lines = []
    for k, opts in spec.items():
        lines.append(f"- {k}: {opts}")
    return "\n".join(lines)


def _format_model_catalog() -> str:
    """Format MODEL_PARADIGM as a readable catalog for LLM."""
    lines = ["Available models per paradigm:"]
    for paradigm, models in MODEL_PARADIGM.items():
        lines.append(f"  {paradigm}: {models}")
    return "\n".join(lines)


@dataclass
class LLMPolicyConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 800


class LLMPolicyFactory:
    """
    Factory for LLM-driven policies. You can reuse a single instance.
    Accepts full config dict (for provider switch) or LLMPolicyConfig.
    """

    def __init__(self, config: Optional[Any] = None):
        if isinstance(config, dict):
            cfg = {**config, "llm_model": config.get("llm_model", "gemini-2.5-flash")}
            cfg.setdefault("llm_max_tokens", 800)
            self.llm = get_llm(cfg)
        elif isinstance(config, LLMPolicyConfig):
            cfg = {
                "llm_model": config.model,
                "llm_temperature": config.temperature,
                "llm_max_tokens": config.max_tokens,
            }
            self.llm = get_llm(cfg)
        else:
            c = config or LLMPolicyConfig()
            self.llm = get_llm({
                "llm_model": c.model,
                "llm_temperature": c.temperature,
                "llm_max_tokens": c.max_tokens,
            })

    def expand_policy(self) -> Callable[[Node, Tree, str, Dict[str, List], List[Dict], Dict[str, Any]], Dict[str, Any]]:
        """
        Pick one value per param from spec. Uses sibling performance to avoid duplicates.
        Uses with_structured_output and a dynamic Pydantic schema with reasoning (CoT).
        """
        def _choose(
            node: Node,
            tree: Tree,
            layer: str,
            spec: Dict[str, List],
            expanded_params: List[Dict],
            context: Dict[str, Any],
        ) -> Dict[str, Any]:
            if not spec:
                return {}
            analysis = context.get("analysis_result", "")
            spec_text = _format_spec(spec)
            tried_text = f"Already expanded paths: {expanded_params}" if expanded_params else "No paths expanded yet. You are the first."
            model_catalog_text = f"\n{_format_model_catalog()}\n" if "L3" in layer else ""

            field_definitions: Dict[str, Any] = {
                "reasoning": (str, Field(description="Step-by-step reasoning based on data analysis and expanded_params; explain why this parameter combination is promising.")),
            }
            for k, opts in spec.items():
                field_definitions[k] = (Any, Field(description=f"Choose EXACTLY ONE from: {opts}"))

            DynamicActionSchema = create_model(f"ActionSchema_{layer}", **field_definitions)
            structured_llm = self.llm.with_structured_output(DynamicActionSchema, method="function_calling")


            system_msg = (
                "You are an Expert Time Series Machine Learning Architect guiding an MCTS search process. "
                "Your task is to select optimal pipeline configurations strictly following the parameter specifications. "
                "Always provide reasoning first, then the chosen parameters."
            )
            prompt = f"""
We are expanding the MCTS search tree for layer '{layer}'.
Data Analysis Context: {analysis}
{model_catalog_text}

Available Parameters and Options:
{spec_text}

{tried_text}

INSTRUCTIONS:
1. Analyze the Data Analysis Context (e.g., trend, seasonality, stationarity).
2. Considering what has already been tried, propose a highly promising NEW parameter combination for this layer that suits the data characteristics.
3. You MUST pick exactly ONE value from the provided options for EACH parameter.
4. First provide your reasoning, then the parameter choices.
"""

            vprint("LLM", "Expand policy: calling LLM for layer=%s (expanded=%d)", layer, len(expanded_params))
            try:
                response_obj = _invoke_with_backoff(structured_llm, [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt),
                ])
                payload = response_obj.model_dump()
            except Exception as e:
                logger.warning("Expand policy: structured LLM call failed: %s", e)
                vprint("LLM", "Expand policy: LLM failed, using defaults")
                return {k: opts[0] for k, opts in spec.items()}

            reasoning_trace = payload.pop("reasoning", "No reasoning provided.")
            logger.debug("Expand policy reasoning (%s): %s", layer, reasoning_trace)
            vprint("LLM", "Expand policy reasoning: %s", reasoning_trace[:200] + "..." if len(reasoning_trace) > 200 else reasoning_trace)

            params = {}
            for k, opts in spec.items():
                v = payload.get(k, opts[0])
                v_ok = list(v) if isinstance(v, (list, tuple)) else v
                params[k] = v_ok if v_ok in opts else opts[0]
            vprint("LLM", "Expand policy: LLM chose %s", params)
            return params

        return _choose

    def rollout_policy(self) -> Callable[[ActionPath, List[str], Callable[[str], Dict[str, List]], Dict[str, Any]], ActionPath]:
        """
        Complete remaining layers: LLM picks one value per param for each layer.
        Uses with_structured_output and a dynamic Pydantic schema with reasoning (CoT).
        """
        def _rollout(
            action_path: ActionPath,
            remaining_layers: List[str],
            get_layer_action_spec_fn: Callable[[str], Dict[str, List]],
            context: Dict[str, Any],
        ) -> ActionPath:
            analysis = context.get("analysis_result", "")

            system_msg = (
                "You are an Expert Time Series Machine Learning Architect performing a rapid Monte Carlo simulation rollout. "
                "Select the most robust and generally effective parameter values. Always provide reasoning first, then the parameters."
            )

            for layer in remaining_layers:
                spec = get_layer_action_spec_fn(layer)
                if not spec:
                    continue
                spec_text = _format_spec(spec)
                model_catalog_text = f"\n{_format_model_catalog()}\n" if "L3" in layer else ""

                field_definitions = {
                    "reasoning": (str, Field(description="Brief step-by-step reasoning for this fast rollout based on data analysis.")),
                }
                for k, opts in spec.items():
                    field_definitions[k] = (Any, Field(description=f"Choose EXACTLY ONE from: {opts}"))

                DynamicActionSchema = create_model(f"ActionSchema_{layer}", **field_definitions)
                structured_llm = self.llm.with_structured_output(DynamicActionSchema, method="function_calling")


                prompt = f"""
We are completing a rapid simulation rollout for layer '{layer}'.
Data Analysis Context: {analysis}
{model_catalog_text}

Available Parameters and Options:
{spec_text}

INSTRUCTIONS:
Based on the data analysis, select the most robust and generally effective parameter combination to rapidly complete this pipeline.
Pick ONE value per parameter. First provide your reasoning, then the parameter choices.
"""

                vprint("LLM", "Rollout policy: calling LLM for layer=%s", layer)
                try:
                    response_obj = _invoke_with_backoff(structured_llm, [
                        SystemMessage(content=system_msg),
                        HumanMessage(content=prompt),
                    ])
                    payload = response_obj.model_dump()
                except Exception as e:
                    logger.warning("Rollout policy: structured LLM call failed for %s: %s", layer, e)
                    payload = None

                if not payload:
                    params = {k: opts[0] for k, opts in spec.items()}
                    vprint("LLM", "Rollout policy: LLM returned no valid output for %s, using defaults", layer)
                else:
                    reasoning_trace = payload.pop("reasoning", "No reasoning provided.")
                    logger.debug("Rollout policy reasoning (%s): %s", layer, reasoning_trace)
                    vprint("LLM", "Rollout policy reasoning (%s): %s", layer, reasoning_trace[:150] + "..." if len(reasoning_trace) > 150 else reasoning_trace)
                    params = {}
                    for k, opts in spec.items():
                        v = payload.get(k, opts[0])
                        if isinstance(opts[0], list) and isinstance(v, (list, tuple)):
                            params[k] = list(v) if list(v) in opts else opts[0]
                        else:
                            params[k] = v if v in opts else opts[0]
                    vprint("LLM", "Rollout policy: LLM chose %s for %s", params, layer)
                action_path.append({"layer": layer, "params": params})
            return action_path

        return _rollout
