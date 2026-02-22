"""
Analysis Agent for Time Series Prediction

Pre-flight "check-up" agent: runs ONCE on the raw data **before** MCTS starts.
Produces:
  1. A human-readable analysis report (text).
  2. A structured ``mcts_constraints`` dict that tells MCTS which actions to
     prune (forbidden preprocessing steps, feature methods, or model paradigms).
"""

import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from utils.data_utils import DataAnalyzer, DataValidator
from utils.visualization_utils import TimeSeriesVisualizer
from agents.memory import ExperimentMemory
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_factory import get_llm
from utils.progress import vprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default (empty) constraints — forbid nothing
# ---------------------------------------------------------------------------
_EMPTY_CONSTRAINTS: Dict[str, Any] = {
    "forbidden_L1_actions": [],
    "forbidden_L2_actions": [],
    "forbidden_L3_models": [],
}

# ---------------------------------------------------------------------------
# System prompt: analyst + constraint generator
# ---------------------------------------------------------------------------
ANALYSIS_SYSTEM_PROMPT = """
You are the Principal Data Analyst Agent for a state-of-the-art time series forecasting platform.

Background:
- You are an expert in time series statistics, pattern recognition, and exploratory data analysis.
- Your analysis runs ONCE on the raw data BEFORE any model search (MCTS) begins.
- Your output has two purposes:
  1. A human-readable analysis report.
  2. A structured JSON object (`mcts_constraints`) that tells the MCTS search engine
     which pipeline actions are UNSAFE or ILLOGICAL for this particular dataset,
     so it can prune them and avoid wasted computation or runtime errors.

Your responsibilities:
- Provide a comprehensive statistical summary: central tendency, dispersion, skewness, kurtosis.
- Detect and describe trends, seasonality, regime shifts, or anomalies.
- Assess stationarity (ADF / KPSS) and discuss implications.
- Identify forecasting challenges: non-stationarity, structural breaks, data quality issues.
- Based on the above findings, decide which MCTS actions should be FORBIDDEN.

CRITICAL — constraint rules you MUST follow:
- If the data contains any values <= 0, you MUST forbid "log" in L1 stationarity (log transform will crash on non-positive values).
- If the data is already stationary, forbidding "diff" is optional but recommended to avoid over-differencing.
- If the data has very few observations (< 50), consider forbidding "deep" paradigm in L3 (needs more data).
- If the data is non-stationary with strong trend and no differencing is planned, consider forbidding "ARIMA" without diff.
- Only forbid actions when you have a clear statistical justification.

OUTPUT FORMAT — you MUST return a single JSON object (no markdown fences, no extra text outside the JSON):

{
  "insights": "Your full text analysis report as a single string. Include all findings, trend/seasonality/stationarity analysis, and recommendations.",
  "mcts_constraints": {
    "forbidden_L1_actions": ["list of forbidden L1 option values, e.g. 'log', 'diff', 'minmax', 'zscore'"],
    "forbidden_L2_actions": ["list of forbidden L2 option values, e.g. 'fourier'"],
    "forbidden_L3_models": ["list of forbidden paradigm names (e.g. 'deep', 'statistical') OR specific model names (e.g. 'ARIMA', 'LSTM')"]
  }
}

IMPORTANT:
- The JSON must be valid and parseable.
- "forbidden_*" lists can be empty [] if nothing should be forbidden.
- Use EXACTLY the option strings from the action space provided below.
"""


class AnalysisAgent:
    """Pre-flight data analysis agent.

    Runs ONCE on raw data before MCTS.  Returns ``(report_text, constraints_dict)``.
    """

    def __init__(self, model: str = "gemini-2.5-flash", config: dict = None, memory: Optional[ExperimentMemory] = None):
        self.config = config or {}
        cfg = {**self.config, "llm_model": self.config.get("llm_model", model)}
        self.llm = get_llm(cfg)
        self.analyzer = DataAnalyzer()
        self.validator = DataValidator()
        self.visualizer = TimeSeriesVisualizer(self.config)
        self.memory = memory or ExperimentMemory(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        visualizations: Dict[str, str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Run the analysis agent.

        Returns
        -------
        report_text : str
            Human-readable analysis report.
        constraints : dict
            ``{"forbidden_L1_actions": [...], "forbidden_L2_actions": [...],
              "forbidden_L3_models": [...]}``
        """
        logger.info("Running analysis agent (pre-flight check-up)...")
        # #region agent log
        try:
            import json
            import time
            from pathlib import Path
            _p = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
            with open(_p, "a") as _f:
                _f.write(json.dumps({"timestamp": int(time.time()*1000), "location": "analysis_agent.py:run", "message": "analysis_run_start", "data": {}, "hypothesisId": "H4"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Compute quick stats for the prompt (and for fallback rules)
        data_profile = self._profile_data(data)

        # Create analysis prompt
        prompt = self._create_analysis_prompt(data, data_profile, visualizations)

        # AMEM: inject relevant prior knowledge (if any)
        try:
            hits = self.memory.retrieve_semantic(
                query="time series analysis constraints preprocessing features model selection",
                top_k=int(self.config.get("amem", {}).get("top_k", 5)),
            )
            if hits:
                mem_block = "\n\n[AMEM — Relevant prior notes]\n" + "\n".join(
                    [f"- ({h['score']:.3f}) {h['text']}" for h in hits]
                )
                prompt = mem_block + "\n\n" + prompt
        except Exception:
            pass

        # Add retry mechanism for rate limiting
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # #region agent log
                try:
                    import json
                    import time
                    from pathlib import Path
                    _p = Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
                    with open(_p, "a") as _f:
                        _f.write(json.dumps({"timestamp": int(time.time()*1000), "location": "analysis_agent.py:llm_invoke", "message": "analysis_llm_invoke_start", "data": {"attempt": attempt+1}, "hypothesisId": "H5"}) + "\n")
                except Exception:
                    pass
                # #endregion
                vprint("ANALYSIS", "Calling LLM for analysis...")
                response = self.llm.invoke([
                    SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                report, constraints = self._parse_response(response.content, data_profile)
                logger.info("AnalysisAgent done. Constraints: %s", constraints)

                # AMEM: store this slice analysis for future slices
                try:
                    self.memory.store_semantic(
                        text=f"Analysis insights: {report}\nConstraints: {json.dumps(constraints, ensure_ascii=False)}",
                        meta={"agent": "AnalysisAgent"},
                    )
                except Exception:
                    pass
                return report, constraints

            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Rate limit hit, retrying in %ds... (attempt %d/%d)",
                            retry_delay, attempt + 1, max_retries,
                        )
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error("Rate limit exceeded after %d attempts", max_retries)
                        return self._generate_fallback_analysis(data, data_profile)
                else:
                    logger.error("Error in analysis agent: %s", e)
                    return self._generate_fallback_analysis(data, data_profile)

        return self._generate_fallback_analysis(data, data_profile)

    # ------------------------------------------------------------------
    # Data profiling (quick statistical checks for prompt + fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Compute quick statistical profile used by the prompt and fallback."""
        if "value" in data.columns:
            vals = data["value"].dropna().values
        else:
            vals = data.iloc[:, 0].dropna().values
        vals = np.asarray(vals, dtype=float)

        profile: Dict[str, Any] = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if len(vals) else 0.0,
            "std": float(np.std(vals)) if len(vals) else 0.0,
            "min": float(np.min(vals)) if len(vals) else 0.0,
            "max": float(np.max(vals)) if len(vals) else 0.0,
            "has_negatives": bool(np.any(vals < 0)) if len(vals) else False,
            "has_zeros": bool(np.any(vals == 0)) if len(vals) else False,
            "has_non_positive": bool(np.any(vals <= 0)) if len(vals) else False,
        }

        # Stationarity: ADF test (safe try)
        try:
            if len(vals) >= 20:
                adf_stat, adf_p, *_ = adfuller(vals, maxlag=min(20, len(vals) // 4))
                profile["adf_statistic"] = float(adf_stat)
                profile["adf_pvalue"] = float(adf_p)
                profile["is_stationary"] = adf_p < 0.05
            else:
                profile["is_stationary"] = None
        except Exception:
            profile["is_stationary"] = None

        # Trend slope
        if len(vals) > 1:
            slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
            profile["trend_slope"] = slope
        else:
            profile["trend_slope"] = 0.0

        return profile

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _create_analysis_prompt(
        self,
        data: pd.DataFrame,
        data_profile: Dict[str, Any],
        visualizations: Dict[str, str] = None,
    ) -> str:
        """Build the user prompt including data, profile, and action space reference."""
        # Truncate data for the prompt (send head + tail if too long)
        n = len(data)
        if n > 200:
            head = data.head(100).to_dict(orient="list")
            tail = data.tail(100).to_dict(orient="list")
            data_str = f"HEAD (first 100 rows): {head}\n... ({n - 200} rows omitted) ...\nTAIL (last 100 rows): {tail}"
        else:
            data_str = str(data.to_dict(orient="list"))

        viz_info = ""
        if visualizations:
            viz_info = f"\nGenerated Visualizations:\n{visualizations}\n"

        # Include action space reference so LLM knows valid option strings
        action_space_ref = """
ACTION SPACE REFERENCE (use these EXACT strings in forbidden lists):

L1_preprocess:
  - missing_value_strategy: ["interpolate", "forward_fill", "backward_fill", "mean", "median", "drop", "zero", "none"]
  - outlier_detect:         ["iqr", "zscore", "percentile", "none"]
  - outlier_handle:         ["clip", "drop", "interpolate", "ffill", "bfill", "mean", "median", "smooth", "none"]
  - normalization:         ["minmax", "zscore", "none"]
  - stationarity:          ["diff", "log", "none"]

L2_features:
  - periodic:      ["fourier", "none"]
  - lags:          [5, 10, 20, 50]
  - window_stats:  ["mean", "std", "min_max", "none"]

L3_models (paradigms):
  - "statistical"  -> ARIMA, ExponentialSmoothing, TBATS, Prophet, Theta, Croston, RandomWalk, MovingAverage
  - "regression"   -> LinearRegression, PolynomialRegression, RidgeRegression, LassoRegression, ElasticNet, SVR
  - "tree"         -> XGBoost, LightGBM, RandomForest, GradientBoosting
  - "deep"         -> LSTM, NeuralNetwork, Transformer
"""

        return f"""Analyze the following time series data and produce BOTH insights and mcts_constraints.

DATA PROFILE (pre-computed statistics):
{json.dumps(data_profile, indent=2, default=str)}

RAW DATA:
{data_str}
{viz_info}
{action_space_ref}

Based on the data profile and raw data above:
1. Provide a comprehensive analysis (trend, seasonality, stationarity, data quality, challenges).
2. Decide which MCTS actions are UNSAFE or ILLOGICAL for this dataset.
3. Return your answer as a single JSON object with "insights" and "mcts_constraints" keys.
"""

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw_response: str,
        data_profile: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract (report_text, constraints_dict) from LLM response.

        Tries multiple strategies:
        1. Direct JSON parse of the full response.
        2. Extract JSON block from markdown fences.
        3. Regex extraction of the JSON object.
        4. Fallback: return raw text as report with rule-based constraints.
        """
        # Strategy 1: direct parse
        try:
            parsed = json.loads(raw_response)
            return self._unpack_parsed(parsed, data_profile)
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: extract from markdown fences ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_response, re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                return self._unpack_parsed(parsed, data_profile)
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: find first { ... } block
        brace_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                return self._unpack_parsed(parsed, data_profile)
            except (json.JSONDecodeError, TypeError):
                pass

        # Strategy 4: fallback — treat entire response as report, use rule-based constraints
        logger.warning("Could not parse JSON from LLM response; using rule-based constraints.")
        constraints = self._rule_based_constraints(data_profile)
        return raw_response, constraints

    @staticmethod
    def _unpack_parsed(
        parsed: Dict[str, Any],
        data_profile: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Unpack a parsed JSON dict into (report, constraints)."""
        insights = parsed.get("insights", "")
        raw_constraints = parsed.get("mcts_constraints", {})

        # Normalize constraints
        constraints = {
            "forbidden_L1_actions": list(raw_constraints.get("forbidden_L1_actions", [])),
            "forbidden_L2_actions": list(raw_constraints.get("forbidden_L2_actions", [])),
            "forbidden_L3_models": list(raw_constraints.get("forbidden_L3_models", [])),
        }

        # Safety net: always enforce hard rules that the LLM might miss
        if data_profile.get("has_non_positive") and "log" not in constraints["forbidden_L1_actions"]:
            constraints["forbidden_L1_actions"].append("log")

        return str(insights), constraints

    # ------------------------------------------------------------------
    # Rule-based fallback constraints (no LLM needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_constraints(data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate constraints purely from statistical profile (fallback)."""
        constraints: Dict[str, Any] = {
            "forbidden_L1_actions": [],
            "forbidden_L2_actions": [],
            "forbidden_L3_models": [],
        }

        # Hard rule: log transform crashes on non-positive data
        if data_profile.get("has_non_positive"):
            constraints["forbidden_L1_actions"].append("log")

        # Soft rule: deep models need enough data
        n = data_profile.get("n", 0)
        if n < 50:
            constraints["forbidden_L3_models"].append("deep")

        return constraints

    # ------------------------------------------------------------------
    # Fallback analysis (when LLM is unreachable)
    # ------------------------------------------------------------------

    def _generate_fallback_analysis(
        self,
        data: pd.DataFrame,
        data_profile: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate fallback analysis + rule-based constraints when LLM fails."""
        logger.info("Generating fallback analysis (rule-based)...")

        trend = "stable"
        slope = data_profile.get("trend_slope", 0.0)
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"

        report = f"""
# Time Series Analysis Report (Fallback)

## Data Overview
- **Dataset Size:** {data_profile['n']} observations
- **Basic Statistics:**
  - Mean: {data_profile['mean']:.4f}
  - Std:  {data_profile['std']:.4f}
  - Min:  {data_profile['min']:.4f}
  - Max:  {data_profile['max']:.4f}
- **Contains non-positive values:** {data_profile.get('has_non_positive', False)}
- **ADF p-value:** {data_profile.get('adf_pvalue', 'N/A')}

## Trend Analysis
The time series data shows a **{trend}** trend (slope={slope:.6f}).

## Data Characteristics
- **Stationarity:** {'Stationary (ADF p<0.05)' if data_profile.get('is_stationary') else 'Possibly non-stationary'}
- **Data Quality:** Data appears suitable for forecasting.

## Recommendations
1. Consider models robust to non-stationarity if ADF test fails.
2. Avoid log transforms if data contains zeros or negatives.
3. Use appropriate preprocessing techniques for trend removal.

*Note: This is a fallback analysis generated due to LLM unavailability.*
"""
        constraints = self._rule_based_constraints(data_profile)
        return report, constraints
