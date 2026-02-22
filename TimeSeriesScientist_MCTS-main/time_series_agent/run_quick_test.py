#!/usr/bin/env python3
"""
Quick test: minimal rollouts (e.g. 8), 1 slice, to verify pipeline runs and MCTS tree plots.
Run from project root: python -m time_series_agent.run_quick_test
Or from time_series_agent: python run_quick_test.py
"""
import os
import sys
from pathlib import Path

# Project root = parent of time_series_agent
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    load_dotenv()
except ImportError:
    pass

from graph.funnel_pipeline import run_funnel
from config.default_config import DEFAULT_CONFIG
from utils.progress import set_verbose


def main():
    # Data path: from repo root
    data_path = _ROOT / "dataset" / "ETTh1.csv"
    if not data_path.exists():
        data_path = _ROOT / ".." / "dataset" / "ETTh1.csv"
    data_path = data_path.resolve()
    if not data_path.exists():
        print(f"Error: data not found at {data_path}")
        sys.exit(1)

    config = DEFAULT_CONFIG.copy()
    config["data_path"] = str(data_path)
    config["output_dir"] = str(_ROOT / "results")
    config["verbose"] = True
    config["funnel_plot_mcts_trees"] = True
    config["funnel_generate_report"] = False  # skip report to save time
    config["num_slices"] = 10
    config["input_length"] = 512
    config["horizon"] = 96
    config["funnel_num_slices"] = 1
    config["mcts_rollouts"] = 3  # minimal for quick LLM test
    config["candidate_pool_size"] = 2
    config["use_funnel_pipeline"] = True
    config["use_llm_policies"] = True  # 打开 LLM 模式测试 expand/rollout 策略
    set_verbose(True)

    print("Quick test: LLM mode ON, 3 rollouts, 1 slice, tree plots enabled")
    print(f"data_path={config['data_path']}")
    print("=" * 60)

    results = run_funnel(
        config,
        mcts_rollouts=config["mcts_rollouts"],
        candidate_pool_size=config["candidate_pool_size"],
        ensemble_method="greedy",
        use_llm_policies=config.get("use_llm_policies", True),
    )

    if results.get("error"):
        print("Funnel error:", results["error"])
        sys.exit(1)

    print("=" * 60)
    print("Quick test completed.")
    print("Slices:", results.get("num_slices"))
    agg = results.get("aggregated", {})
    if agg.get("test_metrics", {}).get("ensemble"):
        em = agg["test_metrics"]["ensemble"]
        print(f"Aggregated ensemble MSE={em['mse']:.4f}, MAE={em['mae']:.4f}")
    tree_plots = results.get("mcts_tree_plots", [])
    print("MCTS tree plots:", tree_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
