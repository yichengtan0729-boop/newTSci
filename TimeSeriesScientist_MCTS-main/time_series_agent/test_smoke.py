#!/usr/bin/env python3
"""
Smoke test — runs the funnel pipeline on 1 slice with minimal MCTS rollouts.
Expected runtime: ~3-5 minutes (depends on LLM API speed).

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_smoke.py
"""

import os
import sys
import time
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

from graph.funnel_pipeline import run_funnel
from config.default_config import DEFAULT_CONFIG
from utils.progress import set_verbose

if __name__ == "__main__":
    # Support both OpenAI and Google (project default)
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY or GOOGLE_API_KEY first.")
        print('  export GOOGLE_API_KEY="..."  # or OPENAI_API_KEY for OpenAI')
        sys.exit(1)
    if os.environ.get("GOOGLE_API_KEY"):
        config_llm = {"llm_provider": "google", "llm_model": "gemini-2.5-flash"}
    else:
        config_llm = {"llm_provider": "openai", "llm_model": "gpt-4o"}

    config = DEFAULT_CONFIG.copy()
    config.update(config_llm)
    config["data_path"] = str(Path(__file__).resolve().parent.parent / "dataset" / "ETTh1.csv")
    config["date_column"] = "date"
    config["value_column"] = "OT"
    config["horizon"] = 96
    config["input_length"] = 512
    config["num_slices"] = 1           # only 1 slice for smoke test
    config["funnel_num_slices"] = 1
    config["mcts_rollouts"] = 5        # very few rollouts (fast, single-tree L1+L2+L3)
    config["candidate_pool_size"] = 2  # only 2 candidates from MCTS
    config["fast_simulation_max_epochs"] = 3  # cap deep model epochs
    config["use_llm_policies"] = False  # random policies (no LLM in MCTS) — faster
    config["funnel_generate_report"] = False  # skip report for speed
    config["verbose"] = True  # 打开进度打印
    set_verbose(True)

    print("=" * 60)
    print("SMOKE TEST — 1 slice, 5 rollouts (single-tree L1+L2+L3), random policies")
    print(f"Data: {config['data_path']}")
    print("=" * 60)

    start = time.time()
    results = run_funnel(
        config,
        mcts_rollouts=config["mcts_rollouts"],
        candidate_pool_size=config["candidate_pool_size"],
        ensemble_method="greedy",
        use_llm_policies=config["use_llm_policies"],
    )
    elapsed = time.time() - start

    print(f"\nElapsed: {elapsed:.1f}s")

    if results.get("error"):
        print(f"ERROR: {results['error']}")
        sys.exit(1)

    sr = results["slice_results"][0]
    if sr.get("error"):
        print(f"Slice error: {sr['error']}")
        sys.exit(1)

    print(f"\nSlice 0 results:")
    print(f"  Best L1 params: {sr.get('best_l1_params')}")
    metrics = sr.get("test_metrics", {})
    ens = metrics.get("ensemble", {})
    print(f"  Ensemble MSE: {ens.get('mse', 'N/A'):.4f}")
    print(f"  Ensemble MAE: {ens.get('mae', 'N/A'):.4f}")
    print(f"  Ensemble MAPE: {ens.get('mape', 'N/A'):.2f}%")

    for k, v in metrics.items():
        if k != "ensemble":
            print(f"  {k}: MSE={v['mse']:.4f}, MAE={v['mae']:.4f}, MAPE={v['mape']:.2f}%")

    print(f"\nPredictions dict keys: {list(sr.get('predictions_dict', {}).keys())}")
    print(f"y_true length: {len(sr.get('y_true', []))}")
    print(f"ensemble_pred length: {len(sr.get('ensemble_pred', []))}")

    agg = results.get("aggregated", {})
    agg_ens = agg.get("test_metrics", {}).get("ensemble", {})
    if agg_ens:
        print(f"\nAggregated ensemble: MSE={agg_ens['mse']:.4f}, MAE={agg_ens['mae']:.4f}")

    print("\n=== SMOKE TEST PASSED ===")
