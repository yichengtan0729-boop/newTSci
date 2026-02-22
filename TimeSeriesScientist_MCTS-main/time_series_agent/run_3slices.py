#!/usr/bin/env python3
"""
Run funnel pipeline with 3 slices, random MCTS, 150 rollouts, MPS device.

Usage:
    export GOOGLE_API_KEY="..."
    export PYTORCH_USE_MPS=1  # default: use MPS if available
    python run_3slices.py
"""

import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

# Enable MPS for PyTorch (Apple Silicon GPU)
os.environ["PYTORCH_USE_MPS"] = os.environ.get("PYTORCH_USE_MPS", "1")

from graph.funnel_pipeline import run_funnel
from config.default_config import DEFAULT_CONFIG
from utils.progress import set_verbose

if __name__ == "__main__":
    if os.environ.get("GOOGLE_API_KEY"):
        config_llm = {"llm_provider": "google", "llm_model": "gemini-2.5-flash"}
    elif os.environ.get("OPENAI_API_KEY"):
        config_llm = {"llm_provider": "openai", "llm_model": "gpt-4o"}
    else:
        print("ERROR: Set GOOGLE_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    config = DEFAULT_CONFIG.copy()
    config.update(config_llm)
    config["data_path"] = str(Path(__file__).resolve().parent.parent / "dataset" / "ETTh1.csv")
    config["date_column"] = "date"
    config["value_column"] = "OT"
    config["horizon"] = 96
    config["input_length"] = 512
    config["num_slices"] = 3
    config["funnel_num_slices"] = 3

    # Random MCTS (no LLM policies)
    config["use_llm_policies"] = False

    # MCTS
    config["mcts_rollouts"] = 150
    config["candidate_pool_size"] = 5
    config["fast_simulation_max_epochs"] = 5

    # ReAct tuning: max 6 steps
    config["tuning_max_trials"] = 6
    config["tuning_max_epochs_per_trial"] = 10

    # MPS device (model_library uses PYTORCH_USE_MPS env)
    config["verbose"] = True
    set_verbose(True)

    print("=" * 60)
    print("Run 3 slices | random MCTS | 150 rollouts | candidate 5")
    print("fast_sim_epoch 5 | ReAct max 6 | MPS device")
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

    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if results.get("error"):
        print(f"ERROR: {results['error']}")
        sys.exit(1)

    agg = results.get("aggregated", {})
    agg_ens = agg.get("test_metrics", {}).get("ensemble", {})
    if agg_ens:
        print(f"Aggregated ensemble: MSE={agg_ens['mse']:.4f}, MAE={agg_ens['mae']:.4f}, MAPE={agg_ens['mape']:.2f}%")
    print("\nDone.")
