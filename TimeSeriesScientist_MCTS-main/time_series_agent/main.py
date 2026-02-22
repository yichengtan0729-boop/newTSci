#!/usr/bin/env python3
"""
Time Series Prediction Agent - Industrial-grade Entry Point
This script orchestrates the full time series agent workflow, similar to tradingagents' main entry.

Two pipelines:
- use_funnel_pipeline=True: MCTS (L1-L3) -> Candidate Pool -> TuningAgent -> EnsembleAgent (funnel).
- use_funnel_pipeline=False: Original LangGraph preprocess -> analyze -> validate -> forecast -> report.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*renamed to.*ensure_all_finite.*",
    category=FutureWarning,
)
# Load .env from project root — API keys (GOOGLE_API_KEY etc.) 单独存放保护
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
    load_dotenv()  # 也检查 cwd
except ImportError:
    pass

from graph.funnel_pipeline import run_funnel
from config.default_config import DEFAULT_CONFIG
from utils.progress import set_verbose

if __name__ == "__main__":
    print("=" * 60)
    print("TimeSeriesSciensist")
    print("=" * 60)

    # 1. Create and customize config
    config = DEFAULT_CONFIG.copy()
    # Example customizations (edit as needed):
    config["num_slices"] = 25
    config["input_length"] = 512
    config["horizon"] = 96
    config["data_path"] = "dataset/ETTh1.csv"
    config["debug"] = False
    config["verbose"] = True   # 打开详细进度打印（设 False 关闭）
    config["date_column"] = "date"
    config["value_column"] = "OT"
    # Funnel pipeline: MCTS (L1-L3) -> Candidate Pool -> Tuning -> Ensemble
    config["use_funnel_pipeline"] = True
    config["funnel_num_slices"] = 1  # 试跑先用 1 个 slice；可改为 None 跑全部

    # 启用 verbose 进度打印
    set_verbose(config.get("verbose", False))

    # LLM: use OpenAI-compatible API
    config["llm_provider"] = "openai"
    config["llm_model"] = "gpt-4o"
    config["llm_api_base"] = "https://api.chatanywhere.org/v1"

    # MCTS: single-tree rollouts (L1+L2+L3 jointly)
    config["mcts_rollouts"] = 50           # 总 rollout 数，UCT 自动分配
    config["candidate_pool_size"] = 5      # Top-K diverse candidates for Tuning

    # LangSmith tracing — auto-enabled via .env (LANGCHAIN_TRACING_V2=true)
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
        _ls_key = os.environ.get("LANGCHAIN_API_KEY", "")
        if _ls_key and not _ls_key.startswith("your_"):
            print(f"LangSmith tracing: ENABLED (project={os.environ.get('LANGCHAIN_PROJECT', 'default')})")
        else:
            print("LangSmith tracing: LANGCHAIN_API_KEY not set, tracing disabled")
            os.environ.pop("LANGCHAIN_TRACING_V2", None)  # disable to avoid errors

    # 2. Check for API key in environment (per provider)
    provider = config.get("llm_provider", "openai")
    key_vars = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    key_name = key_vars.get(provider, "OPENAI_API_KEY")
    if not os.environ.get(key_name):
        print(f"Error: Set {key_name} environment variable before running (provider={provider}).")
        sys.exit(1)

    # 2b. When using funnel: validate and unify data_path (must exist, resolve to absolute)
    if config.get("use_funnel_pipeline"):
        data_path = config.get("data_path")
        if not data_path or not str(data_path).strip():
            print("Error: use_funnel_pipeline=True requires config['data_path'] to be set.")
            sys.exit(1)
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"Error: data_path does not exist: {data_path}")
            sys.exit(1)
        config["data_path"] = str(data_path.resolve())

    start_time = time.time()
    if config.get("use_funnel_pipeline"):
        # --- Funnel: per-slice single-tree MCTS (L1+L2+L3) -> Tuning -> Ensemble -> Report ---
        print("Running Funnel pipeline (single-tree MCTS: L1+L2+L3 → Analysis(cached) → Tuning → Ensemble)...")
        print(f"Configuration: horizon={config['horizon']}, data_path={config['data_path']}")
        print("=" * 60)
        results = run_funnel(
            config,
            mcts_rollouts=config.get("mcts_rollouts", 30),
            candidate_pool_size=config.get("candidate_pool_size", 5),
            ensemble_method=config.get("ensemble_method", "greedy"),
            use_llm_policies=config.get("use_llm_policies", True),
        )
        if results.get("error"):
            print(f"Funnel error: {results['error']}")
        else:
            print(f"Slices processed: {results.get('num_slices', 0)}")
            agg = results.get("aggregated", {})
            agg_metrics = agg.get("test_metrics", {})
            if agg_metrics.get("ensemble"):
                em = agg_metrics["ensemble"]
                print(f"Aggregated ensemble — MSE: {em['mse']:.4f}, MAE: {em['mae']:.4f}, MAPE: {em['mape']:.2f}%")
    else:
        # --- Original LangGraph workflow ---
        from graph.agent_graph import TimeSeriesAgentGraph
        print("Initializing Time Series Agent Graph...")
        graph = TimeSeriesAgentGraph(config=config, model=config["llm_model"], debug=config["debug"])
        print("Running the time series agent workflow...")
        print(f"Configuration: {config['num_slices']} slices, {config['horizon']} horizon steps")
        print("=" * 60)
        delay_between_slices = 5
        results = graph.run()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    if not config.get("use_funnel_pipeline") and config.get("num_slices"):
        print(f"Average time per slice: {execution_time/config['num_slices']:.2f} seconds")
    if not config.get("use_funnel_pipeline"):
        print("Adding final delay to ensure API rate limit compliance...")
        time.sleep(5)

    # 5. Save results to file
    print("\n" + "=" * 60)
    print("Workflow execution completed!")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete results (including all slice results)
    complete_report_filename = f"complete_time_series_report_{timestamp}.json"
    complete_report_path = results_dir / complete_report_filename
    
    try:
        # Strip non-JSON-serializable mcts_tree from slice_results before saving
        results_to_save = dict(results)
        if "slice_results" in results_to_save:
            results_to_save["slice_results"] = [
                {k: v for k, v in sr.items() if k != "mcts_tree"}
                for sr in results["slice_results"]
            ]
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False, default=str)
        print(f"Complete results saved to: {complete_report_path}")
    except Exception as e:
        print(f"Error saving complete results: {e}")

    if config.get("use_funnel_pipeline") and not results.get("error"):
        # Funnel summary — per-slice + aggregated
        print("\n" + "=" * 60)
        print("FUNNEL PIPELINE RESULTS")
        print("=" * 60)
        slice_results = results.get("slice_results", [])
        print(f"Slices processed: {results.get('num_slices', len(slice_results))}")
        for sr in slice_results:
            sid = sr.get("slice_id", "?")
            if sr.get("error"):
                print(f"  Slice {sid}: ERROR — {sr['error']}")
            else:
                ens = sr.get("test_metrics", {}).get("ensemble", {})
                print(
                    f"  Slice {sid}: L1={sr.get('best_l1_params', {})}, "
                    f"ensemble MSE={ens.get('mse', float('nan')):.4f}, "
                    f"MAE={ens.get('mae', float('nan')):.4f}, "
                    f"MAPE={ens.get('mape', float('nan')):.2f}%"
                )

        agg = results.get("aggregated", {})
        agg_metrics = agg.get("test_metrics", {})
        if agg_metrics.get("ensemble"):
            em = agg_metrics["ensemble"]
            print(f"\nAggregated (mean over slices):")
            print(f"  Ensemble — MSE: {em['mse']:.4f}, MAE: {em['mae']:.4f}, MAPE: {em['mape']:.2f}%")
        for model_name, metrics in agg_metrics.items():
            if model_name != "ensemble":
                print(f"  {model_name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")

        # Save funnel JSON
        funnel_report_filename = f"funnel_results_{timestamp}.json"
        funnel_report_path = results_dir / funnel_report_filename
        funnel_export = {
            "timestamp": timestamp,
            "num_slices": results.get("num_slices"),
            "aggregated": agg,
            "slice_results": [
                {
                    "slice_id": sr.get("slice_id"),
                    "best_l1_params": sr.get("best_l1_params"),
                    "best_reward": sr.get("best_reward"),
                    "ensemble_info": sr.get("ensemble_info"),
                    "test_metrics": sr.get("test_metrics"),
                    "y_true": sr.get("y_true"),
                    "ensemble_pred": sr.get("ensemble_pred"),
                }
                for sr in slice_results
            ],
        }
        try:
            with open(funnel_report_path, "w", encoding="utf-8") as f:
                json.dump(funnel_export, f, indent=2, ensure_ascii=False, default=str)
            print(f"Funnel report saved to: {funnel_report_path}")
        except Exception as e:
            print(f"Error saving funnel report: {e}")

        # Save narrative report (markdown)
        if results.get("report"):
            funnel_report_md = results_dir / f"funnel_report_{timestamp}.md"
            try:
                with open(funnel_report_md, "w", encoding="utf-8") as f:
                    f.write(results["report"])
                print(f"Funnel narrative report saved to: {funnel_report_md}")
            except Exception as e:
                print(f"Error saving funnel narrative report: {e}")

    # Save aggregated results separately (final averaged predictions) — original graph only
    if results.get("aggregated_results"):
        aggregated_report_filename = f"aggregated_forecast_results_{timestamp}.json"
        aggregated_report_path = results_dir / aggregated_report_filename
        
        aggregated_summary = {
            "timestamp": timestamp,
            "aggregation_info": results["aggregated_results"]["aggregation_info"],
            "final_individual_predictions": results["aggregated_results"]["individual_predictions"],
            "final_ensemble_predictions": results["aggregated_results"]["ensemble_predictions"],
            "final_test_metrics": results["aggregated_results"]["test_metrics"],
            "final_forecast_metrics": results["aggregated_results"]["forecast_metrics"]
        }
        
        try:
            with open(aggregated_report_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"Aggregated forecast results saved to: {aggregated_report_path}")
        except Exception as e:
            print(f"Error saving aggregated results: {e}")
    
    # Print summary of aggregated results
    if results.get("aggregated_results"):
        print("\n" + "=" * 60)
        print("FINAL AGGREGATED FORECAST RESULTS")
        print("=" * 60)
        
        agg_info = results["aggregated_results"]["aggregation_info"]
        print(f"Number of slices processed: {agg_info['num_slices']}")
        print(f"Aggregation method: {agg_info['aggregation_method']}")
        
        # Print final ensemble metrics
        if results["aggregated_results"]["test_metrics"].get("ensemble"):
            ensemble_metrics = results["aggregated_results"]["test_metrics"]["ensemble"]
            print(f"\nFinal Ensemble Performance:")
            print(f"  MSE: {ensemble_metrics['mse']:.4f}")
            print(f"  MAE: {ensemble_metrics['mae']:.4f}")
            print(f"  MAPE: {ensemble_metrics['mape']:.2f}%")
        
        # Print individual model metrics
        print(f"\nIndividual Model Performance (averaged across slices):")
        for model_name, metrics in results["aggregated_results"]["test_metrics"].items():
            if model_name != "ensemble":
                print(f"  {model_name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")

    print("\nExperiment completed!") 
