#!/usr/bin/env python3
"""
Cost Audit Tool
Analyze and optimize existing cloud ML compute spending.
"""

import argparse
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calculator import CostCalculator
from src.providers import ProviderRegistry
from src.recommender import Recommender


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {path} not found, using defaults.")
        return {}


def audit_current_spend(config: dict):
    """Audit current cloud spending and suggest optimizations."""
    registry = ProviderRegistry()
    recommender = Recommender(registry)

    print("=" * 70)
    print("  💰 ML CLOUD COMPUTE COST AUDIT")
    print("=" * 70)

    workload_cfg = config.get("workload", {})
    gpu_type = workload_cfg.get("gpu_type", "h100")
    gpu_count = workload_cfg.get("gpu_count", 8)
    hours = workload_cfg.get("duration_hours", 720)  # 1 month
    workload_type = workload_cfg.get("type", "training")

    print(f"\n  Current Config:")
    print(f"  GPU:          {gpu_type} x{gpu_count}")
    print(f"  Workload:     {workload_type}")
    print(f"  Monthly Hours: {hours}")

    calculator = CostCalculator(registry)
    result = calculator.estimate_training_cost(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        training_hours=hours,
    )

    print(f"\n  Monthly Cost Estimates:")
    print(f"  {'─' * 55}")
    for bd in sorted(result.breakdowns, key=lambda b: b.total_cost):
        print(f"  {bd.provider:15s} ${bd.total_cost:>10.2f}/mo  ({bd.instance_type})")

    if result.breakdowns:
        cheapest = result.cheapest_provider()
        most_expensive = max(result.breakdowns, key=lambda b: b.total_cost)
        if cheapest and most_expensive:
            savings = most_expensive.total_cost - cheapest.total_cost
            savings_pct = (savings / most_expensive.total_cost * 100) if most_expensive.total_cost > 0 else 0
            print(f"\n  💡 Potential Savings: ${savings:.2f}/mo ({savings_pct:.0f}%)")
            print(f"     Switch to {cheapest.provider.upper()} {cheapest.instance_type}")

    opt_cfg = config.get("optimization", {})
    max_monthly = opt_cfg.get("max_monthly_budget", 50000)
    if result.breakdowns:
        current = max(result.breakdowns, key=lambda b: b.total_cost)
        if current.total_cost > max_monthly:
            print(f"\n  ⚠️  WARNING: Estimated cost ${current.total_cost:.2f} exceeds budget ${max_monthly:.2f}")

    print(f"\n  Recommendations:")
    rec_report = recommender.recommend(
        task_type=workload_type,
        params_b=config.get("analysis", {}).get("model_params_b", 13.0),
        prioritize=config.get("analysis", {}).get("prioritize", "balanced"),
    )
    for rec in rec_report.recommendations[:3]:
        print(f"    #{rec.rank} {rec.provider.upper()} {rec.instance_type}")
        print(f"       ${rec.hourly_cost:.2f}/hr — {rec.reasoning[0] if rec.reasoning else ''}")


def main():
    parser = argparse.ArgumentParser(description="ML Cloud Compute Cost Audit")
    parser.add_argument("--config", default="configs/default.yml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    audit_current_spend(config)
    print()


if __name__ == "__main__":
    main()
