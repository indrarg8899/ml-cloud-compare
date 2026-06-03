#!/usr/bin/env python3
"""
CLI Comparison Tool
Quick GPU/CPU cost comparisons from the command line.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calculator import CostCalculator
from src.providers import ProviderRegistry
from src.performance import PerformanceAnalyzer
from src.recommender import Recommender


def main():
    parser = argparse.ArgumentParser(
        description="ML Cloud Compute Cost Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare.py --gpu h100 --providers aws,gcp,azure
  python compare.py --gpu a100 --workload training --hours 48
  python compare.py --gpu l4 --workload inference --days 30
  python compare.py --gpu h100 --recommend --params 70
  python compare.py --gpu t4 --spot --storage 500
        """,
    )
    parser.add_argument("--gpu", default="h100", help="GPU type (h100, a100, l4, t4, a10)")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs (default: 8)")
    parser.add_argument("--providers", default=None, help="Comma-separated provider list")
    parser.add_argument("--workload", default="training", choices=["training", "inference", "finetuning", "batch"])
    parser.add_argument("--hours", type=float, default=None, help="Duration in hours")
    parser.add_argument("--days", type=float, default=None, help="Duration in days (converted to hours)")
    parser.add_argument("--spot", action="store_true", help="Use spot/preemptible pricing")
    parser.add_argument("--storage", type=float, default=100, help="Storage in GB (default: 100)")
    parser.add_argument("--params", type=float, default=13.0, help="Model params in billions (for recommendations)")
    parser.add_argument("--recommend", action="store_true", help="Show recommendations")
    parser.add_argument("--ranking", action="store_true", help="Show overall ranking")
    parser.add_argument("--output", default=None, help="Output file for HTML report")

    args = parser.parse_args()

    registry = ProviderRegistry()
    providers = args.providers.split(",") if args.providers else None
    duration = args.days * 24 if args.days else args.hours

    if args.recommend:
        recommender = Recommender(registry)
        report = recommender.recommend(
            task_type=args.workload,
            params_b=args.params,
            providers=providers,
        )
        print(report)

    elif args.ranking:
        perf = PerformanceAnalyzer(registry)
        ranking = perf.ranking(top_n=10)
        print("\n" + "=" * 75)
        print("  OVERALL RANKING: Performance per Dollar")
        print("=" * 75)
        for i, r in enumerate(ranking):
            print(f"  #{i+1:2d} {r.summary()}")

    elif args.output:
        from src.report import ReportGenerator
        gen = ReportGenerator()
        output = gen.generate(
            gpu_type=args.gpu,
            workload=args.workload,
            output=args.output,
            params_b=args.params,
            providers=providers,
        )
        print(f"✅ Report saved to {output}")

    else:
        calculator = CostCalculator(registry)

        if args.workload == "training":
            result = calculator.estimate_training_cost(
                gpu_type=args.gpu,
                gpu_count=args.gpus,
                training_hours=duration or 48,
                providers=providers,
                spot=args.spot,
                storage_gb=args.storage,
            )
        else:
            days = args.days or 30
            hours_per_day = args.hours or 24 if not args.days else 24
            result = calculator.estimate_inference_cost(
                gpu_type=args.gpu,
                gpu_count=args.gpus,
                hours_per_day=hours_per_day,
                days_per_month=days,
                providers=providers,
                spot=args.spot,
            )

        print(result.summary())

    print()


if __name__ == "__main__":
    main()
