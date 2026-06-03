"""
Recommendation Engine
Smart recommendations for optimal ML cloud compute configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from src.calculator import CostCalculator, ComparisonResult, WorkloadSpec, WorkloadType
from src.performance import PerformanceAnalyzer
from src.analyzer import WorkloadAnalyzer, TaskType


@dataclass
class Recommendation:
    """A single recommendation."""
    rank: int
    provider: str
    instance_type: str
    gpu_type: str
    gpu_count: int
    hourly_cost: float
    estimated_total: float
    performance_score: float
    cost_efficiency: float
    reasoning: list
    pros: list = field(default_factory=list)
    cons: list = field(default_factory=list)


@dataclass
class RecommendationReport:
    """Complete recommendation report."""
    workload_description: str
    recommendations: list
    summary: str = ""

    def __str__(self):
        lines = [
            "=" * 65,
            "  ML CLOUD COMPUTE RECOMMENDATIONS",
            "=" * 65,
            f"  Workload: {self.workload_description}",
            "=" * 65,
        ]
        for rec in self.recommendations:
            lines.append(f"\n  #{rec.rank} — {rec.provider.upper()}")
            lines.append(f"  Instance:     {rec.instance_type}")
            lines.append(f"  GPU:          {rec.gpu_type} x{rec.gpu_count}")
            lines.append(f"  Hourly:       ${rec.hourly_cost:.2f}")
            lines.append(f"  Est. Total:   ${rec.estimated_total:.2f}")
            lines.append(f"  Perf Score:   {rec.performance_score:.2f}")
            lines.append(f"  Cost Eff:     {rec.cost_efficiency:.2f} TFLOPS/$")
            if rec.reasoning:
                lines.append(f"  Why:          {'; '.join(rec.reasoning)}")
            if rec.pros:
                lines.append(f"  Pros:         {', '.join(rec.pros)}")
            if rec.cons:
                lines.append(f"  Cons:         {', '.join(rec.cons)}")
        if self.summary:
            lines.append(f"\n{'=' * 65}")
            lines.append(f"  SUMMARY: {self.summary}")
        lines.append("=" * 65)
        return "\n".join(lines)


class Recommender:
    """Smart recommendation engine for ML cloud compute."""

    def __init__(self, provider_registry):
        self.registry = provider_registry
        self.calculator = CostCalculator(provider_registry)
        self.performance = PerformanceAnalyzer(provider_registry)
        self.analyzer = WorkloadAnalyzer()

    def recommend(
        self,
        task_type: str = "inference",
        params_b: float = 7.0,
        budget_monthly: Optional[float] = None,
        duration_hours: Optional[float] = None,
        providers: Optional[list] = None,
        prioritize: str = "balanced",
    ) -> RecommendationReport:
        """Generate recommendations based on workload requirements."""
        task = TaskType(task_type) if task_type in [t.value for t in TaskType] else TaskType.INFERENCE
        profile = self.analyzer.analyze(task, params_b, duration_override=duration_hours)

        recommendations = []
        provider_list = providers or self.registry.list_providers()

        for provider_name in provider_list:
            provider = self.registry.get(provider_name)
            if provider is None:
                continue

            perf_results = self.performance.analyze_provider(provider_name)
            for perf in perf_results:
                # Check if this GPU matches our recommendation
                if profile.recommended_gpu not in perf.gpu_type and perf.gpu_type not in profile.recommended_gpu:
                    continue

                estimated_total = perf.hourly_cost * profile.estimated_duration_hours
                if budget_monthly and estimated_total > budget_monthly:
                    continue

                reasoning = []
                pros = []
                cons = []

                if perf.composite_score > 0.7:
                    reasoning.append("Excellent performance-per-dollar")
                    pros.append("High compute efficiency")
                elif perf.composite_score > 0.4:
                    reasoning.append("Good balance of performance and cost")
                else:
                    reasoning.append("Lower performance-per-dollar ratio")
                    cons.append("May not be cost-optimal")

                if provider_name in ("digitalocean", "akamai"):
                    reasoning.append("Competitive pricing for smaller providers")
                    pros.append("Simpler pricing model")
                elif provider_name == "gcp":
                    reasoning.append("Strong spot/preemptible discounts")
                    pros.append("Up to 70% spot savings")
                elif provider_name == "aws":
                    reasoning.append("Broadest GPU selection")
                    pros.append("Most instance types available")
                elif provider_name == "azure":
                    reasoning.append("Enterprise integration")
                    pros.append("Good for hybrid cloud")

                if perf.hourly_cost > 50:
                    cons.append("High hourly cost")

                recommendations.append(Recommendation(
                    rank=0,
                    provider=provider_name,
                    instance_type=perf.instance_type,
                    gpu_type=perf.gpu_type,
                    gpu_count=perf.gpu_count if hasattr(perf, 'gpu_count') else 1,
                    hourly_cost=perf.hourly_cost,
                    estimated_total=estimated_total,
                    performance_score=perf.composite_score,
                    cost_efficiency=perf.tflops_per_dollar,
                    reasoning=reasoning,
                    pros=pros,
                    cons=cons,
                ))

        # Sort by priority
        if prioritize == "cost":
            recommendations.sort(key=lambda r: r.hourly_cost)
        elif prioritize == "performance":
            recommendations.sort(key=lambda r: r.performance_score, reverse=True)
        elif prioritize == "availability":
            recommendations.sort(key=lambda r: (r.provider not in ["digitalocean", "akamai"], -r.performance_score))
        else:
            recommendations.sort(key=lambda r: (r.performance_score / max(r.hourly_cost, 0.01)), reverse=True)

        # Assign ranks
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1

        recs = recommendations[:5]

        summary = ""
        if recs:
            best = recs[0]
            summary = (
                f"Best option: {best.provider.upper()} {best.instance_type} "
                f"at ${best.hourly_cost:.2f}/hr — {best.reasoning[0] if best.reasoning else 'Optimal choice'}"
            )

        return RecommendationReport(
            workload_description=f"{task_type} ({params_b}B params, {profile.recommended_gpu} x{profile.recommended_gpu_count})",
            recommendations=recs,
            summary=summary,
        )

    def quick_recommend(self, gpu_type: str, task: str = "inference") -> RecommendationReport:
        """Quick recommendation for a specific GPU type."""
        task_map = {
            "training": (TaskType.PRETRAINING, 13.0),
            "inference": (TaskType.INFERENCE, 7.0),
            "finetuning": (TaskType.FINE_TUNING, 7.0),
        }
        task_type, default_params = task_map.get(task, (TaskType.INFERENCE, 7.0))
        return self.recommend(
            task_type=task_type.value,
            params_b=default_params,
            prioritize="balanced",
        )
