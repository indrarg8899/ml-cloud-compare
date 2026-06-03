"""
Cost comparison engine for cloud GPU providers.

Aggregates pricing data from AWS, Azure, GCP, Lambda, CoreWeave
and computes cost metrics for ML workloads.
"""

from dataclasses import dataclass
from typing import Optional

from .data import GPU_CATALOG, PROVIDER_PRICING


@dataclass
class GPUSpec:
    """GPU specification."""
    name: str
    manufacturer: str
    vram_gb: float
    tflops_fp16: float
    tflops_fp32: float
    memory_bandwidth_gbps: float
    tdp_watts: int


@dataclass
class CostResult:
    """Cost comparison result."""
    gpu: str
    provider: str
    usage_type: str
    hourly_rate: float
    usage_hours: float
    total_cost: float
    cost_per_tflop: float
    cost_per_gb_vram: float
    cost_per_hour_compute: float
    gpu_specs: GPUSpec


class CostCalculator:
    """Calculate and compare GPU costs across providers."""

    def __init__(self):
        self.gpu_catalog = GPU_CATALOG
        self.pricing = PROVIDER_PRICING

    def compare(
        self,
        gpus: list[str],
        providers: list[str] = None,
        usage_hours: float = 730,
        usage_type: str = "on-demand",
    ) -> list[CostResult]:
        """Compare costs for specified GPUs across providers."""
        if providers is None:
            providers = list(self.pricing.keys())

        results = []
        for gpu_name in gpus:
            gpu_spec = self.gpu_catalog.get(gpu_name)
            if gpu_spec is None:
                print(f"Warning: Unknown GPU {gpu_name}, skipping")
                continue

            for provider in providers:
                provider_pricing = self.pricing.get(provider, {})
                gpu_pricing = provider_pricing.get(gpu_name, {})

                rate = gpu_pricing.get(usage_type)
                if rate is None:
                    rate = gpu_pricing.get("on-demand", 0)

                if rate == 0:
                    continue

                total_cost = rate * usage_hours
                cost_per_tflop = total_cost / (gpu_spec.tflops_fp16 * usage_hours) if gpu_spec.tflops_fp16 > 0 else 0
                cost_per_gb = total_cost / gpu_spec.vram_gb
                cost_per_compute = total_cost / (gpu_spec.tflops_fp16 * usage_hours / gpu_spec.tdp_watts * 1000) if gpu_spec.tdp_watts > 0 else 0

                results.append(CostResult(
                    gpu=gpu_name,
                    provider=provider,
                    usage_type=usage_type,
                    hourly_rate=rate,
                    usage_hours=usage_hours,
                    total_cost=total_cost,
                    cost_per_tflop=cost_per_tflop,
                    cost_per_gb_vram=cost_per_gb,
                    cost_per_hour_compute=cost_per_compute,
                    gpu_specs=gpu_spec,
                ))

        return sorted(results, key=lambda x: x.total_cost)

    def find_best_value(
        self,
        workload: str = "training",
        budget: Optional[float] = None,
        providers: list[str] = None,
    ) -> list[CostResult]:
        """Find best value GPUs for a workload type."""
        # Filter GPUs by workload suitability
        workload_gpus = {
            "training": ["NVIDIA H100 80GB", "NVIDIA A100 80GB", "AMD MI300X 192GB"],
            "inference": ["NVIDIA L4 24GB", "NVIDIA A10G 24GB", "AMD MI210 64GB"],
            "fine_tuning": ["NVIDIA A100 80GB", "AMD MI300X 192GB", "NVIDIA L40S 48GB"],
        }
        gpus = workload_gpus.get(workload, workload_gpus["training"])
        results = self.compare(gpus=gpus, providers=providers, usage_hours=730)

        if budget:
            results = [r for r in results if r.total_cost <= budget]

        return results

    def get_monthly_estimate(
        self,
        gpu: str,
        provider: str,
        hours_per_day: float = 8,
        days_per_month: int = 30,
        usage_type: str = "on-demand",
    ) -> dict:
        """Get monthly cost estimate."""
        total_hours = hours_per_day * days_per_month
        results = self.compare(
            gpus=[gpu],
            providers=[provider],
            usage_hours=total_hours,
            usage_type=usage_type,
        )
        if results:
            return {
                "gpu": gpu,
                "provider": provider,
                "hours_per_month": total_hours,
                "monthly_cost": results[0].total_cost,
                "daily_cost": results[0].total_cost / days_per_month,
                "hourly_cost": results[0].hourly_rate,
            }
        return {"error": f"No pricing data for {gpu} on {provider}"}
