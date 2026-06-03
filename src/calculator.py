"""
Cost Calculator Engine
Core module for computing ML workload costs across cloud providers.
Supports training, inference, fine-tuning, and batch processing workloads.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class WorkloadType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    FINE_TUNING = "fine_tuning"
    BATCH = "batch"
    PREEMPTIBLE = "preemptible"


@dataclass
class WorkloadSpec:
    """Specification for an ML workload."""
    gpu_type: str
    gpu_count: int
    vcpus_per_gpu: int = 8
    ram_gb_per_gpu: int = 64
    duration_hours: float = 1.0
    workload_type: WorkloadType = WorkloadType.TRAINING
    storage_gb: float = 0
    data_transfer_gb: float = 0
    spot_enabled: bool = False


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a single provider."""
    provider: str
    instance_type: str
    hourly_rate: float
    spot_hourly_rate: Optional[float] = None
    duration_hours: float = 1.0
    gpu_count: int = 1
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    transfer_cost: float = 0.0
    total_cost: float = 0.0
    spot_savings_pct: float = 0.0

    def compute(self) -> "CostBreakdown":
        """Calculate total cost from components."""
        rate = self.spot_hourly_rate if self.spot_hourly_rate else self.hourly_rate
        self.compute_cost = rate * self.duration_hours * self.gpu_count
        self.total_cost = self.compute_cost + self.storage_cost + self.transfer_cost
        if self.hourly_rate > 0 and self.spot_hourly_rate:
            self.spot_savings_pct = (1 - self.spot_hourly_rate / self.hourly_rate) * 100
        return self

    def summary(self) -> str:
        lines = [
            f"Provider:     {self.provider}",
            f"Instance:     {self.instance_type}",
            f"Hourly Rate:  ${self.hourly_rate:.2f}/hr",
        ]
        if self.spot_hourly_rate:
            lines.append(f"Spot Rate:    ${self.spot_hourly_rate:.2f}/hr ({self.spot_savings_pct:.0f}% savings)")
        lines.extend([
            f"GPU Count:    {self.gpu_count}",
            f"Duration:     {self.duration_hours:.1f} hours",
            f"Compute:      ${self.compute_cost:.2f}",
            f"Storage:      ${self.storage_cost:.2f}",
            f"Transfer:     ${self.transfer_cost:.2f}",
            f"TOTAL:        ${self.total_cost:.2f}",
        ])
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of comparing costs across multiple providers."""
    workload: WorkloadSpec
    breakdowns: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"COST COMPARISON — {self.workload.gpu_type.upper()} x{self.workload.gpu_count}",
            f"Workload: {self.workload.workload_type.value}",
            f"Duration: {self.workload.duration_hours:.1f} hours",
            "=" * 60,
        ]
        sorted_breakdowns = sorted(self.breakdowns, key=lambda b: b.total_cost)
        for i, bd in enumerate(sorted_breakdowns):
            prefix = "★ BEST" if i == 0 else f"  #{i+1}"
            lines.append(f"\n{prefix} — {bd.provider}")
            lines.append(bd.summary())
        cheapest = sorted_breakdowns[0].total_cost if sorted_breakdowns else 0
        most_expensive = sorted_breakdowns[-1].total_cost if sorted_breakdowns else 0
        if most_expensive > 0:
            savings_pct = (1 - cheapest / most_expensive) * 100
            lines.append(f"\n{'=' * 60}")
            lines.append(f"POTENTIAL SAVINGS: {savings_pct:.0f}% (${most_expensive - cheapest:.2f})")
        return "\n".join(lines)

    def cheapest_provider(self) -> Optional[CostBreakdown]:
        if not self.breakdowns:
            return None
        return min(self.breakdowns, key=lambda b: b.total_cost)


class CostCalculator:
    """Main cost calculator engine."""

    STORAGE_COST_PER_GB_MONTH = {
        "aws": 0.023,
        "gcp": 0.020,
        "azure": 0.018,
        "digitalocean": 0.020,
        "akamai": 0.019,
    }

    TRANSFER_COST_PER_GB = {
        "aws": 0.09,
        "gcp": 0.12,
        "azure": 0.087,
        "digitalocean": 0.01,
        "akamai": 0.05,
    }

    def __init__(self, provider_registry=None):
        self.registry = provider_registry

    def estimate_training_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        training_hours: float,
        providers: Optional[list] = None,
        spot: bool = False,
        storage_gb: float = 100,
    ) -> ComparisonResult:
        """Estimate training cost across providers."""
        workload = WorkloadSpec(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            duration_hours=training_hours,
            workload_type=WorkloadType.TRAINING,
            storage_gb=storage_gb,
            spot_enabled=spot,
        )
        return self._compare(workload, providers)

    def estimate_inference_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        hours_per_day: float,
        days_per_month: int,
        providers: Optional[list] = None,
        spot: bool = False,
    ) -> ComparisonResult:
        """Estimate monthly inference cost."""
        total_hours = hours_per_day * days_per_month
        workload = WorkloadSpec(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            duration_hours=total_hours,
            workload_type=WorkloadType.INFERENCE,
            spot_enabled=spot,
        )
        return self._compare(workload, providers)

    def estimate_custom_cost(self, workload: WorkloadSpec, providers: Optional[list] = None) -> ComparisonResult:
        """Estimate cost for a custom workload specification."""
        return self._compare(workload, providers)

    def _compare(self, workload: WorkloadSpec, providers: Optional[list] = None) -> ComparisonResult:
        """Run cost comparison across providers."""
        result = ComparisonResult(workload=workload)

        if self.registry is None:
            return result

        provider_list = providers or self.registry.list_providers()

        for provider_name in provider_list:
            provider = self.registry.get(provider_name)
            if provider is None:
                continue

            instance = provider.find_instance(workload.gpu_type, workload.gpu_count)
            if instance is None:
                continue

            bd = CostBreakdown(
                provider=provider_name,
                instance_type=instance["name"],
                hourly_rate=instance["hourly_cost"],
                spot_hourly_rate=instance.get("spot_hourly_rate"),
                duration_hours=workload.duration_hours,
                gpu_count=workload.gpu_count,
                storage_cost=self._estimate_storage(
                    provider_name, workload.storage_gb, workload.duration_hours
                ),
                transfer_cost=self._estimate_transfer(
                    provider_name, workload.data_transfer_gb
                ),
            )
            bd.compute()
            result.breakdowns.append(bd)

        return result

    def _estimate_storage(self, provider: str, storage_gb: float, hours: float) -> float:
        """Estimate storage cost."""
        monthly_rate = self.STORAGE_COST_PER_GB_MONTH.get(provider, 0.023)
        months = hours / (24 * 30)
        return storage_gb * monthly_rate * months

    def _estimate_transfer(self, provider: str, transfer_gb: float) -> float:
        """Estimate data transfer cost."""
        rate = self.TRANSFER_COST_PER_GB.get(provider, 0.10)
        return transfer_gb * rate

    def total_cost_with_storage(
        self, provider: str, gpu_type: str, gpu_count: int,
        hours: float, storage_gb: float = 0
    ) -> float:
        """Quick single-provider total cost."""
        if self.registry is None:
            return 0.0
        p = self.registry.get(provider)
        if p is None:
            return 0.0
        inst = p.find_instance(gpu_type, gpu_count)
        if inst is None:
            return 0.0
        compute = inst["hourly_cost"] * hours * gpu_count
        storage = self._estimate_storage(provider, storage_gb, hours)
        return compute + storage
