"""
Benchmark-to-Cost Ratio Analysis
Compute performance per dollar using MLPerf and synthetic benchmarks.
"""

from dataclasses import dataclass
from typing import Optional


# Benchmark data: GPU → performance metrics
# TFLOPS (FP16), memory bandwidth (GB/s), inference throughput (tokens/sec on Llama 7B)
BENCHMARK_DATA = {
    "h100": {
        "tflops_fp16": 989,
        "tflops_fp8": 1979,
        "memory_bw_gbps": 3350,
        "memory_gb": 80,
        "inference_throughput": 12500,
        "training_efficiency": 1.00,
    },
    "h200": {
        "tflops_fp16": 989,
        "tflops_fp8": 1979,
        "memory_bw_gbps": 4800,
        "memory_gb": 141,
        "inference_throughput": 18000,
        "training_efficiency": 1.10,
    },
    "a100": {
        "tflops_fp16": 312,
        "tflops_fp8": 624,
        "memory_bw_gbps": 2039,
        "memory_gb": 80,
        "inference_throughput": 4200,
        "training_efficiency": 0.45,
    },
    "a10g": {
        "tflops_fp16": 31,
        "tflops_fp8": 62,
        "memory_bw_gbps": 600,
        "memory_gb": 24,
        "inference_throughput": 800,
        "training_efficiency": 0.15,
    },
    "l4": {
        "tflops_fp16": 30,
        "tflops_fp8": 60,
        "memory_bw_gbps": 300,
        "memory_gb": 24,
        "inference_throughput": 1500,
        "training_efficiency": 0.12,
    },
    "t4": {
        "tflops_fp16": 8.1,
        "tflops_fp8": 16.2,
        "memory_bw_gbps": 300,
        "memory_gb": 16,
        "inference_throughput": 600,
        "training_efficiency": 0.05,
    },
    "a10": {
        "tflops_fp16": 31,
        "tflops_fp8": 62,
        "memory_bw_gbps": 600,
        "memory_gb": 24,
        "inference_throughput": 900,
        "training_efficiency": 0.14,
    },
    "rtx6000": {
        "tflops_fp16": 16,
        "tflops_fp8": 32,
        "memory_bw_gbps": 672,
        "memory_gb": 48,
        "inference_throughput": 500,
        "training_efficiency": 0.08,
    },
}


@dataclass
class PerformanceMetric:
    """A single performance-per-dollar metric."""
    provider: str
    gpu_type: str
    instance_type: str
    hourly_cost: float
    tflops_per_dollar: float
    inference_per_dollar: float
    memory_per_dollar: float
    composite_score: float

    def summary(self) -> str:
        return (
            f"{self.provider:15s} | {self.gpu_type:8s} | ${self.hourly_cost:>8.2f}/hr | "
            f"{self.tflops_per_dollar:>8.2f} TFLOPS/$ | "
            f"{self.inference_per_dollar:>8.0f} tok/s/$ | "
            f"Score: {self.composite_score:.2f}"
        )


class PerformanceAnalyzer:
    """Compute performance-per-dollar ratios across providers and GPUs."""

    def __init__(self, provider_registry):
        self.registry = provider_registry

    def get_benchmarks(self, gpu_type: str) -> Optional[dict]:
        """Get benchmark data for a GPU type."""
        gpu_type = gpu_type.lower()
        for key in BENCHMARK_DATA:
            if key in gpu_type or gpu_type in key:
                return BENCHMARK_DATA[key]
        return None

    def analyze_provider(self, provider_name: str) -> list:
        """Analyze all instances for a provider."""
        provider = self.registry.get(provider_name)
        if provider is None:
            return []

        results = []
        for inst in provider.instances:
            benchmarks = self.get_benchmarks(inst["gpu_type"])
            if benchmarks is None:
                continue

            hourly = inst["hourly_cost"]
            if hourly <= 0:
                continue

            gpu_count = inst["gpu_count"]
            tflops = benchmarks["tflops_fp16"] * gpu_count
            inference = benchmarks["inference_throughput"] * gpu_count
            memory = benchmarks["memory_gb"] * gpu_count

            tflops_per_dollar = tflops / hourly
            inference_per_dollar = inference / hourly
            memory_per_dollar = memory / hourly

            composite = self._composite_score(
                tflops_per_dollar, inference_per_dollar, memory_per_dollar
            )

            results.append(PerformanceMetric(
                provider=provider_name,
                gpu_type=inst["gpu_type"],
                instance_type=inst["name"],
                hourly_cost=hourly,
                tflops_per_dollar=tflops_per_dollar,
                inference_per_dollar=inference_per_dollar,
                memory_per_dollar=memory_per_dollar,
                composite_score=composite,
            ))

        return sorted(results, key=lambda r: r.composite_score, reverse=True)

    def compare_gpu(self, gpu_type: str) -> list:
        """Compare a specific GPU across all providers."""
        results = []
        for name in self.registry.list_providers():
            results.extend(self.analyze_provider(name))

        gpu_lower = gpu_type.lower()
        filtered = [r for r in results if gpu_lower in r.gpu_type]
        if not filtered:
            return results
        return sorted(filtered, key=lambda r: r.composite_score, reverse=True)

    def ranking(self, top_n: int = 10) -> list:
        """Global ranking of all instances by performance-per-dollar."""
        all_results = []
        for name in self.registry.list_providers():
            all_results.extend(self.analyze_provider(name))
        return all_results[:top_n]

    def _composite_score(self, tflops_pd: float, inference_pd: float, memory_pd: float) -> float:
        """Weighted composite score for overall value."""
        # Normalize to H100 baseline
        h100_tflops_pd = BENCHMARK_DATA["h100"]["tflops_fp16"] / 32.77
        h100_inference_pd = BENCHMARK_DATA["h100"]["inference_throughput"] / 32.77

        tflops_norm = tflops_pd / h100_tflops_pd if h100_tflops_pd else 0
        inference_norm = inference_pd / h100_inference_pd if h100_inference_pd else 0

        return (tflops_norm * 0.4 + inference_norm * 0.4 + (memory_pd / 100) * 0.2)
