"""
Performance analyzer for cloud GPU training workloads.

Estimates training throughput, memory usage, and cost-efficiency
across different GPU configurations and model sizes.
"""

import math
from dataclasses import dataclass
from typing import Optional

from .compare import GPU_CATALOG


@dataclass
class WorkloadProfile:
    """ML workload profile for analysis."""
    name: str
    model_size_params: float
    batch_size: int
    seq_len: int
    dtype: str = "fp16"
    optimizer: str = "adamw"
    gradient_accumulation: int = 1


# Pre-defined workload profiles
WORKLOADS = {
    "llm_pretraining": WorkloadProfile(
        name="LLM Pre-training (7B)", model_size_params=7e9,
        batch_size=32, seq_len=2048, dtype="fp16",
    ),
    "llm_finetuning": WorkloadProfile(
        name="LLM Fine-tuning (7B)", model_size_params=7e9,
        batch_size=8, seq_len=4096, dtype="fp16",
    ),
    "llm_inference": WorkloadProfile(
        name="LLM Inference (7B)", model_size_params=7e9,
        batch_size=1, seq_len=2048, dtype="fp16",
    ),
    "vision_training": WorkloadProfile(
        name="Vision Model Training", model_size_params=50e6,
        batch_size=128, seq_len=224, dtype="fp32",
    ),
    "rlhf": WorkloadProfile(
        name="RLHF Training", model_size_params=7e9,
        batch_size=4, seq_len=4096, dtype="fp16",
    ),
}


class PerformanceAnalyzer:
    """Analyze ML training performance on different GPUs."""

    def __init__(self):
        self.gpu_catalog = GPU_CATALOG

    def estimate_training_throughput(
        self,
        workload: WorkloadProfile,
        gpu_name: str,
        num_gpus: int = 1,
    ) -> dict:
        """Estimate training throughput for a workload on a GPU."""
        gpu = self.gpu_catalog.get(gpu_name)
        if gpu is None:
            raise ValueError(f"Unknown GPU: {gpu_name}")

        # Model FLOPs estimate (simplified)
        # 6 * params * tokens_per_step for forward + backward
        tokens_per_step = workload.batch_size * workload.seq_len
        model_flops = 6 * workload.model_size_params * tokens_per_step

        # GPU throughput
        effective_tflops = gpu.tflops_fp16 * 0.6 * num_gpus  # 60% MFU estimate
        step_time = model_flops / (effective_tflops * 1e12)

        # Memory estimate
        if workload.dtype == "fp16":
            param_memory = workload.model_size_params * 2  # 2 bytes per param
        else:
            param_memory = workload.model_size_params * 4

        # AdamW optimizer states: 2x params (fp32 m + v)
        optimizer_memory = workload.model_size_params * 2 * 4
        # Gradients: 2 bytes per param (fp16)
        gradient_memory = workload.model_size_params * 2
        # Activations estimate
        activation_memory = tokens_per_step * 4 * 40  # rough estimate

        total_memory_gb = (param_memory + optimizer_memory + gradient_memory + activation_memory) / 1e9
        memory_per_gpu = total_memory_gb / num_gpus

        memory_utilization = memory_per_gpu / gpu.vram_gb

        return {
            "workload": workload.name,
            "gpu": gpu_name,
            "num_gpus": num_gpus,
            "tokens_per_sec": tokens_per_step / step_time,
            "steps_per_sec": 1 / step_time,
            "estimated_step_time_sec": step_time,
            "effective_tflops": effective_tflops,
            "mfu_estimate": 0.6,
            "total_memory_gb": total_memory_gb,
            "memory_per_gpu_gb": memory_per_gpu,
            "memory_utilization_pct": memory_utilization * 100,
            "fits_in_vram": memory_per_gpu <= gpu.vram_gb,
            "tokens_per_gpu_per_sec": tokens_per_step / step_time / num_gpus,
        }

    def compare_gpus(
        self,
        workload: WorkloadProfile,
        gpu_names: list[str],
        num_gpus: int = 1,
    ) -> list[dict]:
        """Compare multiple GPUs for a workload."""
        results = []
        for gpu_name in gpu_names:
            try:
                result = self.estimate_training_throughput(
                    workload, gpu_name, num_gpus
                )
                results.append(result)
            except ValueError as e:
                print(f"Skipping {gpu_name}: {e}")

        return sorted(results, key=lambda x: x["tokens_per_sec"], reverse=True)

    def recommend_gpu(
        self,
        workload: WorkloadProfile,
        budget_per_hour: Optional[float] = None,
        providers: list[str] = None,
    ) -> dict:
        """Recommend best GPU configuration for a workload."""
        candidates = ["NVIDIA H100 80GB", "NVIDIA A100 80GB", "AMD MI300X 192GB"]
        comparisons = self.compare_gpus(workload, candidates, num_gpus=1)

        recommendations = []
        for comp in comparisons:
            gpu_name = comp["gpu"]
            gpu = self.gpu_catalog[gpu_name]
            tokens_per_dollar = comp["tokens_per_sec"] / 2.0  # assume $2/hr avg
            comp["tokens_per_dollar"] = tokens_per_dollar
            recommendations.append(comp)

        return {
            "workload": workload.name,
            "recommendations": sorted(
                recommendations, key=lambda x: x["tokens_per_dollar"], reverse=True
            ),
        }
