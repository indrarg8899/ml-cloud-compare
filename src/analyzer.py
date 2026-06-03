"""
Workload Analyzer
Analyzes ML workload requirements and maps them to optimal GPU/instance configurations.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ModelSize(Enum):
    SMALL = "small"          # <1B params
    MEDIUM = "medium"        # 1-10B params
    LARGE = "large"          # 10-70B params
    EXTRA_LARGE = "xlarge"   # 70B+ params


class TaskType(Enum):
    FINE_TUNING = "fine_tuning"
    PRETRAINING = "pretraining"
    INFERENCE = "inference"
    RAG = "rag"
    EMBEDDING = "embedding"
    IMAGE_GEN = "image_generation"
    BATCH_INFERENCE = "batch_inference"


@dataclass
class WorkloadProfile:
    """Analyzed workload profile with resource recommendations."""
    task_type: TaskType
    model_size: ModelSize
    estimated_params_b: float
    recommended_gpu: str
    recommended_gpu_count: int
    estimated_vram_gb: float
    estimated_duration_hours: float
    min_ram_gb: int
    priority_factors: list

    def summary(self) -> str:
        lines = [
            f"Task:          {self.task_type.value}",
            f"Model Size:    {self.model_size.value} ({self.estimated_params_b:.1f}B params)",
            f"Recommended:   {self.recommended_gpu} x{self.recommended_gpu_count}",
            f"Est. VRAM:     {self.estimated_vram_gb:.0f} GB",
            f"Est. Duration: {self.estimated_duration_hours:.1f} hours",
            f"Min RAM:       {self.min_ram_gb} GB",
            f"Priorities:    {', '.join(self.priority_factors)}",
        ]
        return "\n".join(lines)


class WorkloadAnalyzer:
    """Analyze ML workloads and recommend configurations."""

    # VRAM requirements per model size (rough estimates for FP16/BF16 training)
    VRAM_REQUIREMENTS = {
        "small": 4,
        "medium": 16,
        "large": 80,
        "xlarge": 320,
    }

    # Model size thresholds (billions of parameters)
    SIZE_THRESHOLDS = [
        (1.0, ModelSize.SMALL),
        (10.0, ModelSize.MEDIUM),
        (70.0, ModelSize.LARGE),
        (float("inf"), ModelSize.EXTRA_LARGE),
    ]

    # Duration estimates by task type and model size (hours)
    DURATION_ESTIMATES = {
        TaskType.FINE_TUNING: {
            ModelSize.SMALL: 2,
            ModelSize.MEDIUM: 8,
            ModelSize.LARGE: 48,
            ModelSize.EXTRA_LARGE: 168,
        },
        TaskType.PRETRAINING: {
            ModelSize.SMALL: 24,
            ModelSize.MEDIUM: 168,
            ModelSize.LARGE: 720,
            ModelSize.EXTRA_LARGE: 2160,
        },
        TaskType.INFERENCE: {
            ModelSize.SMALL: 720,
            ModelSize.MEDIUM: 720,
            ModelSize.LARGE: 720,
            ModelSize.EXTRA_LARGE: 720,
        },
        TaskType.RAG: {
            ModelSize.SMALL: 720,
            ModelSize.MEDIUM: 720,
            ModelSize.LARGE: 720,
            ModelSize.EXTRA_LARGE: 720,
        },
        TaskType.EMBEDDING: {
            ModelSize.SMALL: 720,
            ModelSize.MEDIUM: 720,
            ModelSize.LARGE: 720,
            ModelSize.EXTRA_LARGE: 720,
        },
        TaskType.IMAGE_GEN: {
            ModelSize.SMALL: 720,
            ModelSize.MEDIUM: 720,
            ModelSize.LARGE: 720,
            ModelSize.EXTRA_LARGE: 720,
        },
        TaskType.BATCH_INFERENCE: {
            ModelSize.SMALL: 24,
            ModelSize.MEDIUM: 72,
            ModelSize.LARGE: 168,
            ModelSize.EXTRA_LARGE: 336,
        },
    }

    def analyze(self, task_type: TaskType, params_b: float, duration_override: Optional[float] = None) -> WorkloadProfile:
        """Analyze a workload and return recommended configuration."""
        model_size = self._classify_size(params_b)
        vram_needed = self._estimate_vram(params_b)
        gpu_type, gpu_count = self._recommend_gpu(model_size, vram_needed)
        ram_needed = self._estimate_ram(model_size, gpu_count)
        duration = duration_override or self._estimate_duration(task_type, model_size)
        priorities = self._determine_priorities(task_type, model_size)

        return WorkloadProfile(
            task_type=task_type,
            model_size=model_size,
            estimated_params_b=params_b,
            recommended_gpu=gpu_type,
            recommended_gpu_count=gpu_count,
            estimated_vram_gb=vram_needed,
            estimated_duration_hours=duration,
            min_ram_gb=ram_needed,
            priority_factors=priorities,
        )

    def analyze_from_description(self, description: str, params_b: float) -> WorkloadProfile:
        """Analyze workload from a text description."""
        task_type = self._infer_task(description)
        return self.analyze(task_type, params_b)

    def _classify_size(self, params_b: float) -> ModelSize:
        for threshold, size in self.SIZE_THRESHOLDS:
            if params_b <= threshold:
                return size
        return ModelSize.EXTRA_LARGE

    def _estimate_vram(self, params_b: float) -> float:
        """Estimate VRAM needed (in GB) for training at BF16."""
        # ~2 bytes per parameter for BF16 + overhead
        base = params_b * 2.5
        # Add overhead for optimizer states (~4x for AdamW)
        training = base * 5
        # Account for activations (rough)
        return max(training, 4)

    def _recommend_gpu(self, model_size: ModelSize, vram_needed: float) -> tuple:
        """Recommend GPU type and count."""
        gpu_configs = [
            ("t4", 16, 1),
            ("l4", 24, 1),
            ("a10", 24, 1),
            ("a100", 80, 1),
            ("h100", 80, 1),
            ("h200", 141, 1),
        ]

        for gpu_type, vram_per_gpu, _ in gpu_configs:
            count = max(1, int(vram_needed / vram_per_gpu) + (1 if vram_needed % vram_per_gpu > 0 else 0))
            if count <= 8:
                return gpu_type, count

        return "h100", 8

    def _estimate_ram(self, model_size: ModelSize, gpu_count: int) -> int:
        """Estimate system RAM needed."""
        base = 32
        size_multiplier = {
            ModelSize.SMALL: 1,
            ModelSize.MEDIUM: 2,
            ModelSize.LARGE: 4,
            ModelSize.EXTRA_LARGE: 8,
        }
        return base * size_multiplier.get(model_size, 1)

    def _estimate_duration(self, task_type: TaskType, model_size: ModelSize) -> float:
        """Estimate task duration in hours."""
        estimates = self.DURATION_ESTIMATES.get(task_type, {})
        return estimates.get(model_size, 24)

    def _infer_task(self, description: str) -> TaskType:
        """Infer task type from description."""
        desc_lower = description.lower()
        if any(w in desc_lower for w in ["finetune", "fine-tune", "sft", "rlhf", "dpo"]):
            return TaskType.FINE_TUNING
        if any(w in desc_lower for w in ["pretrain", "pre-train", "continue pretrain"]):
            return TaskType.PRETRAINING
        if any(w in desc_lower for w in ["embed", "vector"]):
            return TaskType.EMBEDDING
        if any(w in desc_lower for w in ["rag", "retrieval"]):
            return TaskType.RAG
        if any(w in desc_lower for w in ["image", "stable diffusion", "flux"]):
            return TaskType.IMAGE_GEN
        if any(w in desc_lower for w in ["batch", "offline"]):
            return TaskType.BATCH_INFERENCE
        return TaskType.INFERENCE

    def _determine_priorities(self, task_type: TaskType, model_size: ModelSize) -> list:
        """Determine priority factors for the workload."""
        priorities = []
        if task_type == TaskType.INFERENCE:
            priorities.extend(["low-latency", "cost-efficiency"])
        elif task_type == TaskType.FINE_TUNING:
            priorities.extend(["cost-efficiency", "gpu-memory"])
        elif task_type == TaskType.PRETRAINING:
            priorities.extend(["throughput", "multi-gpu", "cost"])
        else:
            priorities.extend(["cost", "availability"])
        if model_size in (ModelSize.LARGE, ModelSize.EXTRA_LARGE):
            priorities.append("high-memory-gpu")
        return priorities
