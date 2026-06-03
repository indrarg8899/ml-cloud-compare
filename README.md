# ml-cloud-compare

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cloud](https://img.shields.io/badge/Cloud-AMD%20%7C%20NVIDIA-orange.svg)](https://cloud.google.com/)
[![ML](https://img.shields.io/badge/ML-Training%20Cost%20Analysis-green.svg)](https://pytorch.org/)

Cloud GPU cost-performance comparison tool for ML training workloads across AMD and NVIDIA providers.

## Architecture

```
┌──────────────────────────────────────────┐
│          ml-cloud-compare                │
├────────────┬────────────┬────────────────┤
│   Cost     │ Perf       │   Report       │
│   Calculator│ Benchmarks│   Generator    │
├────────────┴────────────┴────────────────┤
│         Provider API Adapters            │
├──────┬──────┬──────┬──────┬──────────────┤
│ AWS  │ Azure│ GCP  │Lambda│ CoreWeave    │
└──────┴──────┴──────┴──────┴──────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Compare GPU costs
python -m src.compare --gpus A100,H100,MI300X --workload training

# Generate full report
python -m src.report --workload training --providers aws,azure,gcp

# Interactive dashboard
python -m src.dashboard --port 8501
```

## Features

- **Multi-Provider Support** - AWS, Azure, GCP, Lambda, CoreWeave, vast.ai
- **AMD vs NVIDIA** - Side-by-side comparison including MI300X, H100, A100, L4
- **Cost Analysis** - On-demand, spot/preemptible, reserved, and enterprise pricing
- **Performance Benchmarks** - Training throughput, inference latency, memory efficiency
- **Workload Profiles** - LLM training, fine-tuning, inference, vision, RLHF
- **TCO Calculator** - Total cost including networking, storage, and support
- **Export Reports** - PDF, HTML, CSV, and Jupyter notebook exports

## Usage

### Cost Comparison

```python
from src.compare import CostCalculator

calc = CostCalculator()

# Compare GPUs across providers
results = calc.compare(
    gpus=["NVIDIA H100 80GB", "NVIDIA A100 80GB", "AMD MI300X 192GB"],
    providers=["aws", "azure", "gcp"],
    usage_hours=730,  # 1 month
    usage_type="on-demand",
)

for r in results:
    print(f"{r.gpu} on {r.provider}: ${r.total_cost:.2f}/mo ({r.cost_per_tflop:.4f}/TFLOP-hr)")
```

### Performance Analysis

```python
from src.analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# Benchmark training throughput
benchmarks = analyzer.benchmark_workload(
    workload="llm_training",
    model_size="7B",
    gpu="MI300X",
    batch_size=32,
)
print(f"Tokens/sec: {benchmarks['throughput']}")
print(f"Memory utilization: {benchmarks['memory_pct']:.1f}%")
```

### Generate Report

```python
from src.report import ReportGenerator

gen = ReportGenerator()
report = gen.generate(
    title="AMD vs NVIDIA Training Cost Analysis",
    workloads=["llm_pretraining", "llm_finetuning", "inference"],
    providers=["aws", "gcp", "lambda"],
    output_format="html",
)
report.save("reports/comparison.html")
```

## GPU Pricing Overview

| GPU | Provider | On-Demand $/hr | Spot $/hr | VRAM |
|-----|---------|---------------|----------|------|
| NVIDIA H100 | AWS | $3.50 | $1.40 | 80 GB |
| NVIDIA A100 | AWS | $2.48 | $0.99 | 80 GB |
| AMD MI300X | CoreWeave | $2.25 | $0.90 | 192 GB |
| NVIDIA H100 | GCP | $3.22 | $0.97 | 80 GB |
| AMD MI300X | Lambda | $2.00 | N/A | 192 GB |

## License

MIT License. See [LICENSE](LICENSE) for details.
