# ML Cloud Compare

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](tests/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-orange.svg)](CONTRIBUTING.md)
[![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen.svg)](https://github.com/indrarg8899/ml-cloud-compare)

> **Compare ML cloud GPU/CPU costs across major providers. Get data-driven recommendations for optimal price-performance.**

Stop overpaying for ML compute. Compare real pricing data from AWS, GCP, Azure, DigitalOcean, and Akamai Cloud to find the best deal for your training and inference workloads.

---

## 🏆 Provider Comparison (Top GPUs, On-Demand)

| GPU | AWS (p5) | GCP (a3) | Azure (ND) | DO (GPU) | Akamai |
|-----|----------|----------|------------|----------|--------|
| NVIDIA H100 80GB | $32.77/hr | $31.21/hr | $33.40/hr | $28.80/hr | $30.50/hr |
| NVIDIA A100 80GB | $14.39/hr | $13.82/hr | $14.69/hr | $12.50/hr | $13.20/hr |
| NVIDIA L4 24GB | $2.38/hr | $2.21/hr | $2.45/hr | — | $2.10/hr |
| NVIDIA T4 16GB | $1.10/hr | $1.05/hr | $1.14/hr | $0.95/hr | $1.00/hr |
| AMD MI300X 192GB | $28.50/hr | — | — | — | — |

*Prices as of 2025. See `data/gpu_pricing.csv` for full dataset.*

---

## ✨ Features

- **📊 Multi-Provider Cost Comparison** — AWS, GCP, Azure, DigitalOcean, Akamai
- **🧠 Workload Analysis** — Estimate costs for training, fine-tuning, inference
- **🎯 Smart Recommendations** — AI-driven provider/instance suggestions
- **📈 Benchmark-to-Cost Ratios** — Performance per dollar analysis
- **📉 Visual Reports** — HTML reports with charts and comparisons
- **🔧 CLI Tools** — Quick comparisons from the command line
- **💰 Cost Auditing** — Analyze and optimize existing cloud spend
- **🔄 Spot/Preemptible Pricing** — Compare on-demand vs spot savings
- **📋 CSV Data** — Transparent, auditable pricing datasets

---

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/indrarg8899/ml-cloud-compare.git
cd ml-cloud-compare
pip install -r requirements.txt

# Compare GPU costs
python scripts/compare.py --gpu h100 --providers aws,gcp,azure

# Analyze workload cost
python -c "
from src.calculator import CostCalculator
from src.providers import ProviderRegistry

calc = CostCalculator(ProviderRegistry())
result = calc.estimate_training_cost(
    gpu_type='a100',
    gpu_count=8,
    training_hours=72,
    providers=['aws', 'gcp', 'azure']
)
print(result.summary())
"

# Generate full HTML report
python -c "
from src.report import ReportGenerator
gen = ReportGenerator()
gen.generate(
    gpu_type='h100',
    workload='training',
    output='comparison_report.html'
)
print('Report saved to comparison_report.html')
"

# Run benchmarks
python benchmarks/benchmark_mi300x.py

# Audit costs
python scripts/audit_costs.py --config configs/default.yml
```

---

## 📁 Project Structure

```
ml-cloud-compare/
├── src/
│   ├── calculator.py      # Core cost calculation engine
│   ├── providers.py       # Provider pricing data & registry
│   ├── performance.py     # Benchmark-to-cost ratio analysis
│   ├── analyzer.py        # Workload type analyzer
│   ├── recommender.py     # Smart recommendation engine
│   ├── visualizer.py      # Chart generation
│   └── report.py          # HTML report generator
├── data/
│   ├── gpu_pricing.csv    # GPU instance pricing dataset
│   └── cpu_pricing.csv    # CPU instance pricing dataset
├── configs/
│   └── default.yml        # Default configuration
├── benchmarks/
│   └── benchmark_mi300x.py  # AMD MI300X benchmark suite
├── docs/
│   ├── methodology.md     # Pricing methodology & sources
│   ├── providers.md       # Provider-specific details
│   └── cost_optimization.md  # Optimization strategies
├── scripts/
│   ├── compare.py         # CLI comparison tool
│   └── audit_costs.py     # Cost auditing tool
├── tests/
│   └── test_calculator.py # Test suite
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 📖 Documentation

- **[Methodology](docs/methodology.md)** — How we collect and verify pricing data
- **[Provider Details](docs/providers.md)** — Instance types, regions, spot pricing
- **[Cost Optimization](docs/cost_optimization.md)** — Strategies to reduce ML compute costs

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Add pricing data or features
4. Run tests (`python -m pytest tests/`)
5. Submit PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Star History

If this tool saved you money, give it a ⭐!
