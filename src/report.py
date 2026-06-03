"""
Report generator for GPU cost-performance analysis.

Generates HTML, PDF, and CSV comparison reports.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from .compare import CostCalculator
from .analysis import PerformanceAnalyzer, WorkloadProfile, WORKLOADS


class ReportGenerator:
    """Generate GPU comparison reports."""

    def __init__(self):
        self.calculator = CostCalculator()
        self.analyzer = PerformanceAnalyzer()

    def generate(
        self,
        title: str = "Cloud GPU Comparison Report",
        workloads: list[str] = None,
        providers: list[str] = None,
        output_format: str = "html",
        output_path: Optional[str] = None,
    ) -> "Report":
        """Generate a comparison report."""
        if workloads is None:
            workloads = ["llm_pretraining"]
        if providers is None:
            providers = ["aws", "azure", "gcp"]

        gpus = ["NVIDIA H100 80GB", "NVIDIA A100 80GB", "AMD MI300X 192GB"]

        # Cost comparison
        cost_results = self.calculator.compare(
            gpus=gpus, providers=providers, usage_hours=730
        )

        # Performance analysis
        perf_results = []
        for wk_name in workloads:
            wk = WORKLOADS.get(wk_name)
            if wk:
                perf = self.analyzer.compare_gpus(wk, gpus)
                perf_results.append({"workload": wk_name, "results": perf})

        report = Report(
            title=title,
            generated_at=datetime.now().isoformat(),
            cost_results=cost_results,
            performance_results=perf_results,
        )

        if output_path:
            report.save(output_path, output_format)

        return report


class Report:
    """Report container."""

    def __init__(
        self,
        title: str,
        generated_at: str,
        cost_results: list = None,
        performance_results: list = None,
    ):
        self.title = title
        self.generated_at = generated_at
        self.cost_results = cost_results or []
        self.performance_results = performance_results or []

    def save(self, path: str, fmt: str = "html"):
        """Save report to file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "html":
            self._save_html(output)
        elif fmt == "csv":
            self._save_csv(output)
        elif fmt == "json":
            self._save_json(output)

    def _save_html(self, path: Path):
        html = f"""<!DOCTYPE html>
<html><head><title>{self.title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 40px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #f4f4f4; }}
.cost {{ color: #d32f2f; }}
.perf {{ color: #388e3c; }}
</style></head><body>
<h1>{self.title}</h1>
<p>Generated: {self.generated_at}</p>

<h2>Cost Comparison (Monthly, On-Demand)</h2>
<table><tr><th>GPU</th><th>Provider</th><th>$/hour</th><th>$/month</th><th>$/TFLOP-hr</th></tr>
"""
        for r in self.cost_results:
            html += f'<tr><td>{r.gpu}</td><td>{r.provider}</td><td class="cost">${r.hourly_rate:.2f}</td><td class="cost">${r.total_cost:.2f}</td><td>${r.cost_per_tflop:.6f}</td></tr>\n'

        html += "</table>\n"

        for perf in self.performance_results:
            html += f"<h2>Performance: {perf['workload']}</h2>\n"
            html += "<table><tr><th>GPU</th><th>Tokens/sec</th><th>Memory/GB</th><th>Fits VRAM</th></tr>\n"
            for p in perf["results"]:
                fits = "✅" if p["fits_in_vram"] else "❌"
                html += f'<tr><td>{p["gpu"]}</td><td class="perf">{p["tokens_per_sec"]:.0f}</td><td>{p["memory_per_gpu_gb"]:.1f}</td><td>{fits}</td></tr>\n'
            html += "</table>\n"

        html += "</body></html>"
        path.write_text(html)

    def _save_csv(self, path: Path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["GPU", "Provider", "Type", "$/hour", "Total $"])
            for r in self.cost_results:
                writer.writerow([r.gpu, r.provider, r.usage_type, r.hourly_rate, r.total_cost])

    def _save_json(self, path: Path):
        data = {
            "title": self.title,
            "generated_at": self.generated_at,
            "cost": [
                {
                    "gpu": r.gpu,
                    "provider": r.provider,
                    "hourly_rate": r.hourly_rate,
                    "total_cost": r.total_cost,
                }
                for r in self.cost_results
            ],
        }
        path.write_text(json.dumps(data, indent=2))
