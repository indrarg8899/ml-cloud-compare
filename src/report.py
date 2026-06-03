"""
HTML Report Generator
Generate comprehensive HTML reports for ML cloud cost comparisons.
"""

from datetime import datetime
from src.calculator import CostCalculator, WorkloadSpec, WorkloadType
from src.providers import ProviderRegistry
from src.performance import PerformanceAnalyzer
from src.analyzer import WorkloadAnalyzer, TaskType
from src.recommender import Recommender
from src.visualizer import Visualizer


class ReportGenerator:
    """Generate full HTML comparison reports."""

    def __init__(self):
        self.registry = ProviderRegistry()
        self.calculator = CostCalculator(self.registry)
        self.performance = PerformanceAnalyzer(self.registry)
        self.analyzer = WorkloadAnalyzer()
        self.recommender = Recommender(self.registry)
        self.visualizer = Visualizer()

    def generate(
        self,
        gpu_type: str = "h100",
        workload: str = "training",
        output: str = "ml_cloud_report.html",
        params_b: float = 13.0,
        providers: list = None,
    ) -> str:
        """Generate and save an HTML report."""
        provider_list = providers or self.registry.list_providers()

        # Run analysis
        comparison = self.calculator.estimate_training_cost(
            gpu_type=gpu_type,
            gpu_count=8,
            training_hours=48,
            providers=provider_list,
        )

        perf_results = self.performance.compare_gpu(gpu_type)
        recommendations = self.recommender.recommend(
            task_type=workload,
            params_b=params_b,
            providers=provider_list,
        )

        # Generate charts
        bar_chart = self.visualizer.cost_bar_chart_data(comparison)
        scatter_data = self.visualizer.performance_scatter_data(perf_results)

        # Build HTML
        html = self._build_html(
            gpu_type, workload, params_b,
            comparison, perf_results, recommendations,
            bar_chart, scatter_data,
        )

        with open(output, "w") as f:
            f.write(html)

        return output

    def _build_html(
        self, gpu_type, workload, params_b,
        comparison, perf_results, recommendations,
        bar_chart, scatter_data,
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Performance table rows
        perf_rows = ""
        for p in perf_results:
            perf_rows += f"""
            <tr>
                <td>{p.provider}</td>
                <td>{p.instance_type}</td>
                <td>${p.hourly_cost:.2f}</td>
                <td>{p.tflops_per_dollar:.2f}</td>
                <td>{p.inference_per_dollar:.0f}</td>
                <td>{p.composite_score:.2f}</td>
            </tr>"""

        # Recommendation rows
        rec_rows = ""
        for rec in recommendations.recommendations:
            pros = ", ".join(rec.pros) if rec.pros else "—"
            cons = ", ".join(rec.cons) if rec.cons else "—"
            rec_rows += f"""
            <tr>
                <td>#{rec.rank}</td>
                <td><strong>{rec.provider.upper()}</strong></td>
                <td>{rec.instance_type}</td>
                <td>${rec.hourly_cost:.2f}</td>
                <td>${rec.estimated_total:.2f}</td>
                <td>{rec.performance_score:.2f}</td>
                <td>{pros}</td>
                <td>{cons}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Cloud Compute Comparison — {gpu_type.upper()} {workload}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f8f9fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 40px 20px; text-align: center; border-radius: 12px; margin-bottom: 30px; }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ font-size: 1.2em; opacity: 0.9; }}
        .card {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px;
                 box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .card h2 {{ color: #667eea; margin-bottom: 20px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #343a40; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #dee2e6; }}
        tr:hover {{ background: #f8f9fa; }}
        .best {{ background: #d4edda !important; font-weight: bold; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat {{ background: linear-gradient(135deg, #667eea22, #764ba222);
                 padding: 20px; border-radius: 10px; text-align: center; }}
        .stat .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .stat .label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        .chart-container {{ margin: 20px 0; }}
        footer {{ text-align: center; padding: 30px; color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ ML Cloud Compute Comparison</h1>
            <p>{gpu_type.upper()} | {workload.title()} | {params_b}B Params | Generated {now}</p>
        </header>

        <div class="stats">
            <div class="stat">
                <div class="value">{len(self.registry.list_providers())}</div>
                <div class="label">Providers Compared</div>
            </div>
            <div class="stat">
                <div class="value">{len(perf_results)}</div>
                <div class="label">Instances Analyzed</div>
            </div>
            <div class="stat">
                <div class="value">{len(recommendations.recommendations)}</div>
                <div class="label">Recommendations</div>
            </div>
            <div class="stat">
                <div class="value">{recommendations.summary[:50] if recommendations.summary else 'N/A'}...</div>
                <div class="label">Top Pick</div>
            </div>
        </div>

        <div class="card">
            <h2>💰 Cost Comparison</h2>
            <div class="chart-container">
                {self.visualizer.generate_html_chart(bar_chart)}
            </div>
        </div>

        <div class="card">
            <h2>📊 Cost Breakdown</h2>
            {self.visualizer.comparison_table_html(comparison)}
        </div>

        <div class="card">
            <h2>⚡ Performance per Dollar</h2>
            <div class="chart-container">
                {self.visualizer.generate_html_chart(scatter_data)}
            </div>
            <table style="margin-top:20px;">
                <thead>
                    <tr>
                        <th>Provider</th><th>Instance</th><th>$/hr</th>
                        <th>TFLOPS/$</th><th>Inference/$</th><th>Score</th>
                    </tr>
                </thead>
                <tbody>{perf_rows}</tbody>
            </table>
        </div>

        <div class="card">
            <h2>🎯 Recommendations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th><th>Provider</th><th>Instance</th>
                        <th>$/hr</th><th>Total</th><th>Score</th>
                        <th>Pros</th><th>Cons</th>
                    </tr>
                </thead>
                <tbody>{rec_rows}</tbody>
            </table>
        </div>

        <footer>
            <p>ML Cloud Compute Comparison Tool v1.0.0 | Data updated as of {now}</p>
            <p>Prices may vary by region and change frequently. Always verify with provider.</p>
        </footer>
    </div>
</body>
</html>"""
