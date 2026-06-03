"""
Visualization Module
Generate charts and visual comparisons for ML cloud cost analysis.
"""

from typing import Optional, List
import json


class Visualizer:
    """Generate visual representations of cost comparisons."""

    def __init__(self):
        self.colors = {
            "aws": "#FF9900",
            "gcp": "#4285F4",
            "azure": "#00BCF2",
            "digitalocean": "#0080FF",
            "akamai": "#0096D6",
        }

    def cost_bar_chart_data(self, comparison_result) -> dict:
        """Generate data for a cost bar chart."""
        labels = []
        values = []
        colors = []

        sorted_bd = sorted(comparison_result.breakdowns, key=lambda b: b.total_cost)
        for bd in sorted_bd:
            labels.append(f"{bd.provider}\n{bd.instance_type}")
            values.append(round(bd.total_cost, 2))
            colors.append(self.colors.get(bd.provider, "#888888"))

        return {
            "type": "bar",
            "title": "Total Cost Comparison",
            "labels": labels,
            "values": values,
            "colors": colors,
            "y_label": "Total Cost ($)",
        }

    def performance_scatter_data(self, perf_results: list) -> dict:
        """Generate data for performance vs cost scatter plot."""
        points = []
        for r in perf_results:
            points.append({
                "x": r.hourly_cost,
                "y": r.tflops_per_dollar,
                "label": f"{r.provider} - {r.instance_type}",
                "color": self.colors.get(r.provider, "#888888"),
                "size": r.performance_score * 20,
            })

        return {
            "type": "scatter",
            "title": "Performance per Dollar",
            "x_label": "Hourly Cost ($)",
            "y_label": "TFLOPS per Dollar",
            "points": points,
        }

    def generate_html_chart(self, chart_data: dict) -> str:
        """Generate an inline HTML chart using Chart.js."""
        chart_type = chart_data["type"]
        chart_id = f"chart_{chart_type}_{hash(str(chart_data)) % 10000}"

        if chart_type == "bar":
            return self._bar_chart_html(chart_id, chart_data)
        elif chart_type == "scatter":
            return self._scatter_chart_html(chart_id, chart_data)
        elif chart_type == "radar":
            return self._radar_chart_html(chart_id, chart_data)
        return f"<div>Unsupported chart type: {chart_type}</div>"

    def _bar_chart_html(self, chart_id: str, data: dict) -> str:
        labels_json = json.dumps(data["labels"])
        values_json = json.dumps(data["values"])
        colors_json = json.dumps(data["colors"])

        return f"""
        <div style="margin:20px 0;">
            <canvas id="{chart_id}" width="800" height="400"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'bar',
            data: {{
                labels: {labels_json},
                datasets: [{{
                    label: '{data.get("y_label", "Cost")}',
                    data: {values_json},
                    backgroundColor: {colors_json},
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: '{data["title"]}' }} }},
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
        </script>
        """

    def _scatter_chart_html(self, chart_id: str, data: dict) -> str:
        datasets = {}
        for point in data["points"]:
            provider = point["label"].split(" - ")[0]
            if provider not in datasets:
                datasets[provider] = {
                    "label": provider,
                    "data": [],
                    "backgroundColor": point["color"],
                }
            datasets[provider]["data"].append({
                "x": point["x"],
                "y": point["y"],
            })

        return f"""
        <div style="margin:20px 0;">
            <canvas id="{chart_id}" width="800" height="400"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'scatter',
            data: {{
                datasets: {json.dumps(list(datasets.values()))}
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: '{data["title"]}' }} }},
                scales: {{
                    x: {{ title: {{ display: true, text: '{data["x_label"]}' }} }},
                    y: {{ title: {{ display: true, text: '{data["y_label"]}' }} }}
                }}
            }}
        }});
        </script>
        """

    def _radar_chart_html(self, chart_id: str, data: dict) -> str:
        return f"""
        <div style="margin:20px 0;">
            <canvas id="{chart_id}" width="600" height="600"></canvas>
        </div>
        <script>
        new Chart(document.getElementById('{chart_id}'), {{
            type: 'radar',
            data: {json.dumps(data.get('datasets', {}))},
            options: {{ responsive: true }}
        }});
        </script>
        """

    def comparison_table_html(self, comparison_result) -> str:
        """Generate an HTML table from comparison results."""
        sorted_bd = sorted(comparison_result.breakdowns, key=lambda b: b.total_cost)
        rows = ""
        for i, bd in enumerate(sorted_bd):
            highlight = 'style="background:#d4edda;font-weight:bold;"' if i == 0 else ""
            spot = f"${bd.spot_hourly_rate:.2f}" if bd.spot_hourly_rate else "—"
            rows += f"""
            <tr {highlight}>
                <td>{bd.provider}</td>
                <td>{bd.instance_type}</td>
                <td>${bd.hourly_rate:.2f}</td>
                <td>{spot}</td>
                <td>{bd.gpu_count}</td>
                <td>{bd.duration_hours:.1f}h</td>
                <td>${bd.total_cost:.2f}</td>
            </tr>"""

        return f"""
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%;">
            <thead style="background:#343a40;color:white;">
                <tr>
                    <th>Provider</th><th>Instance</th><th>On-Demand $/hr</th>
                    <th>Spot $/hr</th><th>GPUs</th><th>Duration</th><th>Total</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """
