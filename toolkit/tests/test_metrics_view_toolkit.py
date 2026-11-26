from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry
from ucm_toolkit.cli import main
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.collector import add_url_label
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.config import (
    load_config,
    metric_names_for_scrape,
    resolve_config_path,
)
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.parser import parse_prometheus_text
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.query import QueryEngine
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.storage import MetricsStore


class MetricsViewToolkitTest(unittest.TestCase):
    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_metrics_view_is_registered_top_level_tool(self):
        registry.init_builtin_tools()

        tool = registry.get("metrics-view")

        self.assertEqual(tool.name, "metrics-view")
        self.assertIn("metrics_view", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_can_run_metrics_view_list_configs(self):
        output = io.StringIO()

        with redirect_stdout(output):
            result = main(["run", "metrics-view", "list-configs"])

        self.assertEqual(result, 0)
        configs = output.getvalue()
        self.assertIn("metrics_lite", configs)
        self.assertIn("vllm", configs)
        self.assertIn("connector", configs)
        self.assertIn("store", configs)
        for preset in ("sglang", "hicache", "linker"):
            self.assertIn(preset, configs)

    def test_sglang_presets_resolve_expected_scrape_series(self):
        for preset, series in {
            "sglang": "sglang:time_to_first_token_seconds_bucket",
            "hicache": "sglang:hicache_backup_storage_duration_seconds_bucket",
            "linker": "ucm:linker_load_layer_duration_ms_bucket",
        }.items():
            with self.subTest(preset=preset):
                config = load_config(resolve_config_path(preset))
                self.assertIn(series, metric_names_for_scrape(config))

    def test_linker_window_preserves_node_and_layer_and_omits_missing_series(self):
        config = load_config(resolve_config_path("linker"))
        worker = 'model_name="test",worker_id="tp0",layer="2"'
        metric = "ucm:linker_load_layer_duration_ms"
        urls = ("http://node-a:30000/metrics", "http://node-b:30000/metrics")
        with TemporaryDirectory() as directory:
            with MetricsStore(Path(directory) / "metrics.db") as store:
                for timestamp, count in ((1000, 0), (2000, 1)):
                    for url, elapsed in zip(urls, (0, 8)):
                        samples = parse_prometheus_text(
                            f'{metric}_bucket{{{worker},le="+Inf"}} {count}\n'
                            f"{metric}_sum{{{worker}}} {count * elapsed}\n"
                            f"{metric}_count{{{worker}}} {count}\n"
                        )
                        store.write_samples(add_url_label(samples, url), timestamp)
                rows = QueryEngine(store).query_config(config, 1, end_ms=2000)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row.group["layer"] for row in rows}, {"2"})
        self.assertEqual({row.group["worker_id"] for row in rows}, {"tp0"})
        self.assertEqual(
            {row.group["url"]: row.values["avg"] for row in rows},
            dict(zip(urls, (0, 8))),
        )

    def test_doctor_does_not_report_metrics_view_environment_checks(self):
        output = io.StringIO()

        with redirect_stdout(output):
            result = main(["doctor", "metrics-view"])

        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("metrics-view: no environment checks", text)
        self.assertNotIn("sqlite3", text)
        self.assertNotIn("PyYAML", text)

    def test_metrics_view_readme_documents_dashboard_configs(self):
        standalone_readme = (
            ROOT.parent / "docs/docs-site/docs/en/toolkit/user/metrics-view.md"
        )

        readme = standalone_readme.read_text(encoding="utf-8")
        self.assertIn("grafana_vllm.json", readme)
        self.assertIn("grafana_connector.json", readme)
        self.assertIn("grafana_store.json", readme)


if __name__ == "__main__":
    unittest.main()
