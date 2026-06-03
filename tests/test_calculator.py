"""Tests for the ML Cloud Compare calculator module."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calculator import (
    CostCalculator, WorkloadSpec, WorkloadType,
    CostBreakdown, ComparisonResult,
)
from src.providers import ProviderRegistry


class TestWorkloadSpec(unittest.TestCase):
    """Test WorkloadSpec dataclass."""

    def test_defaults(self):
        spec = WorkloadSpec(gpu_type="h100", gpu_count=8)
        self.assertEqual(spec.gpu_type, "h100")
        self.assertEqual(spec.gpu_count, 8)
        self.assertEqual(spec.workload_type, WorkloadType.TRAINING)
        self.assertEqual(spec.duration_hours, 1.0)
        self.assertFalse(spec.spot_enabled)

    def test_custom(self):
        spec = WorkloadSpec(
            gpu_type="a100",
            gpu_count=4,
            workload_type=WorkloadType.INFERENCE,
            duration_hours=720,
            storage_gb=500,
            spot_enabled=True,
        )
        self.assertEqual(spec.workload_type, WorkloadType.INFERENCE)
        self.assertEqual(spec.duration_hours, 720)
        self.assertTrue(spec.spot_enabled)


class TestCostBreakdown(unittest.TestCase):
    """Test CostBreakdown calculations."""

    def test_basic_compute(self):
        bd = CostBreakdown(
            provider="aws",
            instance_type="p4d.24xlarge",
            hourly_rate=32.77,
            duration_hours=10,
            gpu_count=8,
        )
        bd.compute()
        self.assertAlmostEqual(bd.compute_cost, 32.77 * 10 * 8, places=2)
        self.assertAlmostEqual(bd.total_cost, bd.compute_cost, places=2)
        self.assertEqual(bd.spot_savings_pct, 0.0)

    def test_spot_pricing(self):
        bd = CostBreakdown(
            provider="aws",
            instance_type="p4d.24xlarge",
            hourly_rate=32.77,
            spot_hourly_rate=16.39,
            duration_hours=10,
            gpu_count=8,
        )
        bd.compute()
        expected_compute = 16.39 * 10 * 8
        self.assertAlmostEqual(bd.compute_cost, expected_compute, places=2)
        self.assertGreater(bd.spot_savings_pct, 40)

    def test_with_storage_and_transfer(self):
        bd = CostBreakdown(
            provider="aws",
            instance_type="p4d.24xlarge",
            hourly_rate=32.77,
            duration_hours=10,
            gpu_count=8,
            storage_cost=50.00,
            transfer_cost=20.00,
        )
        bd.compute()
        self.assertAlmostEqual(bd.total_cost, bd.compute_cost + 70.00, places=2)

    def test_summary_string(self):
        bd = CostBreakdown(
            provider="aws",
            instance_type="p4d.24xlarge",
            hourly_rate=32.77,
            duration_hours=10,
            gpu_count=8,
        )
        bd.compute()
        summary = bd.summary()
        self.assertIn("aws", summary)
        self.assertIn("p4d.24xlarge", summary)
        self.assertIn("$32.77", summary)


class TestComparisonResult(unittest.TestCase):
    """Test ComparisonResult."""

    def test_empty_result(self):
        result = ComparisonResult(workload=WorkloadSpec(gpu_type="h100", gpu_count=1))
        self.assertIsNone(result.cheapest_provider())

    def test_cheapest_provider(self):
        result = ComparisonResult(workload=WorkloadSpec(gpu_type="h100", gpu_count=8))
        result.breakdowns = [
            CostBreakdown(provider="aws", instance_type="p5", hourly_rate=98.32, duration_hours=1, gpu_count=8).compute(),
            CostBreakdown(provider="gcp", instance_type="a3", hourly_rate=94.13, duration_hours=1, gpu_count=8).compute(),
            CostBreakdown(provider="do", instance_type="h100-8x", hourly_rate=43.10, duration_hours=1, gpu_count=8).compute(),
        ]
        cheapest = result.cheapest_provider()
        self.assertIsNotNone(cheapest)
        self.assertEqual(cheapest.provider, "do")


class TestCostCalculator(unittest.TestCase):
    """Test CostCalculator engine."""

    def setUp(self):
        self.registry = ProviderRegistry()
        self.calculator = CostCalculator(self.registry)

    def test_estimate_training_cost(self):
        result = self.calculator.estimate_training_cost(
            gpu_type="h100",
            gpu_count=8,
            training_hours=48,
        )
        self.assertIsInstance(result, ComparisonResult)
        self.assertGreater(len(result.breakdowns), 0)
        for bd in result.breakdowns:
            self.assertGreater(bd.total_cost, 0)

    def test_estimate_inference_cost(self):
        result = self.calculator.estimate_inference_cost(
            gpu_type="t4",
            gpu_count=1,
            hours_per_day=8,
            days_per_month=30,
        )
        self.assertGreater(len(result.breakdowns), 0)

    def test_specific_providers(self):
        result = self.calculator.estimate_training_cost(
            gpu_type="a100",
            gpu_count=8,
            training_hours=24,
            providers=["aws", "gcp"],
        )
        providers_found = {bd.provider for bd in result.breakdowns}
        self.assertTrue(providers_found.issubset({"aws", "gcp"}))

    def test_storage_cost(self):
        result = self.calculator.estimate_training_cost(
            gpu_type="h100",
            gpu_count=8,
            training_hours=48,
            storage_gb=1000,
        )
        for bd in result.breakdowns:
            self.assertGreater(bd.storage_cost, 0)

    def test_total_cost_with_storage(self):
        total = self.calculator.total_cost_with_storage(
            "aws", "a100", 8, 48, 500
        )
        self.assertGreater(total, 0)

    def test_summary_output(self):
        result = self.calculator.estimate_training_cost(
            gpu_type="h100",
            gpu_count=8,
            training_hours=48,
        )
        summary = result.summary()
        self.assertIn("COST COMPARISON", summary)
        self.assertIn("h100", summary.lower())


class TestProviderRegistry(unittest.TestCase):
    """Test ProviderRegistry."""

    def setUp(self):
        self.registry = ProviderRegistry()

    def test_list_providers(self):
        providers = self.registry.list_providers()
        self.assertIn("aws", providers)
        self.assertIn("gcp", providers)
        self.assertIn("azure", providers)
        self.assertIn("digitalocean", providers)
        self.assertIn("akamai", providers)
        self.assertEqual(len(providers), 5)

    def test_get_provider(self):
        aws = self.registry.get("aws")
        self.assertIsNotNone(aws)
        self.assertEqual(aws.info.display_name, "Amazon Web Services")

    def test_find_instance(self):
        aws = self.registry.get("aws")
        inst = aws.find_instance("h100", 8)
        self.assertIsNotNone(inst)
        self.assertEqual(inst["gpu_type"], "h100")
        self.assertGreaterEqual(inst["gpu_count"], 8)

    def test_no_match(self):
        do = self.registry.get("digitalocean")
        inst = do.find_instance("h100", 16)  # DO doesn't have 16x H100
        self.assertIsNone(inst)

    def test_export_all(self):
        all_instances = self.registry.export_all_instances()
        self.assertGreater(len(all_instances), 20)
        for inst in all_instances:
            self.assertIn("provider", inst)


if __name__ == "__main__":
    unittest.main()
