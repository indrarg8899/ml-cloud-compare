"""
Provider Pricing Data & Registry
Contains instance configurations, pricing, and availability for all supported providers.
Data sourced from official pricing pages (2025).
"""

from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class ProviderInfo:
    """Metadata about a cloud provider."""
    name: str
    display_name: str
    website: str
    gpu_brands: list
    regions: list
    spot_available: bool = True
    reserved_available: bool = True


class Provider:
    """Represents a single cloud provider with its instances."""

    def __init__(self, info: ProviderInfo):
        self.info = info
        self.instances: List[dict] = []

    def add_instance(self, name: str, gpu_type: str, gpu_count: int,
                     vcpus: int, ram_gb: int, hourly_cost: float,
                     spot_hourly_rate: Optional[float] = None,
                     category: str = "gpu"):
        """Add an instance type."""
        self.instances.append({
            "name": name,
            "gpu_type": gpu_type.lower(),
            "gpu_count": gpu_count,
            "vcpus": vcpus,
            "ram_gb": ram_gb,
            "hourly_cost": hourly_cost,
            "spot_hourly_rate": spot_hourly_rate,
            "category": category,
        })

    def find_instance(self, gpu_type: str, min_gpus: int = 1) -> Optional[dict]:
        """Find best matching instance for GPU type and count."""
        gpu_type = gpu_type.lower()
        matches = [
            inst for inst in self.instances
            if inst["gpu_type"] == gpu_type and inst["gpu_count"] >= min_gpus
        ]
        if not matches:
            matches = [
                inst for inst in self.instances
                if gpu_type in inst["gpu_type"] and inst["gpu_count"] >= min_gpus
            ]
        if not matches:
            matches = [
                inst for inst in self.instances
                if inst["gpu_count"] >= min_gpus
            ]
        if not matches:
            return None
        return min(matches, key=lambda i: i["hourly_cost"])

    def list_gpus(self) -> list:
        """List all available GPU types."""
        return list(set(inst["gpu_type"] for inst in self.instances))


class ProviderRegistry:
    """Registry of all cloud providers and their pricing."""

    def __init__(self):
        self._providers: Dict[str, Provider] = {}
        self._load_default_pricing()

    def register(self, name: str, provider: Provider):
        """Register a provider."""
        self._providers[name.lower()] = provider

    def get(self, name: str) -> Optional[Provider]:
        """Get a provider by name."""
        return self._providers.get(name.lower())

    def list_providers(self) -> list:
        """List all registered provider names."""
        return list(self._providers.keys())

    def _load_default_pricing(self):
        """Load default pricing data for all providers."""

        # ── AWS (us-east-1) ──
        aws = Provider(ProviderInfo(
            name="aws",
            display_name="Amazon Web Services",
            website="https://aws.amazon.com/ec2/pricing/",
            gpu_brands=["nvidia"],
            regions=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
        ))
        aws.add_instance("p5.48xlarge", "h100", 8, 192, 2048, 98.32, spot_hourly_rate=49.16)
        aws.add_instance("p4d.24xlarge", "a100", 8, 96, 1152, 32.77, spot_hourly_rate=16.39)
        aws.add_instance("p5e.48xlarge", "h200", 8, 192, 2048, 120.00, spot_hourly_rate=60.00)
        aws.add_instance("g6.12xlarge", "l4", 4, 48, 192, 9.52, spot_hourly_rate=4.76)
        aws.add_instance("g6.xlarge", "t4", 1, 4, 16, 1.10, spot_hourly_rate=0.55)
        aws.add_instance("g5.12xlarge", "a10g", 4, 48, 192, 16.28, spot_hourly_rate=8.14)
        aws.add_instance("g5.24xlarge", "a10g", 8, 96, 384, 32.56, spot_hourly_rate=16.28)
        aws.add_instance("p4de.24xlarge", "a100", 8, 96, 1536, 40.97, spot_hourly_rate=20.49)
        self.register("aws", aws)

        # ── GCP (us-central1) ──
        gcp = Provider(ProviderInfo(
            name="gcp",
            display_name="Google Cloud Platform",
            website="https://cloud.google.com/compute/gpus-pricing",
            gpu_brands=["nvidia"],
            regions=["us-central1", "us-west1", "europe-west1", "asia-east1"],
        ))
        gcp.add_instance("a3-highgpu-8g", "h100", 8, 208, 1872, 94.13, spot_hourly_rate=28.24)
        gcp.add_instance("a2-highgpu-8g", "a100", 8, 96, 1360, 29.39, spot_hourly_rate=8.82)
        gcp.add_instance("a3-highgpu-8g-megagpu", "h100", 8, 208, 1872, 98.31, spot_hourly_rate=29.49)
        gcp.add_instance("g2-standard-48", "l4", 4, 48, 192, 8.84, spot_hourly_rate=2.65)
        gcp.add_instance("n1-standard-8-t4", "t4", 1, 8, 30, 1.05, spot_hourly_rate=0.32)
        gcp.add_instance("g2-standard-24", "l4", 2, 24, 96, 4.42, spot_hourly_rate=1.33)
        gcp.add_instance("a2-highgpu-4g", "a100", 4, 48, 680, 14.69, spot_hourly_rate=4.41)
        self.register("gcp", gcp)

        # ── Azure (East US) ──
        azure = Provider(ProviderInfo(
            name="azure",
            display_name="Microsoft Azure",
            website="https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/",
            gpu_brands=["nvidia"],
            regions=["eastus", "westus2", "westeurope", "southeastasia"],
        ))
        azure.add_instance("Standard_ND96isr_H100_v5", "h100", 8, 96, 1900, 100.20, spot_hourly_rate=40.08)
        azure.add_instance("Standard_ND96asr_v4", "a100", 8, 96, 900, 27.19, spot_hourly_rate=10.88)
        azure.add_instance("Standard_NC24ads_A100_v4", "a100", 1, 24, 220, 3.67, spot_hourly_rate=1.47)
        azure.add_instance("Standard_NC4as_T4_v3", "t4", 1, 4, 28, 1.14, spot_hourly_rate=0.46)
        azure.add_instance("Standard_NC8as_T4_v3", "t4", 2, 8, 56, 2.28, spot_hourly_rate=0.91)
        azure.add_instance("Standard_NV36ads_A10_v5", "a10", 1, 36, 440, 3.67, spot_hourly_rate=1.47)
        azure.add_instance("Standard_NC96isr_H100_v5", "h100", 8, 96, 1900, 98.32, spot_hourly_rate=39.33)
        self.register("azure", azure)

        # ── DigitalOcean ──
        do = Provider(ProviderInfo(
            name="digitalocean",
            display_name="DigitalOcean",
            website="https://www.digitalocean.com/pricing/gpu-droplets",
            gpu_brands=["nvidia"],
            regions=["nyc1", "sfo3", "ams3"],
            spot_available=False,
        ))
        do.add_instance("gpu-h100-1x", "h100", 1, 24, 320, 5.39)
        do.add_instance("gpu-h100-8x", "h100", 8, 192, 2560, 43.10)
        do.add_instance("gpu-a100-1x", "a100", 1, 24, 320, 2.89)
        do.add_instance("gpu-a100-8x", "a100", 8, 192, 2560, 23.10)
        do.add_instance("gpu-a10-1x", "a10", 1, 24, 320, 1.10)
        do.add_instance("gpu-rtx6000-1x", "rtx6000", 1, 12, 64, 0.76)
        do.add_instance("gpu-rtx6000-4x", "rtx6000", 4, 48, 256, 3.04)
        self.register("digitalocean", do)

        # ── Akamai Cloud (Linode) ──
        akamai = Provider(ProviderInfo(
            name="akamai",
            display_name="Akamai Cloud Computing",
            website="https://www.linode.com/pricing/",
            gpu_brands=["nvidia"],
            regions=["us-east", "us-west", "eu-west", "ap-south"],
            spot_available=False,
        ))
        akamai.add_instance("GPU-f48000-x1", "h100", 1, 24, 512, 4.99)
        akamai.add_instance("GPU-f48000-x8", "h100", 8, 192, 4096, 39.90)
        akamai.add_instance("GPU-a100-xl", "a100", 1, 24, 320, 2.49)
        akamai.add_instance("GPU-a100-8xl", "a100", 8, 192, 2560, 19.90)
        akamai.add_instance("GPU-rtx6000-x1", "rtx6000", 1, 12, 64, 0.65)
        akamai.add_instance("GPU-l4-x1", "l4", 1, 24, 96, 1.10)
        akamai.add_instance("GPU-t4-x1", "t4", 1, 4, 16, 0.50)
        self.register("akamai", akamai)

    def export_all_instances(self) -> list:
        """Export all instances from all providers."""
        all_instances = []
        for name, provider in self._providers.items():
            for inst in provider.instances:
                all_instances.append({**inst, "provider": name})
        return all_instances
