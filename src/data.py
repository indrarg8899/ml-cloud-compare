"""Data module: loads GPU specs and pricing from JSON."""
import json
from pathlib import Path
from .compare import GPUSpec

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_pricing() -> dict:
    path = _DATA_DIR / "defaults.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _load_gpu_catalog() -> dict:
    from .compare import GPUSpec
    # Import the catalog from compare module defaults
    from .compare import GPU_CATALOG
    return GPU_CATALOG
