import requests
from typing import Dict, Any, Optional

class SpecsLoader:
    SPECS_BASE_URL = "https://raw.githubusercontent.com/monocle2ai/monocle-specs/main/metamodel/entities"
    REQUIRED_SPEC_FILES = [
        "entities.json",
        "inference_types.json",
        "model_types.json",
        "vector_store_types.json",
        "app_hosting_types.json",
        "workflow_types.json"
    ]

    _cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def load_specs() -> Dict[str, Any]:
        """Load specs from cache or fetch from GitHub"""

        # Return cached specs if available
        if SpecsLoader._cache is not None:
            return SpecsLoader._cache

        # Fetch specs from GitHub
        specs = {}
        for filename in SpecsLoader.REQUIRED_SPEC_FILES:
            url = f"{SpecsLoader.SPECS_BASE_URL}/{filename}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                specs[filename.replace(".json", "")] = response.json()
            except requests.RequestException as e:
                print(f"Warning: Could not load {filename}: {e}")

        # Cache specs for subsequent calls
        SpecsLoader._cache = specs
        return specs

    @staticmethod
    def clear_cache():
        """Clear cache (useful for testing)"""
        SpecsLoader._cache = None
