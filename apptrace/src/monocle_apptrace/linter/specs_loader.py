import requests
from typing import Dict, Any, Optional


class SpecsLoader:
    """Load and cache Monocle validation specifications from monocle-specs repository.

    This class is responsible for fetching validation specifications from the public
    monocle-specs GitHub repository and caching them in memory for performance.

    The specs define the metamodel for Monocle traces, including entity types, required
    attributes, and validation rules.

    Example:
        >>> specs = SpecsLoader.load_specs()
        >>> print(specs.keys())
        dict_keys(['entities', 'inference_types', 'model_types', ...])
    """

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
        """Load validation specs from cache or fetch from GitHub.

        On first call, fetches all required specification files from the monocle-specs
        repository and caches them in memory. Subsequent calls return the cached specs
        immediately without network requests.

        Returns:
            Dict[str, Any]: Dictionary mapping spec name to spec content.
                Example: {'entities': {...}, 'inference_types': {...}, ...}

        Example:
            >>> specs = SpecsLoader.load_specs()
            >>> entities = specs['entities']
        """

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
    def clear_cache() -> None:
        """Clear the in-memory specs cache.

        Useful for testing when you need to reload specs or when you want to
        free memory. The next call to load_specs() will fetch from GitHub again.

        Example:
            >>> SpecsLoader.clear_cache()
            >>> specs = SpecsLoader.load_specs()  # Fetches from GitHub again
        """
        SpecsLoader._cache = None
