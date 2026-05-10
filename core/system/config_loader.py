"""
SAFE Config Loader (NO DEPENDENCY ON MISSING models.py)
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Safe Config Loader (no pydantic dependency)"""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir.resolve()
        self.config_dir.mkdir(exist_ok=True)
        self._config: Dict[str, Any] = {}

    def load_all_configs(self) -> Dict[str, Any]:
        """Load all YAML configs safely"""
        if self._config:
            return self._config

        print("🔄 Loading config/*.yaml...")

        merged = {}

        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    merged[yaml_file.stem] = data
                    print(f"✅ Loaded {yaml_file.name}")
            except Exception as e:
                print(f"❌ Failed {yaml_file.name}: {e}")

        self._config = merged
        print("🎉 Configs loaded successfully")
        return self._config

    def get(self, name: str, default=None):
        """Get specific config"""
        cfg = self.load_all_configs()
        return cfg.get(name, default)

    def get_orchestrator_config(self) -> Dict[str, Any]:
        return self.get("orchestrator", {})


# Global instance
config = ConfigLoader()


if __name__ == "__main__":
    loader = ConfigLoader()
    data = loader.load_all_configs()
    print(data)
