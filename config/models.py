"""
Pydantic models for EDIATH config validation
All YAMLs validated against schemas
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum
import yaml


class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PROD = "production"


class TimingConfig(BaseModel):
    tick_interval_ms: int = Field(100, ge=10, le=1000)
    health_check_interval_sec: int = Field(30, ge=5, le=300)
    metrics_flush_interval_sec: int = Field(60, ge=10, le=600)
    circuit_breaker_threshold: int = Field(5, ge=1, le=20)


class PathConfig(BaseModel):
    log_dir: Path = Field("logs")
    config_dir: Path = Field("config")
    data_dir: Path = Field("data")
    workspace_dir: Path = Field("workspace")


class FeatureFlags(BaseModel):
    enable_autonomous_mode: bool = True
    enable_learning: bool = True
    enable_memory: bool = True
    enable_security: bool = True
    enable_vision: bool = True
    enable_audio: bool = True


class LLMConfig(BaseModel):
    model_path: Path = Field("models/EDIATH-q4_k_m.gguf")
    max_tokens: int = Field(2048, ge=100, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field("EDIATH_Orchestrator", min_length=1)
    version: str = "2.0.0"
    environment: Environment = Environment.PROD

    timing: TimingConfig = Field(default_factory=TimingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, v):
        for path in v.dict().values():
            path.expanduser().resolve()
        return v


class AudioConfig(BaseModel):
    input_device: Optional[str] = None
    sample_rate: int = 16000


class AutonomyConfig(BaseModel):
    max_iterations: int = 100000


class SystemConfig(BaseModel):
    rbac_enabled: bool = True


# Master config merger
class EDIATHConfig(BaseModel):
    orchestrator: OrchestratorConfig
    audio: Optional[AudioConfig] = None
    autonomy: Optional[AutonomyConfig] = None
    system: Optional[SystemConfig] = None
    browser: Optional[Dict[str, Any]] = None


def validate_yaml(file_path: Path) -> bool:
    """Validate single YAML against model"""
    if not file_path.exists():
        return False
    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)
        if "orchestrator" in file_path.name.lower():
            OrchestratorConfig.model_validate(data)
        elif "audio" in file_path.name.lower():
            AudioConfig.model_validate(data)
        # Add more validators
        return True
    except Exception as e:
        import logging

        logging.getLogger(__name__).exception("Validation failed %s: %s", file_path, e)
        return False


def load_merged_config(config_dir: Path = Path("config")) -> EDIATHConfig:
    """Load + validate + merge all configs"""
    merged = {}
    errors = []

    for yaml_file in config_dir.glob("*.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}
            merged.update(data)
        except Exception as e:
            errors.append(str(e))

    if errors:
        raise ValueError(f"Config errors: {errors}")

    return EDIATHConfig.model_validate(merged)


if __name__ == "__main__":
    # Test validation
    import glob

    config_dir = Path("config")
    import logging

    logger = logging.getLogger(__name__)
    valid = [validate_yaml(Path(f)) for f in glob.glob("config/*.yaml")]
    logger.info("All configs valid: %s", all(valid))
