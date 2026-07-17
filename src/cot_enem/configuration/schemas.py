"""Pydantic schemas for portable execution settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
PrecisionName = Literal["auto", "fp32", "fp16", "bf16"]
EnvironmentName = Literal["auto", "windows", "linux", "wsl", "colab", "server", "slurm"]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(ConfigModel):
    name: str = "cot-enem"
    seed: int = 42


class RuntimeConfig(ConfigModel):
    environment: EnvironmentName = "auto"
    device: DeviceName = "auto"
    precision: PrecisionName = "auto"
    num_workers: int = Field(default=2, ge=0)


class ModelConfig(ConfigModel):
    provider: Literal["openai_compatible", "huggingface", "mock"] = "openai_compatible"
    name: str = "qwen2.5:1.5b"
    judge_name: str | None = None
    batch_size: int = Field(default=1, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_seconds: float = Field(default=60, gt=0)
    max_attempts: int = Field(default=3, ge=1)


class PipelineConfig(ConfigModel):
    stages: list[Literal["specify", "complicate", "diversify"]] = Field(
        default_factory=lambda: ["specify"]
    )
    checkpoint_interval: int = Field(default=25, ge=1)
    resume: bool = True
    fail_fast: bool = False
    limit: int | None = Field(default=None, ge=1)

    @field_validator("stages")
    @classmethod
    def stages_cannot_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("pipeline.stages must contain at least one strategy")
        return value


class StorageConfig(ConfigModel):
    backend: Literal["local", "drive"] = "local"
    base_path: str = "outputs"
    persistent_path: str | None = None
    cache_path: str | None = None


class LoggingConfig(ConfigModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_to_file: bool = True
    structured: bool = True


class ApplicationConfig(ConfigModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
