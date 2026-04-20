from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator


class ObjectClassConfig(BaseModel):
    name: str = Field(description="Class name used in YOLO labels and LLM outputs.")
    description: str = Field(description="Natural-language description of what this class means.")


class FewShotExampleConfig(BaseModel):
    image_path: Path = Field(description="Path to the example image used as a few-shot demonstration.")
    label_path: Path = Field(description="Path to the YOLO-format ground-truth label file for the example image.")


class DatasetConfig(BaseModel):
    images_dir: Path
    classes: list[ObjectClassConfig] = Field(default_factory=list)
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    output_dir: Path = Path("outputs")


class YoloConfig(BaseModel):
    enabled: bool = True
    model_path: Path
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: str | None = None

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: Path) -> Path:
        if value.suffix.lower() != ".onnx":
            raise ValueError("yolo.model_path must point to an ONNX file.")
        return value


class LlamaFactoryConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "0"
    annotator_model: str
    reviewer_model: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    request_timeout: float = Field(default=120.0, gt=0.0)
    annotator_few_shots: list[FewShotExampleConfig] = Field(default_factory=list)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized


class PipelineConfig(BaseModel):
    max_images: int | None = Field(default=None, ge=1)
    save_intermediate_json: bool = True


class AppConfig(BaseModel):
    dataset: DatasetConfig
    yolo: YoloConfig
    llama_factory: LlamaFactoryConfig = Field(
        validation_alias=AliasChoices("llama_factory", "ollama")
    )
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: DatasetConfig) -> DatasetConfig:
        if not value.classes:
            raise ValueError("dataset.classes must be provided.")
        return value


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(data)
    base_dir = config_path.parent

    def resolve(path: Path | None) -> Path | None:
        if path is None or path.is_absolute():
            return path
        return (base_dir / path).resolve()

    config.dataset.images_dir = resolve(config.dataset.images_dir)  # type: ignore[assignment]
    config.dataset.output_dir = resolve(config.dataset.output_dir)  # type: ignore[assignment]
    config.yolo.model_path = resolve(config.yolo.model_path)  # type: ignore[assignment]
    config.llama_factory.annotator_few_shots = [  # type: ignore[assignment]
        FewShotExampleConfig(
            image_path=resolve(item.image_path),
            label_path=resolve(item.label_path),
        )
        for item in config.llama_factory.annotator_few_shots
    ]
    return config
