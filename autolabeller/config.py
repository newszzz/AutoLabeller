from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class ObjectClassConfig(BaseModel):
    id: int | None = Field(default=None, ge=0, description="Class id used by the YOLO model output.")
    name: str = Field(description="Class name used in YOLO labels and LLM outputs.")
    description: str = Field(description="Natural-language description of this class.")


class FewShotExampleConfig(BaseModel):
    image_path: Path = Field(description="Original few-shot image.")
    annotation_path: Path = Field(
        validation_alias=AliasChoices("annotation_path", "target_label_path", "label_path"),
        description="Few-shot annotation JSON using the same schema as model outputs.",
    )


class DatasetConfig(BaseModel):
    images_dir: Path
    classes: list[ObjectClassConfig] = Field(default_factory=list)
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    output_dir: Path = Path("outputs")
    few_shots: list[FewShotExampleConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_missing_class_ids(self) -> "DatasetConfig":
        for index, item in enumerate(self.classes):
            if item.id is None:
                item.id = index
        return self


class YoloConfig(BaseModel):
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


class ModelApiConfig(BaseModel):
    backend: Literal["vllm", "ollama"] = "vllm"
    base_url: str | None = None
    api_key: str = "0"
    model: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    request_timeout: float = Field(default=120.0, gt=0.0)
    annotator_few_shots: list[FewShotExampleConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_model(self) -> "ModelApiConfig":
        if not self.model:
            raise ValueError("llm_api.model is required.")
        if not self.base_url:
            self.base_url = (
                "http://localhost:11434/v1"
                if self.backend == "ollama"
                else "http://localhost:8000/v1"
            )
        return self

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized


class AppConfig(BaseModel):
    dataset: DatasetConfig
    yolo: YoloConfig
    llm_api: ModelApiConfig = Field(
        validation_alias=AliasChoices("llm_api", "vllm", "ollama")
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_api_section(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "llm_api" not in data:
            for key, backend in (("vllm", "vllm"), ("ollama", "ollama")):
                if key in data:
                    data["llm_api"] = data[key]
                    if isinstance(data["llm_api"], dict):
                        data["llm_api"].setdefault("backend", backend)
                    break
        return data

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: DatasetConfig) -> DatasetConfig:
        if not value.classes:
            raise ValueError("dataset.classes must be provided.")
        class_ids = [item.id for item in value.classes]
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("dataset.classes.id values must be unique.")
        return value

    @model_validator(mode="after")
    def merge_few_shots(self) -> "AppConfig":
        if self.dataset.few_shots and not self.llm_api.annotator_few_shots:
            self.llm_api.annotator_few_shots = self.dataset.few_shots
        return self


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
    config.dataset.few_shots = _resolve_few_shots(config.dataset.few_shots, resolve)  # type: ignore[assignment]
    config.llm_api.annotator_few_shots = _resolve_few_shots(
        config.llm_api.annotator_few_shots,
        resolve,
    )  # type: ignore[assignment]
    return config


def _resolve_few_shots(
    examples: list[FewShotExampleConfig],
    resolve: Callable[[Path | None], Path | None],
) -> list[FewShotExampleConfig]:
    return [
        FewShotExampleConfig(
            image_path=resolve(item.image_path),
            annotation_path=resolve(item.annotation_path),
        )
        for item in examples
    ]
