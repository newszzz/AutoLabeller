from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_model_path(self) -> "YoloConfig":
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError("yolo.model_path must point to an ONNX file.")
        return self


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    annotator_model: str
    reviewer_model: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    annotator_few_shots: list[FewShotExampleConfig] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    max_images: int | None = Field(default=None, ge=1)
    save_intermediate_json: bool = True


class FineTuneDatasetConfig(BaseModel):
    images_dir: Path
    labels_dir: Path


class FineTuneConfig(BaseModel):
    dataset: FineTuneDatasetConfig | None = None
    generic_output: Path = Path("outputs/sft_generic.jsonl")
    llamafactory_output: Path = Path("outputs/sft_llamafactory.json")
    task_prompt: str
    use_reviewed_annotations: bool = False


class AppConfig(BaseModel):
    dataset: DatasetConfig
    yolo: YoloConfig
    ollama: OllamaConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    finetune: FineTuneConfig

    @model_validator(mode="after")
    def validate_config(self) -> "AppConfig":
        if not self.dataset.classes:
            raise ValueError("dataset.classes must be provided.")
        if not self.finetune.use_reviewed_annotations and self.finetune.dataset is None:
            return self
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
    config.ollama.annotator_few_shots = [  # type: ignore[assignment]
        FewShotExampleConfig(
            image_path=resolve(item.image_path),
            label_path=resolve(item.label_path),
        )
        for item in config.ollama.annotator_few_shots
    ]
    if config.finetune.dataset is not None:
        config.finetune.dataset.images_dir = resolve(config.finetune.dataset.images_dir)  # type: ignore[assignment]
        config.finetune.dataset.labels_dir = resolve(config.finetune.dataset.labels_dir)  # type: ignore[assignment]
    config.finetune.generic_output = resolve(config.finetune.generic_output)  # type: ignore[assignment]
    config.finetune.llamafactory_output = resolve(config.finetune.llamafactory_output)  # type: ignore[assignment]
    return config
