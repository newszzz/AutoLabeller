from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ObjectClassConfig(BaseModel):
    name: str
    description: str


class YoloDatasetConfig(BaseModel):
    images_dir: Path
    labels_dir: Path
    classes: list[ObjectClassConfig]
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])


class ReviewGenerationConfig(BaseModel):
    mode: Literal["ground_truth"] = "ground_truth"
    include_positive_sample: bool = True
    negative_samples_per_image: int = Field(default=2, ge=0)
    random_seed: int = 42

    @field_validator("negative_samples_per_image")
    @classmethod
    def validate_negative_samples_per_image(cls, value: int) -> int:
        if value < 0:
            raise ValueError("negative_samples_per_image must be >= 0.")
        return value


class OutputConfig(BaseModel):
    output_dir: Path = Path("outputs/llamafactory_dataset")
    dataset_name: str = "yolo_multitask_sft"
    data_file: Path = Path("dataset.json")
    dataset_info_file: Path = Path("dataset_info.json")
    image_path_mode: Literal["relative", "absolute"] = "relative"


class ExporterConfig(BaseModel):
    dataset: YoloDatasetConfig
    review_generation: ReviewGenerationConfig = Field(default_factory=ReviewGenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: str | Path) -> ExporterConfig:
    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ExporterConfig.model_validate(data)
    base_dir = config_path.parent

    def resolve(path: Path) -> Path:
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()

    config.dataset.images_dir = resolve(config.dataset.images_dir)  # type: ignore[assignment]
    config.dataset.labels_dir = resolve(config.dataset.labels_dir)  # type: ignore[assignment]
    config.output.output_dir = resolve(config.output.output_dir)  # type: ignore[assignment]
    return config
