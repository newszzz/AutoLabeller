from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CONFIG_DIR = Path("config")


class ExportObjectClassConfig(BaseModel):
    id: int | None = Field(default=None, ge=0)
    name: str
    description: str


class ExportDatasetConfig(BaseModel):
    images_dir: Path
    labels_dir: Path
    classes: list[ExportObjectClassConfig]
    image_extensions: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    output_dir: Path
    max_few_shots: int = Field(default=5, ge=0, le=5)
    synthesize_issues: bool = True
    negative_sample_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    dataset_name: str = "autolabeller_annotate"
    image_path_mode: Literal["relative", "absolute"] = "relative"
    random_seed: int = 0

    @model_validator(mode="after")
    def fill_missing_class_ids(self) -> "ExportDatasetConfig":
        for index, item in enumerate(self.classes):
            if item.id is None:
                item.id = index
        return self


class ExportYamlConfig(BaseModel):
    dataset: ExportDatasetConfig


@dataclass(frozen=True)
class LoadedExportConfig:
    path: Path
    dataset: ExportDatasetConfig


def load_export_config(config_name: str | Path) -> LoadedExportConfig:
    config_path = _resolve_export_config_path(config_name)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    export_config = ExportYamlConfig.model_validate(payload)
    base_dir = config_path.parent
    dataset = export_config.dataset.model_copy(
        update={
            "images_dir": _resolve_path(export_config.dataset.images_dir, base_dir),
            "labels_dir": _resolve_path(export_config.dataset.labels_dir, base_dir),
            "output_dir": _resolve_path(export_config.dataset.output_dir, base_dir),
        }
    )
    return LoadedExportConfig(path=config_path, dataset=dataset)


def _resolve_export_config_path(config_name: str | Path) -> Path:
    candidate = Path(config_name)
    candidates: list[Path] = []
    if candidate.is_absolute() or candidate.parent != Path("."):
        candidates.append(candidate)
        if candidate.suffix == "":
            candidates.append(candidate.with_suffix(".yaml"))
            candidates.append(candidate.with_suffix(".yml"))
    else:
        candidates.append(CONFIG_DIR / candidate)
        if candidate.suffix == "":
            candidates.append(CONFIG_DIR / f"{candidate}.yaml")
            candidates.append(CONFIG_DIR / f"{candidate}.yml")

    for path in candidates:
        if path.exists():
            return path.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Export config not found. Searched: {searched}")


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
