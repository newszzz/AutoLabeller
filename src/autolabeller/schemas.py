from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


AnnotationSource = Literal["ground_truth", "yolo", "vlm", "reviewer"]


class BoundingBox(BaseModel):
    label: str = Field(description="Target class name. Must exactly match one class defined in config.")
    x_center: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized horizontal center coordinate of the bounding box, from 0 to 1.",
    )
    y_center: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized vertical center coordinate of the bounding box, from 0 to 1.",
    )
    width: float = Field(
        gt=0.0,
        le=1.0,
        description="Normalized box width relative to the full image width.",
    )
    height: float = Field(
        gt=0.0,
        le=1.0,
        description="Normalized box height relative to the full image height.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score for this bounding box.",
    )
    source: AnnotationSource = Field(description="Which stage produced this annotation.")
    rationale: str | None = Field(
        default=None,
        description="Optional short reason explaining why the model kept this box.",
    )

    def to_yolo_line(self, class_to_idx: dict[str, int]) -> str:
        if self.label not in class_to_idx:
            raise KeyError(f"Unknown label {self.label!r} not found in class mapping.")
        class_id = class_to_idx[self.label]
        return (
            f"{class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    def as_list(self) -> list[float]:
        return [self.x_center, self.y_center, self.width, self.height]


class AnnotationResult(BaseModel):
    image_path: Path
    boxes: list[BoundingBox] = Field(default_factory=list, description="All boxes for this image.")
    source: AnnotationSource
    summary: str = Field(default="", description="Short summary of the annotation result.")
    issues: list[str] = Field(
        default_factory=list,
        description="Potential issues or uncertainties found during annotation.",
    )


class ReviewResult(BaseModel):
    image_path: Path
    final_boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="Final reviewed boxes after merging YOLO and VLM results.",
    )
    summary: str = Field(default="", description="Short summary of the final reviewed result.")
    missing_from_yolo: list[str] = Field(
        default_factory=list,
        description="Objects that appear missing from the YOLO proposal.",
    )
    missing_from_vlm: list[str] = Field(
        default_factory=list,
        description="Objects that appear missing from the multimodal model proposal.",
    )
    suspicious_labels: list[str] = Field(
        default_factory=list,
        description="Potential wrong labels, false positives, or uncertain cases.",
    )


class ImageRecord(BaseModel):
    image_path: Path
    width: int
    height: int


class LlmBox(BaseModel):
    """Structured box returned by a multimodal model."""

    label: str = Field(
        min_length=1,
        description=(
            "Target class name. It must exactly match one configured class name and must not include"
            " coordinates, ids, or explanations."
        ),
    )
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Normalized bounding box in YOLO format [x_center, y_center, width, height]."
            " All values must be between 0 and 1."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score for the object.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional short reason for the label or reviewer decision.",
    )

    @model_validator(mode="after")
    def validate_bbox(self) -> "LlmBox":
        for value in self.bbox:
            if not 0.0 <= value <= 1.0:
                raise ValueError("bbox values must be normalized between 0 and 1.")
        if self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError("bbox width and height must be positive.")
        return self


class LlmAnnotationPayload(BaseModel):
    """Structured annotation response returned by the multimodal annotation agent."""

    objects: list[LlmBox] = Field(
        default_factory=list,
        description="All detected objects that belong to configured classes. Return an empty list if none are found.",
    )
    summary: str = Field(
        description="One or two sentences summarizing the image content and what was annotated."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Optional list of uncertain, ambiguous, occluded, or hard-to-judge cases.",
    )


class ReviewPayload(BaseModel):
    final_objects: list[LlmBox] = Field(
        default_factory=list,
        description="Final reviewed object list after comparing YOLO and VLM outputs with the image.",
    )
    summary: str = Field(
        default="",
        description="Short summary of the review result and important corrections.",
    )
    missing_from_yolo: list[str] = Field(
        default_factory=list,
        description="Short descriptions of objects YOLO appears to have missed.",
    )
    missing_from_vlm: list[str] = Field(
        default_factory=list,
        description="Short descriptions of objects the multimodal agent appears to have missed.",
    )
    suspicious_labels: list[str] = Field(
        default_factory=list,
        description="Potential false positives, wrong labels, or uncertain annotations worth attention.",
    )
