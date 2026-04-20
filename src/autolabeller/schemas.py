from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    label: str = Field(
        description="Target class name. It must exactly match one name defined in Class definitions."
    )
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
        description="Optional confidence score, mainly used for YOLO proposals.",
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
    boxes: list[BoundingBox] = Field(default_factory=list, description="All boxes for one annotation result.")
    summary: str = Field(default="", description="Short summary of the annotation result.")
    issues: list[str] = Field(
        default_factory=list,
        description="Potential issues or uncertainties found during annotation.",
    )


class ReviewResult(BaseModel):
    final_boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="Final reviewed boxes after merging YOLO and multimodal results.",
    )
    has_issues: bool = Field(
        default=False,
        description="Whether the reviewer found missing boxes, extra boxes, or other problems in the candidate inputs.",
    )
    issue_summary: str = Field(
        default="",
        description="Short summary of the review result and any issues found in the candidate inputs.",
    )


class ImageRecord(BaseModel):
    image_path: Path
    width: int
    height: int


class LlmBox(BaseModel):
    label: str = Field(
        description="Target class name. It must exactly match one name defined in Class definitions."
    )
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

    @model_validator(mode="after")
    def validate_bbox(self) -> "LlmBox":
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        return self

    def as_list(self) -> list[float]:
        return [self.x_center, self.y_center, self.width, self.height]


class LlmAnnotationResult(BaseModel):
    objects: list[LlmBox] = Field(
        default_factory=list,
        description="All detected objects that belong to configured classes.",
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
    has_issues: bool = Field(
        default=False,
        description="Whether the reviewer found missing boxes, extra boxes, or other problems in the candidate inputs.",
    )
    issue_summary: str = Field(
        default="",
        description="Short summary describing detected issues, or stating that no obvious issues were found.",
    )
