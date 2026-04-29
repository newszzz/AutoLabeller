from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    label: str = Field(
        description="Target class name. It must exactly match one configured class name."
    )
    x_min: float = Field(ge=0.0, description="Left edge in absolute image pixels.")
    y_min: float = Field(ge=0.0, description="Top edge in absolute image pixels.")
    x_max: float = Field(ge=0.0, description="Right edge in absolute image pixels.")
    y_max: float = Field(ge=0.0, description="Bottom edge in absolute image pixels.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional detector confidence, mainly used for YOLO proposals.",
    )

    @model_validator(mode="after")
    def validate_corners(self) -> "BoundingBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("x_max/y_max must be greater than x_min/y_min.")
        return self

    def without_confidence(self) -> "LlmBox":
        return LlmBox(
            label=self.label,
            x_min=self.x_min,
            y_min=self.y_min,
            x_max=self.x_max,
            y_max=self.y_max,
        )


class AnnotationResult(BaseModel):
    objects: list[BoundingBox] = Field(
        default_factory=list,
        description="All boxes for one annotation result.",
    )


class ImageRecord(BaseModel):
    image_path: Path
    width: int
    height: int


class LlmBox(BaseModel):
    label: str = Field(
        description="Target class name. It must exactly match one configured class name."
    )
    x_min: float = Field(ge=0.0, description="Left edge in absolute image pixels.")
    y_min: float = Field(ge=0.0, description="Top edge in absolute image pixels.")
    x_max: float = Field(ge=0.0, description="Right edge in absolute image pixels.")
    y_max: float = Field(ge=0.0, description="Bottom edge in absolute image pixels.")

    @model_validator(mode="after")
    def validate_corners(self) -> "LlmBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("x_max/y_max must be greater than x_min/y_min.")
        return self


class LlmAnnotationResult(BaseModel):
    objects: list[LlmBox] = Field(
        default_factory=list,
        description="Corrected objects that belong to configured classes.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Annotation changes made relative to YOLO, such as removed boxes, added boxes, fixed labels, or adjusted coordinates.",
    )


class ReviewResult(BaseModel):
    yolo_is_correct: bool = Field(
        description="Whether the YOLO annotation has no missing, extra, duplicate, or mislabeled boxes."
    )
    llm_is_correct: bool = Field(
        description="Whether the multimodal annotation has no missing, extra, duplicate, or mislabeled boxes."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems found in either annotation result.",
    )
