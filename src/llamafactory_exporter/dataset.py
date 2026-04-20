from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from autolabeller.config import ObjectClassConfig as AutoLabellerClassConfig
from autolabeller.dataset import annotation_result_to_llm_result, build_annotation_summary
from autolabeller.prompts import (
    build_annotation_system_prompt,
    build_annotation_user_prompt,
    build_review_system_prompt,
    build_review_user_prompt,
)
from autolabeller.schemas import AnnotationResult, BoundingBox, LlmAnnotationResult, ReviewPayload

from .config import ExporterConfig, ObjectClassConfig as ExporterClassConfig


def export_dataset(config: ExporterConfig) -> dict[str, Any]:
    output_dir = config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.review_generation.mode != "ground_truth":
        raise ValueError(f"Unsupported review_generation.mode: {config.review_generation.mode}")

    rng = random.Random(config.review_generation.random_seed)
    allowed_extensions = {ext.lower() for ext in config.dataset.image_extensions}
    classes = [_convert_class(item) for item in config.dataset.classes]
    class_names = [item.name for item in classes]
    records: list[dict[str, Any]] = []
    annotation_record_count = 0
    review_record_count = 0
    negative_review_record_count = 0

    image_paths = sorted(
        path
        for path in config.dataset.images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )

    for image_path in image_paths:
        label_path = config.dataset.labels_dir / image_path.relative_to(config.dataset.images_dir).with_suffix(".txt")
        if not label_path.exists():
            continue

        final_annotation = _load_annotation_result(label_path, class_names)
        image_ref = _format_image_path(image_path, output_dir, config.output.image_path_mode)

        records.append(_build_annotation_record(classes, final_annotation, image_ref))
        annotation_record_count += 1

        if config.review_generation.include_positive_sample:
            positive_vlm_result = annotation_result_to_llm_result(final_annotation)
            records.append(
                _build_review_record(
                    classes=classes,
                    final_annotation=final_annotation,
                    image_ref=image_ref,
                    yolo_result=_build_positive_review_source_annotation(final_annotation),
                    vlm_result=positive_vlm_result,
                    review_payload=ReviewPayload(
                        final_objects=positive_vlm_result.objects,
                        has_issues=False,
                        issue_summary="No obvious issues were found in the reviewer inputs.",
                    ),
                    sample_type="review_positive",
                )
            )
            review_record_count += 1

        negative_variants = _build_negative_review_variants(
            final_annotation=final_annotation,
            rng=rng,
            count=config.review_generation.negative_samples_per_image,
        )
        for yolo_result, vlm_result, review_payload in negative_variants:
            records.append(
                _build_review_record(
                    classes=classes,
                    final_annotation=final_annotation,
                    image_ref=image_ref,
                    yolo_result=yolo_result,
                    vlm_result=vlm_result,
                    review_payload=review_payload,
                    sample_type="review_negative",
                )
            )
            review_record_count += 1
            negative_review_record_count += 1

    data_path = output_dir / config.output.data_file
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        config.output.dataset_name: {
            "file_name": config.output.data_file.as_posix(),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    dataset_info_path = output_dir / config.output.dataset_info_file
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "dataset_name": config.output.dataset_name,
        "exported_records": len(records),
        "annotation_records": annotation_record_count,
        "review_records": review_record_count,
        "negative_review_records": negative_review_record_count,
        "dataset_path": str(data_path),
        "dataset_info_path": str(dataset_info_path),
    }


def _build_annotation_record(
    classes: list[AutoLabellerClassConfig],
    final_annotation: AnnotationResult,
    image_ref: str,
) -> dict[str, Any]:
    assistant_payload = annotation_result_to_llm_result(final_annotation)
    return {
        "messages": [
            {
                "role": "system",
                "content": build_annotation_system_prompt(),
            },
            {
                "role": "user",
                "content": f"<image>\n{build_annotation_user_prompt(classes, few_shot_count=0)}",
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload.model_dump(), ensure_ascii=False),
            },
        ],
        "images": [image_ref],
        "task": "annotate",
    }


def _build_review_record(
    classes: list[AutoLabellerClassConfig],
    final_annotation: AnnotationResult,
    image_ref: str,
    yolo_result: AnnotationResult,
    vlm_result: LlmAnnotationResult,
    review_payload: ReviewPayload,
    sample_type: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": build_review_system_prompt(),
            },
            {
                "role": "user",
                "content": f"<image>\n{build_review_user_prompt(classes, yolo_result, vlm_result)}",
            },
            {
                "role": "assistant",
                "content": json.dumps(review_payload.model_dump(), ensure_ascii=False),
            },
        ],
        "images": [image_ref],
        "task": "review",
        "sample_type": sample_type,
        "has_issues": review_payload.has_issues,
        "reference_object_count": len(final_annotation.boxes),
    }


def _build_positive_review_source_annotation(final_annotation: AnnotationResult) -> AnnotationResult:
    return AnnotationResult(
        boxes=[_clone_box(box, confidence=1.0) for box in final_annotation.boxes],
        summary=final_annotation.summary,
        issues=[],
    )


def _build_negative_review_variants(
    final_annotation: AnnotationResult,
    rng: random.Random,
    count: int,
) -> list[tuple[AnnotationResult, LlmAnnotationResult, ReviewPayload]]:
    if count <= 0 or not final_annotation.boxes:
        return []

    base_vlm = annotation_result_to_llm_result(final_annotation)
    variants: list[tuple[AnnotationResult, LlmAnnotationResult, ReviewPayload]] = []
    issue_builders = [_build_missing_box_variant, _build_extra_box_variant]

    for _ in range(count):
        builder = rng.choice(issue_builders)
        yolo_result, vlm_result, issue_summary = builder(final_annotation, base_vlm, rng)
        variants.append(
            (
                yolo_result,
                vlm_result,
                ReviewPayload(
                    final_objects=base_vlm.objects,
                    has_issues=True,
                    issue_summary=issue_summary,
                ),
            )
        )
    return variants


def _build_missing_box_variant(
    final_annotation: AnnotationResult,
    base_vlm: LlmAnnotationResult,
    rng: random.Random,
) -> tuple[AnnotationResult, LlmAnnotationResult, str]:
    missing_index = rng.randrange(len(final_annotation.boxes))
    missing_box = final_annotation.boxes[missing_index]
    target = rng.choice(["yolo", "vlm"])

    if target == "yolo":
        kept_boxes = [box for idx, box in enumerate(final_annotation.boxes) if idx != missing_index]
        yolo_result = AnnotationResult(
            boxes=[_clone_box(box, confidence=1.0) for box in kept_boxes],
            summary=build_annotation_summary(kept_boxes),
            issues=[],
        )
        return (
            yolo_result,
            base_vlm,
            f"YOLO proposal is missing one {missing_box.label} box that appears in the image.",
        )

    kept_objects = [box for idx, box in enumerate(base_vlm.objects) if idx != missing_index]
    kept_summary = build_annotation_summary([_bounding_box_from_llm_box(box) for box in kept_objects])
    vlm_result = base_vlm.model_copy(update={"objects": kept_objects, "summary": kept_summary})
    return (
        _build_positive_review_source_annotation(final_annotation),
        vlm_result,
        f"Multimodal proposal is missing one {missing_box.label} box that appears in the image.",
    )


def _build_extra_box_variant(
    final_annotation: AnnotationResult,
    base_vlm: LlmAnnotationResult,
    rng: random.Random,
) -> tuple[AnnotationResult, LlmAnnotationResult, str]:
    extra_source_box = rng.choice(final_annotation.boxes)
    extra_box = _build_extra_box(extra_source_box)
    target = rng.choice(["yolo", "vlm"])

    if target == "yolo":
        extra_boxes = [_clone_box(box, confidence=1.0) for box in final_annotation.boxes]
        extra_boxes.append(_clone_box(extra_box, confidence=0.65))
        return (
            AnnotationResult(
                boxes=extra_boxes,
                summary=build_annotation_summary(extra_boxes),
                issues=[],
            ),
            base_vlm,
            f"YOLO proposal contains an extra {extra_box.label} box that should be removed.",
        )

    extra_objects = [box.model_copy(deep=True) for box in base_vlm.objects]
    extra_objects.append(extra_box.model_copy())
    extra_summary = build_annotation_summary([_bounding_box_from_llm_box(box) for box in extra_objects])
    vlm_result = base_vlm.model_copy(
        update={
            "objects": extra_objects,
            "summary": extra_summary,
        }
    )
    return (
        _build_positive_review_source_annotation(final_annotation),
        vlm_result,
        f"Multimodal proposal contains an extra {extra_box.label} box that should be removed.",
    )


def _build_extra_box(source_box: BoundingBox) -> BoundingBox:
    return BoundingBox(
        label=source_box.label,
        x_center=min(0.95, source_box.x_center + min(0.08, source_box.width * 0.5)),
        y_center=min(0.95, source_box.y_center + min(0.08, source_box.height * 0.5)),
        width=source_box.width,
        height=source_box.height,
    )


def _clone_box(box: BoundingBox, confidence: float | None = None) -> BoundingBox:
    return BoundingBox(
        label=box.label,
        x_center=box.x_center,
        y_center=box.y_center,
        width=box.width,
        height=box.height,
        confidence=confidence if confidence is not None else box.confidence,
    )


def _bounding_box_from_llm_box(box) -> BoundingBox:
    return BoundingBox(
        label=box.label,
        x_center=box.x_center,
        y_center=box.y_center,
        width=box.width,
        height=box.height,
    )


def _load_annotation_result(label_path: Path, class_names: list[str]) -> AnnotationResult:
    boxes: list[BoundingBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"Unknown class id {class_id} found in {label_path}.")
        x_center, y_center, width, height = map(float, parts[1:5])
        boxes.append(
            BoundingBox(
                label=class_names[class_id],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return AnnotationResult(
        boxes=boxes,
        summary=build_annotation_summary(boxes),
        issues=[],
    )


def _convert_class(item: ExporterClassConfig) -> AutoLabellerClassConfig:
    return AutoLabellerClassConfig(name=item.name, description=item.description)


def _format_image_path(image_path: Path, output_dir: Path, mode: str) -> str:
    if mode == "absolute":
        return str(image_path)
    return os.path.relpath(image_path, start=output_dir)
