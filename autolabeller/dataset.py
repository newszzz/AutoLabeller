from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .config import DatasetConfig, ObjectClassConfig
from .schemas import AnnotationResult, BoundingBox, ImageRecord, LlmAnnotationResult, LlmBox


def load_classes(config: DatasetConfig) -> list[ObjectClassConfig]:
    return config.classes


def load_class_names(config: DatasetConfig) -> list[str]:
    return [item.name for item in config.classes]


def build_class_catalog_text(classes: list[ObjectClassConfig]) -> str:
    return "\n".join(f"- {item.name}: {item.description}" for item in classes)


def collect_image_records(config: DatasetConfig) -> list[ImageRecord]:
    exts = {ext.lower() for ext in config.image_extensions}
    image_paths = sorted(
        path
        for path in config.images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    )

    records: list[ImageRecord] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
        records.append(ImageRecord(image_path=image_path, width=width, height=height))
    return records


def validate_annotation_result(
    result: AnnotationResult,
    class_names: list[str],
    image_width: int,
    image_height: int,
) -> AnnotationResult:
    allowed_labels = set(class_names)
    invalid_labels = sorted({item.label for item in result.objects if item.label not in allowed_labels})
    if invalid_labels:
        raise ValueError(
            f"Annotation contains unsupported labels {invalid_labels}. Allowed labels: {class_names}"
        )

    for item in result.objects:
        if item.x_min >= image_width or item.x_max > image_width:
            raise ValueError(f"Box {item.label!r} exceeds image width {image_width}.")
        if item.y_min >= image_height or item.y_max > image_height:
            raise ValueError(f"Box {item.label!r} exceeds image height {image_height}.")
    return result


def validate_llm_annotation_result(
    result: LlmAnnotationResult,
    class_names: list[str],
    image_width: int,
    image_height: int,
) -> LlmAnnotationResult:
    annotation = AnnotationResult(
        objects=[llm_box_to_bounding_box(item) for item in result.objects],
    )
    validate_annotation_result(annotation, class_names, image_width, image_height)
    return result


def llm_box_to_bounding_box(item: LlmBox) -> BoundingBox:
    return BoundingBox(
        label=item.label,
        x_min=item.x_min,
        y_min=item.y_min,
        x_max=item.x_max,
        y_max=item.y_max,
    )


def llm_result_to_annotation(result: LlmAnnotationResult) -> AnnotationResult:
    boxes = [llm_box_to_bounding_box(item) for item in result.objects]
    return AnnotationResult(objects=boxes)


def annotation_result_to_llm_result(result: AnnotationResult) -> LlmAnnotationResult:
    return LlmAnnotationResult(
        objects=[box.without_confidence() for box in result.objects],
        issues=[],
    )


def load_annotation_json(path: Path) -> AnnotationResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnnotationResult.model_validate(data)


def load_annotation_file(
    annotation_path: Path,
    image_path: Path,
    classes: list[ObjectClassConfig],
) -> AnnotationResult:
    if annotation_path.suffix.lower() == ".json":
        return load_annotation_json(annotation_path)
    if annotation_path.suffix.lower() == ".txt":
        return load_yolo_txt_as_pixel_annotation(annotation_path, image_path, classes)
    raise ValueError(f"Unsupported annotation file type: {annotation_path}")


def load_yolo_txt_as_pixel_annotation(
    label_path: Path,
    image_path: Path,
    classes: list[ObjectClassConfig],
) -> AnnotationResult:
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    class_id_to_name = {int(item.id): item.name for item in classes if item.id is not None}
    boxes: list[BoundingBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line {line_number} in {label_path}: {line!r}")

        class_id = int(parts[0])
        label = class_id_to_name.get(class_id)
        if label is None:
            continue

        x_center, y_center, width, height = map(float, parts[1:5])
        box_width = width * image_width
        box_height = height * image_height
        center_x = x_center * image_width
        center_y = y_center * image_height
        boxes.append(
            BoundingBox(
                label=label,
                x_min=max(0.0, center_x - box_width / 2),
                y_min=max(0.0, center_y - box_height / 2),
                x_max=min(float(image_width), center_x + box_width / 2),
                y_max=min(float(image_height), center_y + box_height / 2),
            )
        )

    return AnnotationResult(objects=boxes)
