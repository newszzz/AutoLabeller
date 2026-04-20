from __future__ import annotations

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


def resolve_label_path(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return labels_dir / relative.with_suffix(".txt")


def load_yolo_annotation(
    label_path: Path,
    class_names: list[str],
) -> AnnotationResult:
    if not label_path.exists():
        return AnnotationResult(boxes=[], summary="")

    boxes: list[BoundingBox] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        confidence = float(parts[5]) if len(parts) >= 6 else None
        label = class_names[class_id]
        boxes.append(
            BoundingBox(
                label=label,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence,
            )
        )

    return AnnotationResult(
        boxes=boxes,
        summary=f"{len(boxes)} objects loaded from {label_path.name}.",
    )


def annotation_result_to_llm_result(result: AnnotationResult) -> LlmAnnotationResult:
    return LlmAnnotationResult(
        objects=[
            LlmBox(
                label=box.label,
                x_center=box.x_center,
                y_center=box.y_center,
                width=box.width,
                height=box.height,
            )
            for box in result.boxes
        ],
        summary=result.summary or build_annotation_summary(result.boxes),
        issues=result.issues,
    )


def validate_llm_annotation_result(
    result: LlmAnnotationResult,
    class_names: list[str],
) -> LlmAnnotationResult:
    allowed_labels = set(class_names)
    invalid_labels = sorted({item.label for item in result.objects if item.label not in allowed_labels})
    if invalid_labels:
        raise ValueError(
            f"Model returned unsupported labels {invalid_labels}. Allowed labels: {class_names}"
        )
    return result


def llm_boxes_to_bounding_boxes(items: list[LlmBox], class_names: list[str]) -> list[BoundingBox]:
    allowed_labels = set(class_names)
    boxes: list[BoundingBox] = []
    for item in items:
        if item.label not in allowed_labels:
            raise ValueError(f"Model returned unsupported label {item.label!r}. Allowed labels: {class_names}")
        boxes.append(
            BoundingBox(
                label=item.label,
                x_center=item.x_center,
                y_center=item.y_center,
                width=item.width,
                height=item.height,
            )
        )
    return boxes


def build_annotation_summary(boxes: list[BoundingBox]) -> str:
    if not boxes:
        return "No configured objects were found in the image."

    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.label] = counts.get(box.label, 0) + 1

    ordered_counts = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
    return f"Annotated objects by class: {ordered_counts}."


def save_yolo_annotation(
    image_path: Path,
    result: AnnotationResult,
    labels_dir: Path,
    images_dir: Path,
    class_names: list[str],
) -> Path:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    relative = image_path.relative_to(images_dir).with_suffix(".txt")
    output_path = labels_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [box.to_yolo_line(class_to_idx) for box in result.boxes]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
