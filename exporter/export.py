from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from autolabeller.dataset import (
    annotation_result_to_llm_result,
    build_class_catalog_text,
    load_annotation_file,
    validate_annotation_result,
    validate_llm_annotation_result,
)
from autolabeller.prompts import build_annotation_system_prompt
from autolabeller.schemas import AnnotationResult, BoundingBox, ImageRecord, LlmAnnotationResult, LlmBox
from autolabeller.utils import (
    build_label_color_map,
    ensure_dir,
    format_color_legend,
    render_annotation_image,
    write_json,
)
from exporter.config import ExportDatasetConfig, ExportObjectClassConfig, load_export_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class GroundTruthRecord:
    image_path: Path
    annotation_path: Path
    image_record: ImageRecord
    annotation: AnnotationResult


def export_annotate_finetune_data(config: ExportDatasetConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir.resolve())
    output_path = output_dir / "dataset.json"
    dataset_info_path = output_dir / "dataset_info.json"
    sft_images_dir = ensure_dir(output_dir / "images")
    output_base_dir = output_dir
    class_names = [item.name for item in config.classes]
    rng = random.Random(config.random_seed)

    records = collect_ground_truth_records(config)
    negative_count_target = min(
        len(records),
        max(0, int(len(records) * config.negative_sample_ratio + 0.5)),
    )
    negative_indexes = set(rng.sample(range(len(records)), k=negative_count_target))

    copied_original_dir = ensure_dir(sft_images_dir / "original")
    rendered_yolo_dir = ensure_dir(sft_images_dir / "yolo")
    few_shot_original_dir = ensure_dir(sft_images_dir / "few_shot_original")
    few_shot_visual_dir = ensure_dir(sft_images_dir / "few_shot_annotation")
    color_by_label = build_label_color_map(class_names)

    samples: list[dict[str, Any]] = []
    negative_count = 0
    for index, record in enumerate(records):
        is_negative = index in negative_indexes
        yolo_result = (
            simulate_negative_yolo_result(record.annotation, config.classes, record.image_record, rng)
            if is_negative
            else _annotation_with_confidence(record.annotation, rng)
        )
        validate_annotation_result(
            yolo_result,
            class_names,
            record.image_record.width,
            record.image_record.height,
        )

        target_image_path = copy_dataset_image(
            record.image_path,
            copied_original_dir / record.image_path.relative_to(config.images_dir),
        )
        yolo_image_path = render_annotation_image(
            record.image_path,
            yolo_result,
            rendered_yolo_dir / record.image_path.relative_to(config.images_dir).with_suffix(".png"),
            color_by_label,
        )
        target_result = prepare_target_result(
            record.annotation,
            yolo_result,
            synthesize_issues=config.synthesize_issues and is_negative,
        )
        validate_llm_annotation_result(
            target_result,
            class_names,
            record.image_record.width,
            record.image_record.height,
        )

        few_shots = sample_few_shot_records(
            records,
            target_index=index,
            max_few_shots=config.max_few_shots,
            rng=rng,
        )
        user_content, images = build_llamafactory_annotation_user_content(
            classes=config.classes,
            yolo_result=yolo_result,
            yolo_image_path=yolo_image_path,
            few_shots=few_shots,
            few_shot_visual_dir=few_shot_visual_dir,
            few_shot_original_dir=few_shot_original_dir,
            color_by_label=color_by_label,
            output_base_dir=output_base_dir,
            image_path_mode=config.image_path_mode,
            images_dir=config.images_dir,
            target_image_path_for_sft=target_image_path,
        )
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": build_annotation_system_prompt()},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": _llm_json(target_result)},
                ],
                "images": images,
            }
        )
        if is_negative:
            negative_count += 1

    validate_llamafactory_samples(samples, output_base_dir)
    write_json(output_path, samples)
    dataset_info_path = write_llamafactory_dataset_info(
        output_path,
        dataset_name=config.dataset_name,
        dataset_info_path=dataset_info_path,
    )
    return {
        "dataset_name": config.dataset_name,
        "samples": len(samples),
        "negative_samples": negative_count,
        "negative_sample_ratio": config.negative_sample_ratio,
        "output_dir": str(output_dir),
        "output_path": str(output_path),
        "dataset_info_path": str(dataset_info_path),
        "sft_images_dir": str(sft_images_dir),
        "format": "llamafactory_sharegpt_multimodal",
    }


def collect_ground_truth_records(config: ExportDatasetConfig) -> list[GroundTruthRecord]:
    if not config.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {config.images_dir}")
    if not config.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {config.labels_dir}")

    extensions = {item.lower() for item in config.image_extensions}
    class_names = [item.name for item in config.classes]
    records: list[GroundTruthRecord] = []
    for image_path in sorted(
        path
        for path in config.images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ):
        annotation_path = _matching_annotation_path(image_path, config.images_dir, config.labels_dir)
        if annotation_path is None:
            continue
        image_record = _image_record(image_path)
        annotation = validate_annotation_result(
            load_annotation_file(annotation_path, image_path, config.classes),
            class_names,
            image_record.width,
            image_record.height,
        )
        records.append(
            GroundTruthRecord(
                image_path=image_path,
                annotation_path=annotation_path,
                image_record=image_record,
                annotation=annotation,
            )
        )
    if not records:
        raise ValueError(
            f"No image/label pairs found under {config.images_dir} and {config.labels_dir}."
        )
    return records


def sample_few_shot_records(
    records: list[GroundTruthRecord],
    *,
    target_index: int,
    max_few_shots: int,
    rng: random.Random,
) -> list[GroundTruthRecord]:
    count = rng.randint(0, max_few_shots)
    if count == 0:
        return []
    candidates = [item for index, item in enumerate(records) if index != target_index]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(count, len(candidates)))


def build_llamafactory_annotation_user_content(
    *,
    classes: list[ExportObjectClassConfig],
    yolo_result: AnnotationResult,
    yolo_image_path: Path,
    few_shots: list[GroundTruthRecord],
    few_shot_visual_dir: Path,
    few_shot_original_dir: Path,
    color_by_label: dict[str, tuple[int, int, int]],
    output_base_dir: Path,
    image_path_mode: Literal["relative", "absolute"] = "relative",
    images_dir: Path,
    target_image_path_for_sft: Path,
) -> tuple[str, list[str]]:
    parts: list[str] = [
        f"Class definitions:\n{build_class_catalog_text(classes)}",
        (
            "Annotation image color legend. Rendered annotation images use rectangle "
            "outlines only, with no text labels:\n"
            f"{format_color_legend(color_by_label)}"
        ),
    ]
    images: list[str] = []
    visual_parts: list[str] = ["VISUAL INPUTS"]
    few_shot_json_parts: list[str] = []

    if few_shots:
        few_shot_json_parts.append("FEW-SHOT EXAMPLES")
        for index, example in enumerate(few_shots, start=1):
            few_shot_image_path = copy_dataset_image(
                example.image_path,
                few_shot_original_dir / example.image_path.relative_to(images_dir),
            )
            rendered_path = render_annotation_image(
                example.image_path,
                example.annotation,
                few_shot_visual_dir / example.image_path.relative_to(images_dir).with_suffix(".png"),
                color_by_label,
            )
            visual_parts.append(f"Few-shot #{index} original image:\n<image>")
            images.append(_format_image_path(few_shot_image_path, output_base_dir, image_path_mode))
            visual_parts.append(f"Few-shot #{index} annotated image:\n<image>")
            images.append(_format_image_path(rendered_path, output_base_dir, image_path_mode))
            few_shot_json_parts.append(
                f"Few-shot #{index} annotation JSON:\n"
                f"{_llm_json(round_llm_annotation_result(annotation_result_to_llm_result(example.annotation)))}"
            )

    visual_parts.extend(
        [
            "TARGET ORIGINAL IMAGE\n<image>",
            "TARGET IMAGE WITH YOLO BOXES\n<image>",
        ]
    )
    images.append(_format_image_path(target_image_path_for_sft, output_base_dir, image_path_mode))
    images.append(_format_image_path(yolo_image_path, output_base_dir, image_path_mode))
    parts.append("\n\n".join(visual_parts))
    parts.extend(few_shot_json_parts)
    parts.extend(
        [
            f"YOLO annotation JSON:\n{_annotation_json(yolo_result)}",
            (
                "Return the corrected annotation JSON. In issues, list only changes made "
                "relative to the YOLO annotation. If the corrected annotation is close to "
                "YOLO, keep the YOLO boxes. Use at most 3 decimal places for coordinates."
            ),
        ]
    )
    return "\n\n".join(parts), images


def simulate_negative_yolo_result(
    target: AnnotationResult,
    classes: list[ExportObjectClassConfig],
    record: ImageRecord,
    rng: random.Random,
) -> AnnotationResult:
    boxes = [_bounding_box_with_confidence(item, rng) for item in target.objects]
    operations = ["extra"]
    if boxes:
        operations.extend(["missing", "duplicate", "wrong_label", "jitter"])

    rng.shuffle(operations)
    op_count = rng.randint(1, min(3, len(operations)))
    for operation in operations[:op_count]:
        if operation == "missing" and boxes:
            boxes.pop(rng.randrange(len(boxes)))
        elif operation == "duplicate" and boxes:
            boxes.append(_jitter_box(rng.choice(boxes), record, rng, scale=0.08))
        elif operation == "wrong_label" and boxes:
            box_index = rng.randrange(len(boxes))
            boxes[box_index] = _box_with_label(
                boxes[box_index],
                _different_label(boxes[box_index].label, classes, rng),
            )
        elif operation == "jitter" and boxes:
            box_index = rng.randrange(len(boxes))
            boxes[box_index] = _jitter_box(boxes[box_index], record, rng, scale=0.18)
        elif operation == "extra":
            boxes.append(_extra_box(target, classes, record, rng))

    if _annotations_equivalent(AnnotationResult(objects=boxes), target):
        boxes.append(_extra_box(target, classes, record, rng))
    return AnnotationResult(objects=boxes)


def prepare_target_result(
    target: AnnotationResult,
    yolo_result: AnnotationResult,
    *,
    synthesize_issues: bool,
) -> LlmAnnotationResult:
    rounded = round_llm_annotation_result(annotation_result_to_llm_result(target))
    if not synthesize_issues:
        return rounded
    return LlmAnnotationResult(
        objects=rounded.objects,
        issues=synthesize_annotation_issues(yolo_result, rounded),
    )


def write_llamafactory_dataset_info(
    output_path: Path,
    *,
    dataset_name: str,
    dataset_info_path: Path,
) -> Path:
    payload = {
        dataset_name: {
            "file_name": _format_posix_path(
                os.path.relpath(output_path.resolve(), dataset_info_path.resolve().parent)
            ),
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
    write_json(dataset_info_path, payload)
    return dataset_info_path


def validate_llamafactory_samples(samples: list[dict[str, Any]], output_base_dir: Path) -> None:
    for index, sample in enumerate(samples):
        text = "\n".join(
            str(message.get("content", ""))
            for message in sample.get("messages", [])
            if isinstance(message, dict)
        )
        image_token_count = text.count("<image>")
        images = sample.get("images", [])
        if image_token_count != len(images):
            raise ValueError(
                f"Sample {index} has {image_token_count} <image> tokens but {len(images)} image paths."
            )
        for image_path in images:
            if "\\" in image_path:
                raise ValueError(f"Sample {index} contains a Windows-style image path: {image_path}")
            if ":" in image_path[:3]:
                raise ValueError(f"Sample {index} contains an absolute drive path: {image_path}")
            if not (output_base_dir / image_path).exists():
                raise FileNotFoundError(f"Sample {index} image path does not exist: {image_path}")


def round_llm_annotation_result(result: LlmAnnotationResult, ndigits: int = 3) -> LlmAnnotationResult:
    return LlmAnnotationResult(
        objects=[
            LlmBox(
                label=item.label,
                x_min=round(item.x_min, ndigits),
                y_min=round(item.y_min, ndigits),
                x_max=round(item.x_max, ndigits),
                y_max=round(item.y_max, ndigits),
            )
            for item in result.objects
        ],
        issues=list(result.issues),
    )


def synthesize_annotation_issues(
    yolo_result: AnnotationResult,
    target_result: LlmAnnotationResult,
) -> list[str]:
    issues: list[str] = []
    used_target_indexes: set[int] = set()

    for yolo_box in yolo_result.objects:
        best_index, best_iou = _best_target_match(yolo_box, target_result.objects, used_target_indexes)
        if best_index is None or best_iou < 0.5:
            issues.append(f"Removed extra {yolo_box.label} box at {_box_coords(yolo_box)}")
            continue

        used_target_indexes.add(best_index)
        target_box = target_result.objects[best_index]
        if yolo_box.label != target_box.label and best_iou >= 0.7:
            issues.append(
                f"Fixed label from {yolo_box.label} to {target_box.label} at {_box_coords(target_box)}"
            )
        elif _box_shift_is_meaningful(yolo_box, target_box) and best_iou < 0.98:
            issues.append(
                f"Adjusted {target_box.label} box from {_box_coords(yolo_box)} to {_box_coords(target_box)}"
            )

    for index, target_box in enumerate(target_result.objects):
        if index not in used_target_indexes:
            issues.append(f"Added missing {target_box.label} box at {_box_coords(target_box)}")

    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export annotate-stage SFT data for LLaMA-Factory multimodal training."
    )
    parser.add_argument(
        "config",
        help="Export YAML config name under config/, for example export_annotate or export_annotate.yaml.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    loaded = load_export_config(args.config)
    summary = export_annotate_finetune_data(loaded.dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _matching_annotation_path(image_path: Path, images_dir: Path, labels_dir: Path) -> Path | None:
    relative = image_path.relative_to(images_dir)
    for suffix in (".json", ".txt"):
        candidate = labels_dir / relative.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def copy_dataset_image(source_path: Path, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    shutil.copy2(source_path, output_path)
    return output_path


def _image_record(image_path: Path) -> ImageRecord:
    with Image.open(image_path) as image:
        width, height = image.size
    return ImageRecord(image_path=image_path, width=width, height=height)


def _annotation_with_confidence(result: AnnotationResult, rng: random.Random) -> AnnotationResult:
    return AnnotationResult(
        objects=[_bounding_box_with_confidence(item, rng) for item in result.objects]
    )


def _bounding_box_with_confidence(item: BoundingBox, rng: random.Random) -> BoundingBox:
    return BoundingBox(
        label=item.label,
        x_min=item.x_min,
        y_min=item.y_min,
        x_max=item.x_max,
        y_max=item.y_max,
        confidence=round(rng.uniform(0.75, 0.98), 6),
    )


def _box_with_label(item: BoundingBox, label: str) -> BoundingBox:
    return BoundingBox(
        label=label,
        x_min=item.x_min,
        y_min=item.y_min,
        x_max=item.x_max,
        y_max=item.y_max,
        confidence=item.confidence,
    )


def _different_label(
    current_label: str,
    classes: list[ExportObjectClassConfig],
    rng: random.Random,
) -> str:
    candidates = [item.name for item in classes if item.name != current_label]
    if not candidates:
        return current_label
    return rng.choice(candidates)


def _jitter_box(
    item: BoundingBox,
    record: ImageRecord,
    rng: random.Random,
    *,
    scale: float,
) -> BoundingBox:
    width = item.x_max - item.x_min
    height = item.y_max - item.y_min
    dx1 = rng.uniform(-scale, scale) * width
    dy1 = rng.uniform(-scale, scale) * height
    dx2 = rng.uniform(-scale, scale) * width
    dy2 = rng.uniform(-scale, scale) * height
    x_min = _clamp(item.x_min + dx1, 0.0, max(0.0, record.width - 2.0))
    y_min = _clamp(item.y_min + dy1, 0.0, max(0.0, record.height - 2.0))
    x_max = _clamp(item.x_max + dx2, x_min + 1.0, float(record.width))
    y_max = _clamp(item.y_max + dy2, y_min + 1.0, float(record.height))
    return BoundingBox(
        label=item.label,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        confidence=item.confidence if item.confidence is not None else round(rng.uniform(0.45, 0.9), 6),
    )


def _extra_box(
    target: AnnotationResult,
    classes: list[ExportObjectClassConfig],
    record: ImageRecord,
    rng: random.Random,
) -> BoundingBox:
    if target.objects:
        source = rng.choice(target.objects)
        width = source.x_max - source.x_min
        height = source.y_max - source.y_min
        x_min = _clamp(
            source.x_min + rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.8) * width,
            0.0,
            max(0.0, record.width - width - 1.0),
        )
        y_min = _clamp(
            source.y_min + rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.8) * height,
            0.0,
            max(0.0, record.height - height - 1.0),
        )
        x_max = min(float(record.width), x_min + width)
        y_max = min(float(record.height), y_min + height)
        label = source.label
    else:
        box_width = rng.uniform(0.08, 0.25) * record.width
        box_height = rng.uniform(0.08, 0.25) * record.height
        x_min = rng.uniform(0.0, max(0.0, record.width - box_width))
        y_min = rng.uniform(0.0, max(0.0, record.height - box_height))
        x_max = x_min + box_width
        y_max = y_min + box_height
        label = rng.choice(classes).name

    return BoundingBox(
        label=label,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        confidence=round(rng.uniform(0.35, 0.88), 6),
    )


def _annotations_equivalent(first: AnnotationResult, second: AnnotationResult) -> bool:
    if len(first.objects) != len(second.objects):
        return False
    used: set[int] = set()
    for item in first.objects:
        match_index, match_iou = _best_annotation_match(item, second.objects, used)
        if match_index is None or match_iou < 0.98 or item.label != second.objects[match_index].label:
            return False
        used.add(match_index)
    return True


def _annotation_json(result: AnnotationResult) -> str:
    return json.dumps(_round_annotation_payload(result.model_dump()), ensure_ascii=False, indent=2)


def _llm_json(result: LlmAnnotationResult) -> str:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


def _round_annotation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rounded = {"objects": []}
    for item in payload.get("objects", []):
        rounded_item = dict(item)
        for key in ("x_min", "y_min", "x_max", "y_max"):
            if key in rounded_item:
                rounded_item[key] = round(float(rounded_item[key]), 3)
        if "confidence" in rounded_item and rounded_item["confidence"] is not None:
            rounded_item["confidence"] = round(float(rounded_item["confidence"]), 6)
        rounded["objects"].append(rounded_item)
    return rounded


def _format_image_path(
    image_path: Path,
    output_base_dir: Path,
    image_path_mode: Literal["relative", "absolute"],
) -> str:
    if image_path_mode == "absolute":
        return _format_posix_path(str(image_path.resolve()))
    return _format_posix_path(os.path.relpath(image_path.resolve(), output_base_dir.resolve()))


def _format_posix_path(path: str) -> str:
    return path.replace("\\", "/")


def _best_target_match(
    yolo_box: BoundingBox,
    target_boxes: list[LlmBox],
    used_target_indexes: set[int],
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_iou = 0.0
    for index, target_box in enumerate(target_boxes):
        if index in used_target_indexes:
            continue
        score = _iou(yolo_box, target_box)
        if score > best_iou:
            best_index = index
            best_iou = score
    return best_index, best_iou


def _best_annotation_match(
    box: BoundingBox,
    target_boxes: list[BoundingBox],
    used_target_indexes: set[int],
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_iou = 0.0
    for index, target_box in enumerate(target_boxes):
        if index in used_target_indexes:
            continue
        score = _iou(box, target_box)
        if score > best_iou:
            best_index = index
            best_iou = score
    return best_index, best_iou


def _iou(first: BoundingBox | LlmBox, second: BoundingBox | LlmBox) -> float:
    inter_x_min = max(first.x_min, second.x_min)
    inter_y_min = max(first.y_min, second.y_min)
    inter_x_max = min(first.x_max, second.x_max)
    inter_y_max = min(first.y_max, second.y_max)
    inter_width = max(0.0, inter_x_max - inter_x_min)
    inter_height = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _box_shift_is_meaningful(first: BoundingBox, second: LlmBox) -> bool:
    return any(
        abs(first_value - second_value) >= 1.0
        for first_value, second_value in (
            (first.x_min, second.x_min),
            (first.y_min, second.y_min),
            (first.x_max, second.x_max),
            (first.y_max, second.y_max),
        )
    )


def _box_coords(box: BoundingBox | LlmBox) -> list[float]:
    return [
        round(box.x_min, 3),
        round(box.y_min, 3),
        round(box.x_max, 3),
        round(box.y_max, 3),
    ]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


if __name__ == "__main__":
    main()
