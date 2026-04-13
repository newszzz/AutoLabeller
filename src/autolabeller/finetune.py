from __future__ import annotations

import json
from pathlib import Path

from .config import AppConfig
from .dataset import iter_finetune_pairs, load_class_names, load_yolo_annotation
from .schemas import AnnotationResult, BoundingBox
from .utils import write_json, write_jsonl


def export_sft_datasets(config: AppConfig) -> dict[str, int]:
    class_names = load_class_names(config.dataset)

    generic_records: list[dict] = []
    llamafactory_records: list[dict] = []

    if config.finetune.use_reviewed_annotations:
        reviewed_json_paths = sorted((config.dataset.output_dir / "json").rglob("*.json"))
        annotations = [_load_reviewed_annotation(path) for path in reviewed_json_paths]
        examples = [(annotation.image_path, annotation) for annotation in annotations]
    else:
        if config.finetune.dataset is None:
            raise ValueError("finetune.dataset must be configured when use_reviewed_annotations is false.")
        pairs = iter_finetune_pairs(config.finetune.dataset, config.dataset.image_extensions)
        examples = [
            (
                image_path,
                load_yolo_annotation(
                    image_path=image_path,
                    label_path=label_path,
                    class_names=class_names,
                    source="ground_truth",
                ),
            )
            for image_path, label_path in pairs
        ]

    for image_path, annotation in examples:
        assistant_payload = {
            "objects": [
                {
                    "label": box.label,
                    "bbox": box.as_list(),
                }
                for box in annotation.boxes
            ],
            "summary": annotation.summary or f"Detected {len(annotation.boxes)} objects.",
        }

        generic_records.append(
            {
                "image": str(image_path),
                "prompt": config.finetune.task_prompt,
                "response": assistant_payload,
            }
        )

        llamafactory_records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>\n{config.finetune.task_prompt}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_payload, ensure_ascii=False),
                    },
                ],
                "images": [str(image_path)],
            }
        )

    write_jsonl(config.finetune.generic_output, generic_records)
    write_json(config.finetune.llamafactory_output, llamafactory_records)
    return {
        "generic_records": len(generic_records),
        "llamafactory_records": len(llamafactory_records),
    }


def _load_reviewed_annotation(reviewed_json_path: Path) -> AnnotationResult:
    payload = json.loads(reviewed_json_path.read_text(encoding="utf-8"))
    image_path = Path(payload["image_path"])
    final_payload = payload.get("final", {})
    return AnnotationResult(
        image_path=image_path,
        boxes=[_box_from_dict(item) for item in final_payload.get("boxes", [])],
        source="reviewer",
        summary=final_payload.get("summary", ""),
        issues=final_payload.get("issues", []),
    )


def _box_from_dict(payload: dict) -> BoundingBox:
    return BoundingBox(
        label=payload["label"],
        x_center=payload["x_center"],
        y_center=payload["y_center"],
        width=payload["width"],
        height=payload["height"],
        confidence=payload.get("confidence"),
        rationale=payload.get("rationale"),
        source=payload.get("source", "reviewer"),
    )
