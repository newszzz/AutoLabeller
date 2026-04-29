from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import FewShotExampleConfig, ObjectClassConfig
from .dataset import (
    annotation_result_to_llm_result,
    build_class_catalog_text,
    load_annotation_file,
)
from .schemas import AnnotationResult, LlmAnnotationResult, ReviewResult
from .utils import (
    build_label_color_map,
    format_color_legend,
    image_to_data_url,
    render_annotation_image,
)


def build_annotation_system_prompt() -> str:
    return f"""
You are a multimodal object annotation correction agent.

Input:
1. The original target image.
2. The YOLO annotation JSON.
3. The target image rendered with YOLO boxes.
4. Optional few-shot examples. Each example contains an original image, its rendered annotation image, and the target annotation JSON.

Task:
Start from the YOLO boxes, remove extra or wrong boxes, add missing boxes, and fix labels when needed.
If your visual judgment is similar to the YOLO annotation, keep the YOLO result and avoid large changes.

Coordinate rules:
1. All coordinates are absolute pixel positions in the original image.
2. Use x_min, y_min, x_max, y_max.
3. x_min/y_min are the top-left corner. x_max/y_max are the bottom-right corner.
4. Boxes must tightly cover visible target objects.
5. Output floating-point coordinates with at most 3 decimal places.

Output rules:
Return exactly one JSON object. Do not output markdown or chain-of-thought.
The JSON object must follow this schema:

{build_schema_prompt(LlmAnnotationResult)}
""".strip()


def build_review_system_prompt() -> str:
    return f"""
You are a multimodal annotation review agent.

Input:
1. The original target image.
2. YOLO annotation JSON and its rendered annotation image.
3. LLM annotation JSON and its rendered annotation image.

Task:
Check both YOLO and LLM annotations against the original image. Decide whether each result has missing boxes, extra boxes, duplicate boxes, wrong labels, or poor box placement.

Coordinate rules:
All annotation coordinates are absolute pixel positions in the original image.

Output rules:
Return exactly one JSON object. Do not output markdown or chain-of-thought.
The JSON object must follow this schema:

{build_schema_prompt(ReviewResult)}
""".strip()


def build_annotation_user_content(
    classes: list[ObjectClassConfig],
    target_image_path: Path,
    yolo_result: AnnotationResult,
    yolo_image_path: Path,
    few_shots: list[FewShotExampleConfig],
    few_shot_visual_dir: Path,
) -> list[dict[str, Any]]:
    content = [
        {"type": "text", "text": f"Class definitions:\n{build_class_catalog_text(classes)}"},
    ]
    color_by_label = build_label_color_map([item.name for item in classes])
    content.append(
        {
            "type": "text",
            "text": (
                "Annotation image color legend. Rendered annotation images use rectangle "
                "outlines only, with no text labels:\n"
                f"{format_color_legend(color_by_label)}"
            ),
        }
    )
    content.extend(_few_shot_content(few_shots, few_shot_visual_dir, classes, color_by_label))
    content.extend(
        [
            {"type": "text", "text": "TARGET ORIGINAL IMAGE"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target_image_path)}},
            {"type": "text", "text": f"YOLO annotation JSON:\n{_annotation_json(yolo_result)}"},
            {"type": "text", "text": "TARGET IMAGE WITH YOLO BOXES"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(yolo_image_path)}},
            {
                "type": "text",
                "text": (
                    "Return the corrected annotation JSON. In issues, list only changes made "
                    "relative to the YOLO annotation. If the corrected annotation is close to "
                    "YOLO, keep the YOLO boxes. Use at most 3 decimal places for coordinates."
                ),
            },
        ]
    )
    return content


def build_review_user_content(
    classes: list[ObjectClassConfig],
    target_image_path: Path,
    yolo_result: AnnotationResult,
    yolo_image_path: Path,
    llm_result: LlmAnnotationResult,
    llm_image_path: Path,
) -> list[dict[str, Any]]:
    color_by_label = build_label_color_map([item.name for item in classes])
    return [
        {"type": "text", "text": f"Class definitions:\n{build_class_catalog_text(classes)}"},
        {
            "type": "text",
            "text": (
                "Annotation image color legend. Rendered annotation images use rectangle "
                "outlines only, with no text labels:\n"
                f"{format_color_legend(color_by_label)}"
            ),
        },
        {"type": "text", "text": "TARGET ORIGINAL IMAGE"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(target_image_path)}},
        {"type": "text", "text": f"YOLO annotation JSON:\n{_annotation_json(yolo_result)}"},
        {"type": "text", "text": "TARGET IMAGE WITH YOLO BOXES"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(yolo_image_path)}},
        {"type": "text", "text": f"LLM annotation JSON:\n{_llm_json(llm_result)}"},
        {"type": "text", "text": "TARGET IMAGE WITH LLM BOXES"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(llm_image_path)}},
        {"type": "text", "text": "Return the review JSON."},
    ]


def build_schema_prompt(schema_model: type[BaseModel]) -> str:
    schema = schema_model.model_json_schema()
    return "\n".join(_schema_lines(schema.get("properties", {}), schema))


def _few_shot_content(
    examples: list[FewShotExampleConfig],
    visual_dir: Path,
    classes: list[ObjectClassConfig],
    color_by_label: dict[str, tuple[int, int, int]],
) -> list[dict[str, Any]]:
    if not examples:
        return [{"type": "text", "text": "FEW-SHOT EXAMPLES\nNone provided."}]

    content: list[dict[str, Any]] = [{"type": "text", "text": "FEW-SHOT EXAMPLES"}]
    for index, example in enumerate(examples, start=1):
        if not example.image_path.exists():
            raise FileNotFoundError(f"Few-shot image not found: {example.image_path}")
        if not example.annotation_path.exists():
            raise FileNotFoundError(f"Few-shot annotation not found: {example.annotation_path}")

        annotation = load_annotation_file(example.annotation_path, example.image_path, classes)
        rendered_path = render_annotation_image(
            example.image_path,
            annotation,
            visual_dir / f"few_shot_{index:03d}_{example.image_path.stem}.png",
            color_by_label,
        )
        content.extend(
            [
                {"type": "text", "text": f"Few-shot #{index} original image:"},
                {"type": "image_url", "image_url": {"url": image_to_data_url(example.image_path)}},
                {"type": "text", "text": f"Few-shot #{index} annotated image:"},
                {"type": "image_url", "image_url": {"url": image_to_data_url(rendered_path)}},
                {
                    "type": "text",
                    "text": (
                        f"Few-shot #{index} annotation JSON:\n"
                        f"{_llm_json(annotation_result_to_llm_result(annotation))}"
                    ),
                },
            ]
        )
    return content


def _annotation_json(result: AnnotationResult) -> str:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


def _llm_json(result: LlmAnnotationResult) -> str:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


def _schema_lines(
    properties: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    indent: int = 0,
) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    required = set(root_schema.get("required", []))
    for name, field_schema in properties.items():
        description = field_schema.get("description", "")
        required_mark = "required" if name in required else "optional"
        lines.append(f"{prefix}- {name}: {_schema_type(field_schema, root_schema)}, {required_mark}. {description}")

        nested = _nested_schema(field_schema, root_schema)
        if nested:
            lines.extend(
                _schema_lines(nested.get("properties", {}), nested, indent=indent + 2)
            )
    return lines


def _schema_type(field_schema: dict[str, Any], root_schema: dict[str, Any]) -> str:
    field_schema = _merge_nullable(field_schema)
    if "$ref" in field_schema:
        return _schema_type(_resolve_ref(field_schema["$ref"], root_schema), root_schema)

    type_name = field_schema.get("type", "unknown")
    if "enum" in field_schema:
        type_name = " | ".join(repr(item) for item in field_schema["enum"])
    elif type_name == "array":
        type_name = f"list[{_schema_type(field_schema.get('items', {}), root_schema)}]"
    elif type_name == "number":
        type_name = "float"
    elif type_name == "integer":
        type_name = "int"
    elif type_name == "boolean":
        type_name = "bool"

    limits = []
    if "minimum" in field_schema:
        limits.append(f">= {field_schema['minimum']}")
    if "exclusiveMinimum" in field_schema:
        limits.append(f"> {field_schema['exclusiveMinimum']}")
    if "maximum" in field_schema:
        limits.append(f"<= {field_schema['maximum']}")
    if limits:
        type_name = f"{type_name} ({', '.join(limits)})"
    if field_schema.get("nullable"):
        type_name = f"{type_name} | null"
    return str(type_name)


def _nested_schema(field_schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any] | None:
    field_schema = _merge_nullable(field_schema)
    if "$ref" in field_schema:
        field_schema = _resolve_ref(field_schema["$ref"], root_schema)
    if field_schema.get("type") == "object":
        return field_schema
    if field_schema.get("type") == "array":
        items = _merge_nullable(field_schema.get("items", {}))
        if "$ref" in items:
            items = _resolve_ref(items["$ref"], root_schema)
        if items.get("type") == "object":
            return items
    return None


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    return root_schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})


def _merge_nullable(field_schema: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" not in field_schema:
        return field_schema
    non_null = [option for option in field_schema["anyOf"] if option.get("type") != "null"]
    if len(non_null) != 1:
        return field_schema
    merged = dict(non_null[0])
    merged["nullable"] = True
    if field_schema.get("description"):
        merged["description"] = field_schema["description"]
    return merged
