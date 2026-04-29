from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import AliasChoices, BaseModel, Field

from autolabeller.config import ObjectClassConfig
from autolabeller.dataset import build_class_catalog_text, validate_llm_annotation_result
from autolabeller.schemas import LlmAnnotationResult, LlmBox
from autolabeller.utils import image_to_data_url


class DemoFewShotConfig(BaseModel):
    image_path: Path
    label_path: Path = Field(validation_alias=AliasChoices("label_path", "target_label_path"))


class DemoDataConfig(BaseModel):
    image: Path
    classes: list[ObjectClassConfig]
    few_shots: list[DemoFewShotConfig] = Field(default_factory=list)


class DemoVllmConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "0"
    model: str
    temperature: float = 0.0
    request_timeout: float = 120.0


class DemoConfig(BaseModel):
    data: DemoDataConfig
    vllm: DemoVllmConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo for LlmAnnotationResult structured detection via vLLM and LangChain."
    )
    parser.add_argument("config", type=Path, help="Path to demo YAML config.")
    return parser


def load_demo_config(config_path: Path) -> DemoConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = DemoConfig.model_validate(raw_config)
    base_dir = config_path.parent

    def resolve(path: Path) -> Path:
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()

    config.data.image = resolve(config.data.image)  # type: ignore[assignment]
    config.data.few_shots = [  # type: ignore[assignment]
        DemoFewShotConfig(
            image_path=resolve(item.image_path),
            label_path=resolve(item.label_path),
        )
        for item in config.data.few_shots
    ]
    config.vllm.base_url = normalize_base_url(config.vllm.base_url)
    return config


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def build_response_format(schema_model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "schema": schema_model.model_json_schema(),
        },
    }


def resolve_schema_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise ValueError(f"Unsupported JSON Schema reference: {ref}")
    return root_schema.get("$defs", {}).get(ref.removeprefix(prefix), {})


def merge_nullable_schema(field_schema: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" not in field_schema:
        return field_schema

    non_null_options = [
        option for option in field_schema["anyOf"] if option.get("type") != "null"
    ]
    if len(non_null_options) != 1:
        return field_schema

    merged = dict(non_null_options[0])
    if field_schema.get("description"):
        merged["description"] = field_schema["description"]
    merged["nullable"] = True
    return merged


def schema_type_name(field_schema: dict[str, Any], root_schema: dict[str, Any]) -> str:
    field_schema = merge_nullable_schema(field_schema)
    if "$ref" in field_schema:
        return schema_type_name(resolve_schema_ref(field_schema["$ref"], root_schema), root_schema)

    type_name = field_schema.get("type", "unknown")
    if type_name == "string":
        result = "str"
    elif type_name == "boolean":
        result = "bool"
    elif type_name == "integer":
        result = "int"
    elif type_name == "number":
        result = "float"
    elif type_name == "array":
        result = f"list[{schema_type_name(field_schema.get('items', {}), root_schema)}]"
    elif type_name == "object":
        result = "object"
    else:
        result = str(type_name)

    constraints: list[str] = []
    if "minimum" in field_schema:
        constraints.append(f">= {field_schema['minimum']}")
    if "exclusiveMinimum" in field_schema:
        constraints.append(f"> {field_schema['exclusiveMinimum']}")
    if "maximum" in field_schema:
        constraints.append(f"<= {field_schema['maximum']}")
    if "exclusiveMaximum" in field_schema:
        constraints.append(f"< {field_schema['exclusiveMaximum']}")
    if constraints:
        result = f"{result} ({', '.join(constraints)})"
    if field_schema.get("nullable"):
        result = f"{result} | null"
    return result


def nested_object_schema(
    field_schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> dict[str, Any] | None:
    field_schema = merge_nullable_schema(field_schema)
    if "$ref" in field_schema:
        field_schema = resolve_schema_ref(field_schema["$ref"], root_schema)

    if field_schema.get("type") == "object":
        return field_schema

    if field_schema.get("type") == "array":
        item_schema = merge_nullable_schema(field_schema.get("items", {}))
        if "$ref" in item_schema:
            item_schema = resolve_schema_ref(item_schema["$ref"], root_schema)
        if item_schema.get("type") == "object":
            return item_schema

    return None


def build_schema_lines(
    properties: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    indent: int = 0,
) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for name, field_schema in properties.items():
        description = field_schema.get("description", "")
        lines.append(f"{prefix}-{name}: {schema_type_name(field_schema, root_schema)}, {description}")

        nested_schema = nested_object_schema(field_schema, root_schema)
        if nested_schema is not None:
            nested_properties = nested_schema.get("properties", {})
            lines.extend(build_schema_lines(nested_properties, root_schema, indent=indent + 2))
    return lines


def build_schema_prompt(schema_model: type[BaseModel]) -> str:
    schema = schema_model.model_json_schema()
    schema_lines = "\n".join(build_schema_lines(schema.get("properties", {}), schema))
    return f"""
You are an object detection annotation assistant.
You will receive one user message containing class definitions, optional few-shot examples, and one target image.

Input interpretation rules:
1. FEW-SHOT EXAMPLES are references only. Use them to learn label choice, box tightness, and annotation style.
2. Do not include few-shot objects in the final answer.
3. TARGET IMAGE is the only image that must be annotated in the final answer.

Annotation rules:
1. Detect only objects that match the configured class definitions.
2. label must exactly match one configured class name.
3. Use normalized decimal YOLO-style coordinates relative to the full target image.
4. x_center and y_center are the normalized center of the box, from 0.0 to 1.0.
5. width and height are the normalized box size, from 0.0 to 1.0.
6. Each box should tightly cover one visible object.
7. Do not annotate the same object multiple times.
8. Put uncertain, occluded, tiny, or hard-to-judge target-image cases in issues.
9. If no configured objects are visible in the target image, return an empty objects list and explain that in summary.

Output rules:
Return exactly one JSON object for the target image. Do not output markdown or chain-of-thought.
The JSON object must follow this schema:

{schema_lines}
""".strip()


def load_yolo_label_as_llm_result(
    label_path: Path,
    class_names: list[str],
) -> LlmAnnotationResult:
    objects: list[LlmBox] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line {line_number} in {label_path}: {line!r}")

        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(
                f"Class id {class_id} on line {line_number} is outside configured classes."
            )
        x_center, y_center, width, height = map(float, parts[1:5])
        objects.append(
            LlmBox(
                label=class_names[class_id],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return LlmAnnotationResult(
        objects=objects,
        summary=build_annotation_summary(objects),
        issues=[],
    )


def build_annotation_summary(objects: list[LlmBox]) -> str:
    if not objects:
        return "No configured objects were found in the image."

    counts: dict[str, int] = {}
    for item in objects:
        counts[item.label] = counts.get(item.label, 0) + 1
    ordered_counts = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
    return f"Annotated objects by class: {ordered_counts}."


def build_user_content(config: DemoConfig) -> list[dict[str, Any]]:
    class_names = [item.name for item in config.data.classes]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"""
Class definitions:
{build_class_catalog_text(config.data.classes)}
""".strip(),
        }
    ]

    if config.data.few_shots:
        content.append({"type": "text", "text": "FEW-SHOT EXAMPLES"})
    else:
        content.append({"type": "text", "text": "FEW-SHOT EXAMPLES\nNone provided."})
    for index, example in enumerate(config.data.few_shots, start=1):
        if not example.image_path.exists():
            raise FileNotFoundError(f"Few-shot image not found: {example.image_path}")
        if not example.label_path.exists():
            raise FileNotFoundError(f"Few-shot label not found: {example.label_path}")

        annotation = load_yolo_label_as_llm_result(example.label_path, class_names)
        content.extend(
            [
                {"type": "text", "text": f"Image {index}:"},
                {"type": "image_url", "image_url": {"url": image_to_data_url(example.image_path)}},
                {
                    "type": "text",
                    "text": (
                        f"Annotation {index}:\n"
                        f"{json.dumps(annotation.model_dump(), ensure_ascii=False)}"
                    ),
                },
            ]
        )

    content.append({"type": "text", "text": "TARGET IMAGE"})
    content.extend(
        [
            {"type": "text", "text": f"Target Image:"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(config.data.image)}},
            {"type": "text", "text": "Target Annotation:"},
        ]
    )
    return content


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def validate_response_content(content: Any, class_names: list[str]) -> LlmAnnotationResult:
    text = extract_text(content)
    try:
        parsed = LlmAnnotationResult.model_validate_json(text)
        return validate_llm_annotation_result(parsed, class_names)
    except Exception as exc:
        raise RuntimeError(f"Structured parsing failed: {exc}\nRaw response: {text}") from exc


def color_for_label(label: str) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (0, 119, 182),
        (255, 183, 3),
        (106, 153, 78),
    ]
    return palette[sum(ord(char) for char in label) % len(palette)]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def draw_annotation_result(
    image_path: Path,
    result: LlmAnnotationResult,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_llm_boxes.png")

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()
    line_width = max(2, round(min(width, height) / 300))

    for item in result.objects:
        box_width = item.width * width
        box_height = item.height * height
        x_center = item.x_center * width
        y_center = item.y_center * height
        x1 = clamp(x_center - box_width / 2, 0, width - 1)
        y1 = clamp(y_center - box_height / 2, 0, height - 1)
        x2 = clamp(x_center + box_width / 2, 0, width - 1)
        y2 = clamp(y_center + box_height / 2, 0, height - 1)

        color = color_for_label(item.label)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        text = item.label
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y1 = max(0, y1 - text_height - 6)
        label_y2 = label_y1 + text_height + 6
        draw.rectangle((x1, label_y1, x1 + text_width + 8, label_y2), fill=color)
        draw.text((x1 + 4, label_y1 + 3), text, fill=(255, 255, 255), font=font)

    image.save(output_path)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    config = load_demo_config(args.config)
    if not config.data.image.exists():
        raise FileNotFoundError(f"Target image not found: {config.data.image}")

    from langchain_openai import ChatOpenAI

    class_names = [item.name for item in config.data.classes]
    agent = ChatOpenAI(
        model=config.vllm.model,
        api_key=config.vllm.api_key,
        base_url=config.vllm.base_url,
        temperature=config.vllm.temperature,
        timeout=config.vllm.request_timeout,
        max_retries=0,
    ).bind(response_format=build_response_format(LlmAnnotationResult))

    messages = [
        {"role": "system", "content": build_schema_prompt(LlmAnnotationResult)},
        {"role": "user", "content": build_user_content(config)},
    ]

    response = agent.invoke(messages)
    parsed = validate_response_content(response.content, class_names)
    visualization_path = draw_annotation_result(config.data.image, parsed)

    print("model:", config.vllm.model)
    print("image:", config.data.image)
    print("classes:", ", ".join(class_names))
    print("few_shot_examples:", len(config.data.few_shots))
    print("base_url:", config.vllm.base_url)
    print("temperature:", config.vllm.temperature)
    print("visualization:", visualization_path)
    print()
    print(response)
    print(parsed)


if __name__ == "__main__":
    main()
