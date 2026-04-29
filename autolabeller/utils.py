from __future__ import annotations

import base64
import colorsys
import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .schemas import AnnotationResult


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def save_annotation_json(
    image_path: Path,
    result: AnnotationResult,
    labels_dir: Path,
    images_dir: Path,
) -> Path:
    output_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".json")
    write_json(output_path, result)
    return output_path


def copy_image_as(image_path: Path, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    shutil.copy2(image_path, output_path)
    return output_path


def build_label_color_map(labels: list[str]) -> dict[str, tuple[int, int, int]]:
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (255, 225, 25),
        (0, 128, 128),
        (128, 0, 0),
        (0, 0, 128),
        (170, 110, 40),
        (128, 128, 0),
        (255, 0, 127),
        (170, 0, 255),
        (0, 170, 255),
    ]
    colors: dict[str, tuple[int, int, int]] = {}
    for index, label in enumerate(labels):
        if index < len(palette):
            colors[label] = palette[index]
            continue
        hue = (index * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
        colors[label] = (round(red * 255), round(green * 255), round(blue * 255))
    return colors


def format_color_legend(color_by_label: dict[str, tuple[int, int, int]]) -> str:
    return "\n".join(
        f"- {label}: {_color_name(color)} rectangle outline, RGB{color}, {_rgb_to_hex(color)}"
        for label, color in color_by_label.items()
    )


def render_annotation_image(
    image_path: Path,
    result: AnnotationResult,
    output_path: Path,
    color_by_label: dict[str, tuple[int, int, int]] | None = None,
) -> Path:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(1, round(min(width, height) / 500))
    colors = color_by_label or build_label_color_map(sorted({item.label for item in result.objects}))

    for item in result.objects:
        x1 = _clamp(item.x_min, 0, width - 1)
        y1 = _clamp(item.y_min, 0, height - 1)
        x2 = _clamp(item.x_max, 0, width - 1)
        y2 = _clamp(item.y_max, 0, height - 1)
        color = colors[item.label]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    ensure_dir(output_path.parent)
    image.save(output_path)
    return output_path


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def _color_name(color: tuple[int, int, int]) -> str:
    names = {
        (255, 0, 0): "pure red",
        (0, 255, 0): "pure green",
        (0, 0, 255): "pure blue",
    }
    return names.get(color, "distinct colored")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
