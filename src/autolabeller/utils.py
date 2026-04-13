from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
