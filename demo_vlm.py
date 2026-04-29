# from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from autolabeller.utils import image_to_data_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal demo for calling a multimodal Qwen3-VL model served by llama-factory."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to the local image file.")
    parser.add_argument(
        "--prompt",
        default="Please describe the main content of this image and list the key objectives you see.",
        help="User prompt sent together with the image.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful multimodal assistant.",
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--model",
        default="models/Qwen3-VL-8B-Instruct-FP8",
        help="Model name registered in llama-factory.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLAMA_FACTORY_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible llama-factory endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLAMA_FACTORY_API_KEY", "0"),
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds.",
    )
    return parser


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
                continue
            if item.get("type") == "output_text" and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def main() -> None:
    args = build_parser().parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from langchain_openai import ChatOpenAI

    client = ChatOpenAI(
        model=args.model,
        api_key=args.api_key,
        base_url=normalize_base_url(args.base_url),
        temperature=args.temperature,
        timeout=args.timeout,
    )

    messages = [
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        },
    ]

    response = client.invoke(messages)

    print("model:", args.model)
    print("image:", image_path)
    print("base_url:", normalize_base_url(args.base_url))
    print("temperature:", args.temperature)
    print()
    # print(response)
    print(extract_text(response.content))


if __name__ == "__main__":
    main()
