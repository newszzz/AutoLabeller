from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

from langchain_ollama import ChatOllama
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """单个目标的归一化框标注结果。"""

    label: str = Field(
        description=(
            "目标类别名称。应填写图片中被框选物体的语义标签，例如 "
            "'person'、'car'、'dog'。不要填写坐标、序号或解释。"
        )
    )
    x_center: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "边界框中心点在图像水平方向上的归一化坐标。"
            "取值范围为 0 到 1，其中 0 表示最左侧，1 表示最右侧。"
        ),
    )
    y_center: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "边界框中心点在图像垂直方向上的归一化坐标。"
            "取值范围为 0 到 1，其中 0 表示最上方，1 表示最下方。"
        ),
    )
    width: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "边界框宽度占整张图片宽度的比例。"
            "必须大于 0 且不超过 1。"
        ),
    )
    height: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "边界框高度占整张图片高度的比例。"
            "必须大于 0 且不超过 1。"
        ),
    )


class AnnotationResult(BaseModel):
    """整张图的结构化标注结果。"""

    boxes: list[BoundingBox] = Field(
        default_factory=list,
        description=(
            "图片中所有需要标注的目标框列表。"
            "每个元素表示一个独立目标；如果没有发现符合要求的目标，可返回空列表。"
        ),
    )
    summary: str = Field(
        description=(
            "对整张图片标注结果的简短总结。"
            "应概括图片主要内容、检测到的关键目标，以及必要时说明未发现目标。"
        )
    )


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_system_prompt() -> str:
    return """
你是一个图像标注助手。
你需要理解图片内容，并根据用户给出的标注要求生成结构化标注结果。
不要输出任何思考过程（thinking）。
""".strip()


def build_user_prompt(task_prompt: str) -> str:
    return f"""
标注要求：
{task_prompt}

输出要求：
1. 仅输出符合给定结构体的数据，不要输出额外解释。
2. 所有坐标都必须是归一化坐标，范围在 0 到 1 之间。
3. x_center 和 y_center 表示框中心点；width 和 height 表示框的宽高比例。
4. boxes 中每个元素对应图片中的一个独立目标，不要重复标注同一目标。
5. summary 用一两句话总结图片内容和标注情况。
6. 如果图片里没有符合要求的目标，boxes 返回空列表，并在 summary 中明确说明。
""".strip()


def annotate_image(
    image_path: Path,
    task_prompt: str,
    model_name: str,
    base_url: str,
    temperature: float,
) -> AnnotationResult:
    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    ).with_structured_output(AnnotationResult)

    messages = [
        ("system", build_system_prompt()),
        (
            "user",
            [
                {"type": "text", "text": build_user_prompt(task_prompt)},
                {"type": "image_url", "image_url": image_to_data_url(image_path)},
            ],
        ),
    ]

    return llm.invoke(messages)


def draw_annotations(image_path: Path, result: AnnotationResult, output_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    for box in result.boxes:
        left = (box.x_center - box.width / 2) * width
        top = (box.y_center - box.height / 2) * height
        right = (box.x_center + box.width / 2) * width
        bottom = (box.y_center + box.height / 2) * height

        left = max(0, min(width, left))
        top = max(0, min(height, top))
        right = max(0, min(width, right))
        bottom = max(0, min(height, bottom))

        draw.rectangle((left, top, right, bottom), outline="red", width=3)

        label_text = box.label
        text_bbox = draw.textbbox((left, top), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_top = max(0, top - text_height - 6)
        text_right = min(width, left + text_width + 8)

        draw.rectangle((left, text_top, text_right, text_top + text_height + 6), fill="red")
        draw.text((left + 4, text_top + 3), label_text, fill="white", font=font)

    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LangChain + Pydantic 调用多模态模型，输出结构化标注结果。"
    )
    parser.add_argument("--image", type=Path, required=True, help="输入图片路径")
    parser.add_argument(
        "--prompt",
        required=True,
        help="给多模态模型的标注要求，例如：标出所有猫和狗",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:9b-q8_0",
        help="支持视觉输入的 Ollama 模型名称",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama 服务地址",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="模型温度，demo 默认使用 0 提高稳定性",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="绘制后的输出图片路径，默认保存在输入图片同目录下",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="绘制完成后尝试在本地直接打开结果图片",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = annotate_image(
        image_path=args.image,
        task_prompt=args.prompt,
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    output_path = args.output or args.image.with_name(f"{args.image.stem}_annotated{args.image.suffix}")
    rendered_path = draw_annotations(args.image, result, output_path)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print(f"\nAnnotated image saved to: {rendered_path}")
    if args.show:
        Image.open(rendered_path).show()


if __name__ == "__main__":
    main()
