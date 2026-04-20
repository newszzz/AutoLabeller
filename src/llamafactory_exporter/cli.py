from __future__ import annotations

import argparse

from .config import load_config
from .dataset import export_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO to LLaMA-Factory dataset exporter")
    parser.add_argument("--config", required=True, help="Path to exporter YAML config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    summary = export_dataset(config)
    print(summary)
