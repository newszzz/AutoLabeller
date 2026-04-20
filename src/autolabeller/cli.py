from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import AutoLabelPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoLabeller CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate_parser = subparsers.add_parser("annotate", help="Run annotation pipeline")
    annotate_parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "annotate":
        summary = AutoLabelPipeline(config).run()
        print(summary)
        return

    raise ValueError(f"Unsupported command: {args.command}")
