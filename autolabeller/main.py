from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import AutoLabelPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoLabeller CLI")
    parser.add_argument("config", help="Path to YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    summary = AutoLabelPipeline(config).run()
    print(summary)


if __name__ == "__main__":
    main()
