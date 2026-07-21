import argparse
from pathlib import Path

from src.loaders import load_function_definitions, load_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate constrained function calls from prompts."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    functions = load_function_definitions(args.functions_definition)
    prompts = load_prompts(args.input)

    print(f"Loaded {len(functions)} functions.")
    print(f"Loaded {len(prompts)} prompts.")


if __name__ == "__main__":
    main()