import argparse
import json
from pathlib import Path

from src.decoder import decode_function_call
from src.loaders import load_function_definitions, load_prompts
from src.models import FunctionCallResult
from src.prompt_builder import build_prompt


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate constrained function calls from prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the prompt input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path for the generated JSON results.",
    )
    return parser


def main() -> None:
    """Generate and write function calls for all input prompts."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)

        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

        model = Small_LLM_Model()

        results: list[FunctionCallResult] = []
        for prompt in prompts:
            generation_prompt = build_prompt(functions, prompt)
            function_call = decode_function_call(
                generation_prompt,
                functions,
                model,
            )
            results.append(
                FunctionCallResult(
                    prompt=prompt.prompt,
                    name=function_call["name"],
                    parameters=function_call["parameters"],
                )
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(
                [result.model_dump() for result in results],
                output_file,
                ensure_ascii=False,
                indent=2,
            )
            output_file.write("\n")
    except Exception as error:
        parser.exit(status=1, message=f"error: {error}\n")


if __name__ == "__main__":
    main()
