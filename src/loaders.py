import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    try:
        return [
            FunctionDefinition.model_validate(function)
            for function in data
        ]
    except ValidationError as error:
        raise ValueError("Invalid function definition file.") from error


def load_prompts(path: Path) -> list[PromptInput]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    try:
        return [
            PromptInput.model_validate(prompt) for prompt in data
        ]
    except ValidationError as error:
        raise ValueError("Invalid prompt input file.") from error
