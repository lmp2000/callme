"""Constrained decoding for schema-valid function calls."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from src.models import FunctionDefinition

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


SUPPORTED_PARAMETER_TYPES = {"string", "integer", "number", "boolean"}


@dataclass(frozen=True)
class _CompiledParameter:
    """Precomputed fixed prefix and type for one parameter."""

    prefix: str
    type: str


@dataclass(frozen=True)
class _CompiledFunction:
    """Fixed JSON fragments needed to match one function schema."""

    opening: str
    parameters: tuple[_CompiledParameter, ...]


_FunctionSignature = tuple[
    tuple[str, tuple[tuple[str, str], ...]],
    ...,
]


@lru_cache(maxsize=32)
def _compile_function_signature(
    signature: _FunctionSignature,
) -> tuple[_CompiledFunction, ...]:
    """Compile an immutable function signature into reusable literals."""
    compiled_functions: list[_CompiledFunction] = []

    for function_name, parameters in signature:
        compiled_parameters = tuple(
            _CompiledParameter(
                prefix=("," if index else "")
                + json.dumps(parameter_name)
                + ":",
                type=parameter_type,
            )
            for index, (parameter_name, parameter_type) in enumerate(
                parameters
            )
        )
        compiled_functions.append(
            _CompiledFunction(
                opening=(
                    '{"name":'
                    f'{json.dumps(function_name)},'
                    '"parameters":{'
                ),
                parameters=compiled_parameters,
            )
        )

    return tuple(compiled_functions)


def _compile_functions(
    functions: list[FunctionDefinition],
) -> tuple[_CompiledFunction, ...]:
    """Return cached fixed JSON fragments for function definitions."""
    signature: _FunctionSignature = tuple(
        (
            function.name,
            tuple(
                (parameter_name, definition.type)
                for parameter_name, definition in function.parameters.items()
            ),
        )
        for function in functions
    )
    return _compile_function_signature(signature)


def _consume_literal(
    text: str,
    position: int,
    literal: str,
) -> tuple[int, bool] | None:
    """Consume a fixed literal, while accepting an unfinished prefix."""
    remaining = text[position:]
    if literal.startswith(remaining):
        return len(text), len(remaining) == len(literal)
    if remaining.startswith(literal):
        return position + len(literal), True
    return None


def _consume_string(text: str, position: int) -> tuple[int, bool] | None:
    """Consume one JSON string value."""
    if position == len(text):
        return position, False
    if text[position] != '"':
        return None

    index = position + 1
    while index < len(text):
        character = text[index]
        if character == '"':
            return index + 1, True
        if ord(character) < 0x20:
            return None
        if character != "\\":
            index += 1
            continue

        index += 1
        if index == len(text):
            return index, False
        escape = text[index]
        if escape in '"\\/bfnrt':
            index += 1
            continue
        if escape != "u":
            return None

        digits = text[index + 1:index + 5]
        if any(
            character not in "0123456789abcdefABCDEF"
            for character in digits
        ):
            return None
        if len(digits) < 4:
            return len(text), False
        index += 5

    return len(text), False


def _consume_boolean(text: str, position: int) -> tuple[int, bool] | None:
    """Consume a JSON boolean value."""
    for literal in ("true", "false"):
        result = _consume_literal(text, position, literal)
        if result is not None:
            return result
    return None


def _consume_number(
    text: str,
    position: int,
    allow_fraction: bool,
) -> tuple[int, bool] | None:
    """Consume a JSON integer or number with partial-state validation."""
    state = "start"
    index = position
    complete_states = {"zero", "integer", "fraction", "exponent_digits"}

    while index < len(text):
        character = text[index]

        if state in {"start", "sign"}:
            if state == "start" and character == "-":
                state = "sign"
            elif character == "0":
                state = "zero"
            elif character in "123456789":
                state = "integer"
            else:
                return None
        elif state == "zero":
            if allow_fraction and character == ".":
                state = "decimal_point"
            elif allow_fraction and character in "eE":
                state = "exponent"
            else:
                return index, True
        elif state == "integer":
            if character in "0123456789":
                pass
            elif allow_fraction and character == ".":
                state = "decimal_point"
            elif allow_fraction and character in "eE":
                state = "exponent"
            else:
                return index, True
        elif state == "decimal_point":
            if character in "0123456789":
                state = "fraction"
            else:
                return None
        elif state == "fraction":
            if character in "0123456789":
                pass
            elif character in "eE":
                state = "exponent"
            else:
                return index, True
        elif state == "exponent":
            if character in "+-":
                state = "exponent_sign"
            elif character in "0123456789":
                state = "exponent_digits"
            else:
                return None
        elif state == "exponent_sign":
            if character in "0123456789":
                state = "exponent_digits"
            else:
                return None
        elif state == "exponent_digits":
            if character not in "0123456789":
                return index, True

        index += 1

    return index, state in complete_states


def _consume_value(
    text: str,
    position: int,
    parameter_type: str,
) -> tuple[int, bool] | None:
    if parameter_type == "string":
        return _consume_string(text, position)
    if parameter_type == "boolean":
        return _consume_boolean(text, position)
    if parameter_type == "integer":
        return _consume_number(text, position, allow_fraction=False)
    if parameter_type == "number":
        return _consume_number(text, position, allow_fraction=True)
    return None


def _match_function_prefix(
    text: str,
    function: _CompiledFunction,
) -> tuple[bool, bool]:
    """Return whether text is a valid prefix and whether it is complete."""
    position = 0
    result = _consume_literal(text, position, function.opening)
    if result is None:
        return False, False
    position, complete = result
    if not complete:
        return True, False

    for parameter in function.parameters:
        result = _consume_literal(text, position, parameter.prefix)
        if result is None:
            return False, False
        position, complete = result
        if not complete:
            return True, False

        result = _consume_value(text, position, parameter.type)
        if result is None:
            return False, False
        position, complete = result
        if not complete:
            return True, False

    result = _consume_literal(text, position, "}}")
    if result is None:
        return False, False
    position, complete = result
    if not complete:
        return True, False
    return position == len(text), position == len(text)


def is_valid_prefix(
    text: str,
    functions: list[FunctionDefinition],
) -> bool:
    """Return whether text can become a valid call for any function."""
    compiled_functions = _compile_functions(functions)
    return any(
        _match_function_prefix(text, function)[0]
        for function in compiled_functions
    )


def is_complete_call(
    text: str,
    functions: list[FunctionDefinition],
) -> bool:
    """Return whether text is exactly one complete schema-valid call."""
    compiled_functions = _compile_functions(functions)
    return any(
        _match_function_prefix(text, function)[1]
        for function in compiled_functions
    )


def _get_allowed_token_ids(
    functions: tuple[_CompiledFunction, ...],
    generated_text: str,
    model: Small_LLM_Model,
    vocabulary_size: int,
    token_text_cache: dict[int, str],
) -> list[int]:
    """Return allowed IDs using already-compiled viable schemas."""
    allowed_ids: list[int] = []

    for token_id in range(vocabulary_size):
        if token_id not in token_text_cache:
            token_text_cache[token_id] = model.decode([token_id])
        token_text = token_text_cache[token_id]
        if token_text and any(
            _match_function_prefix(
                generated_text + token_text,
                function,
            )[0]
            for function in functions
        ):
            allowed_ids.append(token_id)

    return allowed_ids


def get_allowed_token_ids(
    functions: list[FunctionDefinition],
    generated_text: str,
    model: Small_LLM_Model,
    vocabulary_size: int,
    token_text_cache: dict[int, str] | None = None,
) -> list[int]:
    """Return token IDs whose decoded text preserves a valid prefix."""
    cache = token_text_cache if token_text_cache is not None else {}
    compiled_functions = tuple(
        function
        for function in _compile_functions(functions)
        if _match_function_prefix(generated_text, function)[0]
    )
    return _get_allowed_token_ids(
        compiled_functions,
        generated_text,
        model,
        vocabulary_size,
        cache,
    )


def _validate_functions(functions: list[FunctionDefinition]) -> None:
    if not functions:
        raise ValueError("At least one function definition is required.")

    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise ValueError("Function names must be unique.")

    for function in functions:
        for name, definition in function.parameters.items():
            if definition.type not in SUPPORTED_PARAMETER_TYPES:
                raise ValueError(
                    f"Unsupported type {definition.type!r} for parameter "
                    f"{name!r} in function {function.name!r}."
                )


def _validate_decoded_call(
    call: dict[str, Any],
    functions: list[FunctionDefinition],
) -> None:
    if list(call) != ["name", "parameters"]:
        raise RuntimeError("Decoded call has invalid top-level keys.")

    selected = next(
        (function for function in functions if function.name == call["name"]),
        None,
    )
    if selected is None:
        raise RuntimeError("Decoded call selected an unknown function.")

    parameters = call["parameters"]
    if not isinstance(parameters, dict):
        raise RuntimeError("Decoded parameters must be a JSON object.")
    if list(parameters) != list(selected.parameters):
        raise RuntimeError(
            "Decoded parameters do not match the function schema."
        )

    expected_python_types = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
    }
    for name, definition in selected.parameters.items():
        value = parameters[name]
        if type(value) not in expected_python_types[definition.type]:
            raise RuntimeError(
                f"Decoded parameter {name!r} has the wrong type."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(
                f"Decoded parameter {name!r} is not a finite number."
            )


def decode_function_call(
    prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Generate one constrained function-call JSON object."""
    _validate_functions(functions)
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")

    input_ids = model.encode(prompt)[0].tolist()
    if not input_ids:
        raise ValueError("The prompt must encode to at least one token.")

    generated_text = ""
    token_text_cache: dict[int, str] = {}
    viable_functions = _compile_functions(functions)

    for _ in range(max_new_tokens):
        viable_functions = tuple(
            function
            for function in viable_functions
            if _match_function_prefix(generated_text, function)[0]
        )
        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids = _get_allowed_token_ids(
            viable_functions,
            generated_text,
            model,
            len(logits),
            token_text_cache,
        )
        if not allowed_ids:
            raise RuntimeError(
                "Constrained decoding found no valid next token."
            )

        chosen_id = max(allowed_ids, key=lambda token_id: logits[token_id])
        input_ids.append(chosen_id)
        generated_text += token_text_cache[chosen_id]

        if any(
            _match_function_prefix(generated_text, function)[1]
            for function in viable_functions
        ):
            decoded: Any = json.loads(generated_text)
            if not isinstance(decoded, dict):
                raise RuntimeError(
                    "Decoded function call is not a JSON object."
                )
            call = cast(dict[str, Any], decoded)
            _validate_decoded_call(call, functions)
            return call

    raise RuntimeError(
        f"Constrained decoding exceeded {max_new_tokens} generated tokens."
    )
