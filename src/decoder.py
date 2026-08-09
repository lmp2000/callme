from enum import Enum


from src.models import FunctionDefinition, ParameterDefinition
from llm_sdk import Small_LLM_Model


def generate_from_options(
    prompt: str,
    options: list[str],
    model: Small_LLM_Model,
) -> str:
    input_ids = model.encode(prompt)[0].tolist()
    current_text = ""
    while current_text not in options:
        allowed_ids = []
        logits = model.get_logits_from_input_ids(input_ids)
        for token_id in range(len(logits)):
            candidate = model.decode([token_id])
            candidate_text = current_text + candidate
            for option in options:
                if option.startswith(candidate_text):
                    allowed_ids.append(token_id)
        token_chosen = max(allowed_ids, key=lambda token_id: logits[token_id])
        input_ids.append(token_chosen)
        current_text += model.decode([token_chosen])
    return current_text


def select_function_name(
    user_prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
) -> str:
    functions_names = [
        function.name for function in functions
    ]
    functions_text = "\n".join(
        f"{function.name}: {function.description}"
        for function in functions
    )
    prompt = (
        "Choose the function that best matches the user's request.\n"
        "You must choose one of the available functions.\n\n"
        f"Available functions:\n{functions_text}\n\n"
        f"User request: {user_prompt}\n\n"
        "Function: "
    )
    return generate_from_options(prompt, functions_names, model)


def extract_parameters(
    user_prompt: str,
    function: FunctionDefinition,
    model: Small_LLM_Model,
) -> dict:
    parameters = {}
    for param_name, param_def in function.parameters.items():
        value = extract_parameter_value(
            user_prompt,
            function,
            param_name,
            param_def,
            model
        )
        parameters[param_name] = value
    return parameters

def extract_parameter_value(
        user_prompt: str,
        function: FunctionDefinition,
        param_name: str,
        param_def: ParameterDefinition,
        model: Small_LLM_Model,
    ):
        if param_def.type == "boolean":
            prompt = (
                f"The selected function is '{function.name}'.\n"
                f"Function description: {function.description}\n"
                f"Extract the value for parameter '{param_name}'.\n"
                f"Expected type: boolean.\n"
                f"User request: {user_prompt}\n"
                f"{param_name}: "
            )

            value = generate_from_options(
                prompt,
                ["true", "false"],
                model,
            )

            return value == "true"
                



class DecoderState(Enum):
    FIXED_PREFIX = "fixed_prefix"
    FUNCTION_NAME = "function_name"
    FIXED_PARAMETERS_PREFIX = "fixed_parameters_prefix"
    PARAMETERS = "parameters"
    FIXED_SUFFIX = "fixed_suffix"
    DONE = "done"


def get_allowed_token_ids(
    state: DecoderState,
    functions: list[FunctionDefinition],
    state_token_ids: list[int],
    model: Small_LLM_Model
) -> list[int]:

    if state == DecoderState.FIXED_PREFIX:
        target = '{"name":"'
        target_tokens = model.encode(target)[0].tolist()
        next_index = len(state_token_ids)
        if next_index < len(target_tokens):
            return [target_tokens[next_index]]
        else:
            return []

    if state == DecoderState.FUNCTION_NAME:
        names = [
            function.name for function in functions
        ]
        names_ids = [
            model.encode(name)[0].tolist() for name in names
        ]
        next_index = len(state_token_ids)
        valid_names_ids = []
        for ids in names_ids:
            candidate = ids[:next_index]
            if candidate == state_token_ids:
                valid_names_ids.append(ids)
        allowed_tokens = []
        for ids in valid_names_ids:
            if next_index == len(ids):
                allowed_tokens.append(
                    model.encode('"')[0].tolist()[0]
                )
            elif next_index < len(ids):
                allowed_tokens.append(ids[next_index])
        return allowed_tokens