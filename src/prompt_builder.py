from src.models import FunctionDefinition, PromptInput


def build_prompt(
    functions: list[FunctionDefinition],
    prompt: PromptInput,
) -> str:
    function_blocks: list[str] = []

    for function in functions:
        parameters_lines: list[str] = []
        for parameter_name, parameter_definition in (
            function.parameters.items()
        ):
            parameters_lines.append(
                f"- {parameter_name}: {parameter_definition.type}"
            )
        parameters_text = "\n\n".join(parameters_lines)

        function_block = (
            f"Function: {function.name}\n"
            f"Description: {function.description}\n"
            f"Parameters:\n{parameters_text}"
        )
        function_blocks.append(function_block)

    functions_text = "\n".join(function_blocks)

    final_prompt = (
        "You are an AI assistant that selects the correct function and "
        "extracts its parameters from a user request.\n"
        "Available functions:\n"
        f"{functions_text}\n\n"
        "User request:\n"
        f"{prompt.prompt}\n\n"
        "Return one JSON function call with exactly the keys name and "
        "parameters.\n"
        "Function call JSON:\n"
    )

    return final_prompt
