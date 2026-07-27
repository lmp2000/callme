from enum import Enum


from src.models import FunctionDefinition
from llm_sdk import Small_LLM_Model


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
    generated_text: str,
    model: Small_LLM_Model
) -> list[int]:

    if state == DecoderState.FIXED_PREFIX:
        pass