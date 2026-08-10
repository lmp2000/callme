from typing import Any


from pydantic import BaseModel


class ParameterDefinition(BaseModel):
    """Define the JSON type of one function parameter."""

    type: str


class ReturnDefinition(BaseModel):
    """Define the JSON type returned by a function."""

    type: str


class FunctionDefinition(BaseModel):
    """Describe one function available for model selection."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Represent one validated function-call result."""

    prompt: str
    name: str
    parameters: dict[str, Any]
