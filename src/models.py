from pydantic import BaseModel
from typing import Any

class ParameterDefinition(BaseModel):
    type: str


class ReturnDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptInput(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]