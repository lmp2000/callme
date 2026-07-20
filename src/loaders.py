import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput