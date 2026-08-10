"""Helpers for explicit LLM inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def get_next_token_logits(
    prompt: str,
    model: Small_LLM_Model,
) -> list[float]:
    """Encode a prompt and return the model's next-token logits."""
    encoded = model.encode(prompt)
    input_ids: list[int] = encoded[0].tolist()
    logits: list[float] = model.get_logits_from_input_ids(input_ids)
    return logits
