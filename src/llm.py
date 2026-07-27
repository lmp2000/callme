from llm_sdk import Small_LLM_Model


model = Small_LLM_Model()


def get_next_token_logits(prompt: str) -> list[float]:

    encoded = model.encode(prompt)
    input_ids = encoded[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)
    return logits