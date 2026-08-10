*This project has been created as part of the 42 curriculum by lude-jes.*

# CallMeMaybe

## Description

CallMeMaybe translates natural-language requests into structured function
calls. Given function definitions and prompts, it asks the Qwen3-0.6B model to
choose a function and extract its arguments. It does not execute the selected
function.

The main challenge is making a small language model produce dependable
machine-readable output. Instead of trusting the model to write valid JSON,
the project uses constrained decoding to permit only tokens that can still
lead to a valid function call. Input records and final results are represented
with Pydantic models.

## Instructions

### Requirements

- Python 3.10 or newer, matching both the subject and `pyproject.toml`.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- `make` when using the convenience targets.
- Internet access on the first run if Qwen3-0.6B is not already cached.

This is a Python project, so there is no separate compilation step. Install
the project and its dependencies with:

```sh
make install
```

The equivalent command is:

```sh
uv sync
```

The Makefile provides the five targets required by the subject:

| Target | Action |
| --- | --- |
| `make install` | Install and synchronize dependencies with `uv sync`. |
| `make run` | Run the program with its default file paths. |
| `make debug` | Run the program through Python's `pdb` debugger. |
| `make clean` | Remove project bytecode, mypy, and pytest caches. |
| `make lint` | Run the required flake8 and mypy checks. |

The two JSON input files must exist before running the program. They are not
generated automatically.

## Input and Output Formats

### Function definitions

The default function-definition path is
`data/input/functions_definition.json`. The file contains a JSON array. Each
function has a name, description, parameter mapping, and return definition:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

The constrained decoder supports parameter types `string`, `integer`,
`number`, and `boolean`. All declared parameters are required.

### Prompts

The default prompt path is `data/input/function_calling_tests.json`. It is a
JSON array whose objects each contain one prompt:

```json
[
  {"prompt": "What is the sum of 2 and 3?"}
]
```

### Results

The default output path is
`data/output/function_calling_results.json`. The program creates the output
directory when needed and writes one JSON array. Every result contains exactly
the original prompt, selected function name, and arguments:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Example Usage

After placing both input files at their default paths, run:

```sh
make run
```

The direct equivalent is:

```sh
uv run python -m src
```

All paths can be overridden:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/custom_results.json
```

The program loads both input files before loading the model. Missing or
malformed files, invalid schemas, model failures, decoding failures, and output
errors produce a clear error message and a non-zero exit status.

## Algorithm Explanation

For each user request, `build_prompt()` combines the available function names,
descriptions, parameter names, parameter types, and the original request. The
decoder then performs one continuous autoregressive generation:

1. The complete generation prompt is encoded once into token IDs.
2. The model receives the current token-ID sequence and returns logits for the
   next token.
3. The decoder considers every ID in the model vocabulary. Each candidate ID
   is decoded to text, and that text is cached for the rest of this prompt.
4. The candidate text is appended temporarily to the text generated so far.
   It is kept only if the result is a valid prefix of at least one permitted
   function-call JSON object. A token may contain one character, several JSON
   characters, or cross a structural boundary.
5. Among the remaining valid token IDs, the decoder greedily selects the one
   with the highest model logit. Invalid IDs are filtered out rather than
   literally having their logits changed to negative infinity.
6. The selected ID is appended to the same growing input sequence, and its
   decoded text is appended to the generated JSON. The process then repeats
   from step 2.

The only accepted generated shape is the compact form:

```text
{"name":"<defined function>","parameters":{"<parameter>":<typed value>}}
```

The prefix recognizer checks every available function definition. It enforces
the function name, parameter names, parameter order, punctuation, and value
types. Separate partial parsers handle JSON strings and escapes, booleans,
integers, and numbers with optional fractions or exponents. This allows an
unfinished prefix such as a partial string escape or exponent sign to remain
valid without accepting it as a complete value.

Generation stops as soon as the text is exactly one complete permitted call.
The result is parsed with `json.loads()` and validated again: it must have only
`name` and `parameters`, select a known function, contain every parameter in
definition order, use the expected Python types, and contain only finite
floating-point numbers. Generation fails clearly if no valid next token exists
or if the default limit of 256 generated tokens is reached.

## Design Decisions

- **Prefix validation instead of prompt-only JSON:** structure is enforced at
  every token, so the model cannot continue down an invalid JSON branch.
- **One canonical JSON layout:** fixed keys, punctuation, and parameter order
  keep the recognizer small and make completion unambiguous.
- **Greedy selection:** choosing the highest-logit allowed token makes decoding
  deterministic for a given model and prompt.
- **One model instance:** the CLI creates the model once and reuses it for all
  prompts, avoiding repeated model loading.
- **Per-prompt token cache:** each vocabulary token is decoded at most once
  during one function-call generation.
- **Validation at the boundaries:** Pydantic validates input and result records,
  while the decoder performs a final schema and type check before returning.
- **Write after complete processing:** results are collected before the output
  file is opened, so a decoding failure does not write a partial result set.

## Performance Analysis

For every successful decoder return, structural reliability is enforced by the
prefix grammar and final validation: the generated value is parseable JSON and
matches one available function schema. This does not guarantee that the model
made the semantically correct function choice or extracted the intended values;
those decisions still depend on its logits.

Loading the model once improves multi-prompt execution, and caching decoded
token text avoids repeating token-decoding work within a prompt. The main cost is
that every vocabulary token is checked at every generation step, in addition
to one model forward pass over the growing token sequence. Runtime therefore
depends on vocabulary size, output length, number and size of function schemas,
and available CPU, MPS, or CUDA hardware.

On a frozen 24-prompt Qwen3-0.6B benchmark, the implementation achieved 95.83%
function-selection accuracy, 91.11% parameter-extraction accuracy, 91.67%
exact-call accuracy, and 100% schema-valid outputs. The total runtime was
275.26 seconds, meeting the subject's 90% accuracy and five-minute targets.

## Challenges Faced

- **Tokenizer boundaries:** model tokens do not align with individual JSON
  characters. Testing the entire decoded candidate token as a continuation
  handles tokens that cross key, punctuation, and value boundaries.
- **Incomplete JSON values:** strings, escapes, fractions, and exponents have
  intermediate states that are valid prefixes but invalid completed values.
  Small state-based consumers distinguish incomplete, complete, and invalid
  states.
- **Overlapping function names:** a partial name may still match more than one
  function. Prefixes are tested against every definition until only valid
  choices remain.
- **Completion versus prefix validity:** a prefix can be extendable without
  being a complete call. Separate checks prevent early termination and reject
  trailing text after the final closing brace.

## Testing Strategy

During development, run the local test suite with:

```sh
uv run python -m pytest -q
```

The decoder tests use a fake model so token choices and token boundaries are
deterministic. They cover overlapping function names, functions without
parameters, strings and escapes, integers, fractional and exponent numbers,
booleans, wrong parameter order, invalid numeric forms, multi-character tokens,
continuous token-ID growth, token-text caching, unsupported parameter types,
duplicate function names, missing valid tokens, and the generation limit.
The suite is intentionally local and untracked, as the subject states that test
programs are not submitted or graded.

The subject-required static checks are available through:

```sh
make lint
```

This runs the repository's exact flake8 and mypy commands. Output validation
should also confirm that the result file parses as JSON, preserves every
original prompt, and matches the supplied function schemas.

## Limitations

- Parameter schemas support only `string`, `integer`, `number`, and `boolean`.
- Arrays, objects, `null`, nested arguments, and optional parameters are not
  supported.
- Every parameter is generated in the insertion order used by its function
  definition.
- Decoding is greedy and does not use sampling, batching, or beam search.
- A token-text cache is local to one prompt and is not shared between prompts.
- If one prompt fails, the run stops; the output schema has no per-prompt error
  record or recovery mechanism.
- The included default inputs are small demonstrations, not the frozen
  benchmark used for the performance measurements above.

## Resources

- [CallMeMaybe subject](subject.pdf)
- [RFC 8259: The JavaScript Object Notation Data Interchange Format](https://datatracker.ietf.org/doc/html/rfc8259)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Pydantic model documentation](https://docs.pydantic.dev/latest/concepts/models/)
- [Hugging Face text generation guide](https://huggingface.co/docs/transformers/main/en/llm_tutorial)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

### AI Usage

OpenAI Codex was used to inspect the subject requirements, assist with the CLI
integration and Makefile, draft this documentation, and run or review tests and
static checks. Its suggestions and documentation claims were checked against
the project source code, subject, and available test results. The runtime
function selection itself uses the local Qwen3-0.6B model through the provided
LLM SDK; it does not call Codex.
