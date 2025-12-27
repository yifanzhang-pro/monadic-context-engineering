# Monadic Context Engineering

[![Website](https://img.shields.io/badge/Project-Website-green)](https://yifanzhang-pro.github.io/monadic-context-engineering)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

Monadic Context Engineering (MCE) is a principled architecture for AI agent orchestration that treats workflows as composable computations in a shared context. It formalizes how state, errors, and side effects propagate through agent steps, using the algebraic structures of Functors, Applicatives, and Monads.

This repository provides a Python implementation aligned with the paper in `docs/paper.tex`, including:

- `AgentMonad`: sequential, stateful, fallible computation chains.
- `AsyncAgentMonad`: async flows with Applicative parallelism via `gather`.
- Pydantic models for structured state and tool calls/results.

## Core Idea

MCE models an agent workflow as a single container that carries:

- State (memory/history)
- A value (current step output)
- A success/failure signal

The `.then()` operator composes steps. If any step fails, the chain short-circuits, preserving the error and state at the point of failure.

## Project Layout

- `src/mce/monads.py`: `AgentMonad` and `AsyncAgentMonad`.
- `src/mce/models.py`: pydantic models and a tool registry.
- `src/mce/steps.py`: reference steps from the paper (plan, execute, synthesize, format).
- `src/mce/__main__.py`: a runnable demo.
- `docs/paper.tex`: the research paper.
- `docs/index.html`: project page.

## Quickstart (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run the demo:

```bash
python -m mce
```

## Example Usage

```python
from mce.steps import run_simple_agent

flow = run_simple_agent("What is a Monad?")
if flow.is_successful:
    print(flow.value)
    for entry in flow.state.history:
        print("-", entry)
else:
    print("Failure:", flow.error_info)
```

## Tooling

- Type checking: `pyright`
- Linting: `ruff`
- Tests: `pytest`

```bash
pyright
ruff check src tests
pytest
```

## Notes

- The implementation follows the conceptual design in the paper, including the `AgentMonad` and `AsyncAgentMonad` APIs.
- The example steps are deterministic and self-contained; they do not call external APIs.
