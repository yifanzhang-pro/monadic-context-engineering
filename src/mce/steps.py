from __future__ import annotations

from .llm import OpenRouterClient
from .models import AgentState, ToolCall, ToolRegistry, default_registry
from .monads import AgentMonad


def plan_action(state: AgentState, task: str) -> AgentMonad[AgentState, ToolCall]:
    call = ToolCall(tool_id="tool-1", name="search", arguments={"query": task})
    next_state = state.with_history(f"Plan: call {call.name} with query='{task}'.")
    return AgentMonad.success(next_state, call)


def execute_tool(
    state: AgentState, call: ToolCall, registry: ToolRegistry | None = None
) -> AgentMonad[AgentState, str]:
    registry = registry or default_registry()
    result = registry.run(state, call)
    next_state = state.with_history(f"Tool Result ({call.name}): {result.content}")
    if result.is_error:
        return AgentMonad.failure(next_state, result.content)
    return AgentMonad.success(next_state, result.content)


def synthesize_answer(state: AgentState, tool_output: str) -> AgentMonad[AgentState, str]:
    answer = (
        "Monadic Context Engineering structures agent workflows as composable steps "
        "with built-in state threading, error short-circuiting, and optional parallelism. "
        f"Evidence: {tool_output}"
    )
    next_state = state.with_history("Synthesized final answer.")
    return AgentMonad.success(next_state, answer)


def synthesize_answer_openrouter(
    state: AgentState,
    tool_output: str,
    client: OpenRouterClient | None = None,
) -> AgentMonad[AgentState, str]:
    client = client or OpenRouterClient.from_env()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {state.task}\n\n"
                f"Tool evidence:\n{tool_output}\n\n"
                "Write the final answer in 3-6 sentences."
            ),
        },
    ]
    answer = client.chat(messages)
    next_state = state.with_history("Synthesized final answer with OpenRouter.")
    return AgentMonad.success(next_state, answer)


def format_output(state: AgentState, answer: str) -> AgentMonad[AgentState, str]:
    formatted = f"Final Report:\n{answer}"
    next_state = state.with_history("Formatted response for delivery.")
    return AgentMonad.success(next_state, formatted)


def run_simple_agent(task: str) -> AgentMonad[AgentState, str]:
    initial_state = AgentState(task=task)
    return (
        AgentMonad.start(initial_state)
        .then(lambda s, _: plan_action(s, task))
        .then(lambda s, call: execute_tool(s, call))
        .then(synthesize_answer)
        .then(format_output)
    )


def run_openrouter_agent(
    task: str, client: OpenRouterClient | None = None
) -> AgentMonad[AgentState, str]:
    initial_state = AgentState(task=task)
    return (
        AgentMonad.start(initial_state)
        .then(lambda s, _: plan_action(s, task))
        .then(lambda s, call: execute_tool(s, call))
        .then(lambda s, output: synthesize_answer_openrouter(s, output, client=client))
        .then(format_output)
    )
