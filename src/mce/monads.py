from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

S = TypeVar("S")
V = TypeVar("V")
R = TypeVar("R")


@dataclass(frozen=True)
class AgentMonad(Generic[S, V]):
    """Monadic container for stateful, fallible agent steps."""

    state: S
    value: V | None
    is_successful: bool = True
    error_info: Any = None

    def _require_value(self) -> V:
        if self.value is None:
            raise ValueError("AgentMonad has no value.")
        return self.value

    def then(self, func: Callable[[S, V], AgentMonad[S, R]]) -> AgentMonad[S, R]:
        if not self.is_successful:
            return AgentMonad.failure(self.state, self.error_info)
        try:
            value = self._require_value()
            return func(self.state, value)
        except Exception as exc:  # pragma: no cover - defensive catch for unexpected errors
            return AgentMonad.failure(self.state, exc)

    def map(self, func: Callable[[V], R]) -> AgentMonad[S, R]:
        if not self.is_successful:
            return AgentMonad.failure(self.state, self.error_info)
        value = self._require_value()
        return AgentMonad.success(self.state, func(value))

    def apply(self, func_flow: AgentMonad[S, Callable[[V], R]]) -> AgentMonad[S, R]:
        if not self.is_successful or not func_flow.is_successful:
            error = self.error_info if not self.is_successful else func_flow.error_info
            return AgentMonad.failure(self.state, error)
        func = func_flow._require_value()
        return self.map(func)

    @staticmethod
    def start(state: S, initial_value: V | None = None) -> AgentMonad[S, V]:
        value = initial_value if initial_value is not None else cast(V, state)
        return AgentMonad(state, value)

    @staticmethod
    def success(state: S, value: V) -> AgentMonad[S, V]:
        return AgentMonad(state, value, is_successful=True)

    @staticmethod
    def failure(state: S, error_info: Any) -> AgentMonad[S, V]:
        return cast(
            AgentMonad[S, V],
            AgentMonad(state, None, is_successful=False, error_info=error_info),
        )


AsyncStep = Callable[[S, V], Awaitable[AgentMonad[S, R]]]


class AsyncAgentMonad(Generic[S, V]):
    """Async monadic container for parallel, stateful, fallible workflows."""

    def __init__(self, run_func: Callable[[], Awaitable[AgentMonad[S, V]]]) -> None:
        self._run = run_func

    async def run(self) -> AgentMonad[S, V]:
        return await self._run()

    def then(self, func: AsyncStep[S, V, R]) -> AsyncAgentMonad[S, R]:
        async def new_run() -> AgentMonad[S, R]:
            current_flow = await self.run()
            if not current_flow.is_successful:
                return AgentMonad.failure(current_flow.state, current_flow.error_info)
            try:
                value = current_flow._require_value()
                return await func(current_flow.state, value)
            except Exception as exc:  # pragma: no cover - defensive catch
                return AgentMonad.failure(current_flow.state, exc)

        return AsyncAgentMonad(new_run)

    @staticmethod
    def start(state: S, initial_value: V | None = None) -> AsyncAgentMonad[S, V]:
        async def run_func() -> AgentMonad[S, V]:
            return AgentMonad.start(state, initial_value)

        return AsyncAgentMonad(run_func)

    @staticmethod
    def gather(
        flows: Sequence[AsyncAgentMonad[S, Any]],
        merge_state: Callable[[Sequence[S]], S] | None = None,
    ) -> AsyncAgentMonad[S, list[Any]]:
        async def new_run() -> AgentMonad[S, list[Any]]:
            if not flows:
                return AgentMonad.failure(cast(S, None), "No flows provided")

            results = await asyncio.gather(*(flow.run() for flow in flows))
            errors = [result for result in results if not result.is_successful]
            if errors:
                failing = errors[0]
                return AgentMonad.failure(failing.state, failing.error_info)

            states = [result.state for result in results]
            final_state = merge_state(states) if merge_state else states[-1]
            values = [result.value for result in results]
            return AgentMonad.success(final_state, values)

        return AsyncAgentMonad(new_run)


def gather_sync(
    flows: Iterable[AgentMonad[S, V]],
    merge_state: Callable[[Sequence[S]], S] | None = None,
) -> AgentMonad[S, list[V]]:
    collected = list(flows)
    if not collected:
        return AgentMonad.failure(cast(S, None), "No flows provided")

    errors = [result for result in collected if not result.is_successful]
    if errors:
        failing = errors[0]
        return AgentMonad.failure(failing.state, failing.error_info)

    states = [result.state for result in collected]
    final_state = merge_state(states) if merge_state else states[-1]
    values = [result.value for result in collected]
    return AgentMonad.success(final_state, values)
