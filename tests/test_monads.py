import asyncio

from mce.monads import AgentMonad, AsyncAgentMonad


def test_then_success() -> None:
    flow = AgentMonad.start("state", "start").then(lambda s, v: AgentMonad.success(s, v + "-next"))
    assert flow.is_successful is True
    assert flow.value == "start-next"


def test_then_failure_short_circuit() -> None:
    flow = AgentMonad.failure("state", "error").then(lambda s, v: AgentMonad.success(s, v))
    assert flow.is_successful is False
    assert flow.error_info == "error"


def test_map_apply() -> None:
    base = AgentMonad.start("state", 2)
    mapped = base.map(lambda v: v + 1)
    assert mapped.value == 3

    func_flow = AgentMonad.start("state", lambda v: v * 5)
    applied = base.apply(func_flow)
    assert applied.value == 10


def test_async_gather() -> None:
    async def step(value: int) -> AgentMonad[str, int]:
        await asyncio.sleep(0)
        return AgentMonad.success("state", value * 2)

    async def run_flow() -> AgentMonad[str, list[int]]:
        flow_a = AsyncAgentMonad.start("state", 2).then(lambda s, v: step(v))
        flow_b = AsyncAgentMonad.start("state", 3).then(lambda s, v: step(v))
        gathered = AsyncAgentMonad.gather([flow_a, flow_b])
        return await gathered.run()

    result = asyncio.run(run_flow())
    assert result.is_successful is True
    assert result.value == [4, 6]
