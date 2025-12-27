from mce.steps import run_simple_agent


def test_run_simple_agent() -> None:
    flow = run_simple_agent("Explain monads")
    assert flow.is_successful is True
    assert flow.value is not None
    assert "Final Report" in flow.value
    assert flow.state.history
