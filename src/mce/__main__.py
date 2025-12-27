from __future__ import annotations

from .steps import run_simple_agent


def main() -> None:
    flow = run_simple_agent("What is a Monad?")
    if flow.is_successful:
        print(flow.value)
        print("\nExecution history:")
        for entry in flow.state.history:
            print(f"- {entry}")
    else:
        print(f"Flow failed: {flow.error_info}")


if __name__ == "__main__":
    main()
