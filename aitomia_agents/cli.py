"""Command-line entry point for the Aitomia agent.

Starts an interactive terminal chat session backed by the planner agent, which
routes requests to the appropriate task agent (single-point energy, geometry
optimization, frequencies, spectra, reactions, ...).

Configuration is read from ``aitomia_agents/.env`` — copy
``aitomia_agents/.env.example`` to that path and fill in your LLM credentials
before running.
"""

import asyncio

from aitomia_agents import Aitomia


def main() -> None:
    # ``Aitomia.create()`` is a coroutine only because it may set up a
    # Postgres-backed checkpointer. In the default in-memory mode it completes
    # immediately, so we run it once here and then hand off to the synchronous
    # REPL loop.
    aitomia = asyncio.run(Aitomia.create(agent="planner_agent"))
    print("Aitomia — AI computational chemist.")
    print("Type your request in natural language. Type 'bye' or 'exit' to quit.\n")
    aitomia.chat_cmd()


if __name__ == "__main__":
    main()
