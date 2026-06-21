"""Run an Aitomia calculation from a natural-language request (non-interactive).

This drives the same planner agent as the interactive CLI (``python main.py``),
but feeds a single request and automatically accepts the agent's clarifying /
confirmation questions, so it runs unattended. Handy as a reproducible example
or a smoke test.

Usage (from anywhere; no install required):

    python examples/run_calculation.py
        -> default: "calculate IR spectrum of hexanol with AIQM2"

    python examples/run_calculation.py "optimize the geometry of ethanol with AIQM2"
    python examples/run_calculation.py "calculate the single point energy of water with AIQM2"

Prerequisites:
  * aitomia_agents/.env configured with a capable LLM. The paper used
    qwen3-max-preview; any sufficiently capable OpenAI-compatible model works.
    Weak models tend to mis-route the tool calls.
  * MLatom installed with support for the requested method (AIQM2 here).

A complete captured run (transcript + generated scripts + output spectrum) is in
``docs/example_session/``.
"""

import os
import sys

# Make `aitomia_agents` importable when running this file directly, without
# needing `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from aitomia_agents import Aitomia

DEFAULT_REQUEST = "calculate IR spectrum of hexanol with AIQM2"
MAX_CONFIRMATIONS = 15


def _show(role: str, text: str) -> None:
    print("\n" + "=" * 72 + f"\n{role}\n" + "=" * 72, flush=True)
    print(text, flush=True)


def main() -> None:
    request = " ".join(sys.argv[1:]).strip() or DEFAULT_REQUEST

    aitomia = asyncio.run(Aitomia.create(agent="planner_agent"))

    _show("USER", request)
    response = aitomia.chat(messages=request)
    _show("AITOMIA", response)

    # The planner asks a couple of confirmation questions (consent + the chosen
    # method/program). Auto-accept them so the example runs unattended.
    for _ in range(MAX_CONFIRMATIONS):
        if not getattr(aitomia.agent, "_interrupt", False):
            break
        _show("USER", "yes")
        response = aitomia.chat(messages="yes")
        _show("AITOMIA", response)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
