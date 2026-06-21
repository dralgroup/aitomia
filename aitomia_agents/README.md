# aitomia_agents

The Python package implementing the Aitomia agents — LangGraph graphs for
single-point energies, geometry optimization, frequencies / IR / Raman, transition
states, IRC, reaction profiles, UV-Vis absorption spectra, molecule preparation,
and chemistry Q&A — coordinated by a planner agent.

Configuration is read from `aitomia_agents/.env` **at import time**, so copy
`aitomia_agents/.env.example` to `aitomia_agents/.env` and set your LLM
credentials before importing or running anything.

See the top-level [README](../README.md) for installation, configuration, and a
complete worked example.

## Running

- **Interactive CLI:** `python main.py` (or the `aitomia` console script), then
  type a request such as `calculate IR spectrum of hexanol with AIQM2`.
- **Non-interactive:** `python examples/run_calculation.py "<your request>"`.

## Programmatic use

```python
import asyncio
from aitomia_agents import Aitomia

# Aitomia.create() is async only because it may set up a database checkpointer;
# in the default in-memory mode it returns immediately.
aitomia = asyncio.run(Aitomia.create(agent="planner_agent"))

# Send a natural-language request. The planner asks a couple of confirmation
# questions before running, so resume with aitomia.chat(messages="yes") while
# aitomia.agent._interrupt is set — see examples/run_calculation.py for the loop.
response = aitomia.chat(messages="calculate IR spectrum of hexanol with AIQM2")
print(response)
```

A full captured run (transcript, generated MLatom scripts, and the output
spectrum) is in [docs/example_session.md](../docs/example_session.md).
