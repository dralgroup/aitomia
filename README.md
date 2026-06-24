# Aitomia

**Aitomia** is an AI computational chemist: an agentic framework that performs
atomistic and quantum chemical simulations from natural-language requests,
built heavily on [MLatom](https://github.com/dralgroup/mlatom). You describe the calculation you
want ("optimize this molecule with AIQM2", "calculate the IR spectrum of
hexanol"); Aitomia plans the workflow, selects the methods and programs, runs the
calculation through MLatom, and summarizes the results.

📺 **See Aitomia in action:** demonstration videos at
[mlatom.com/aitomia](http://mlatom.com/aitomia/).

This repository accompanies the paper:

> Jinming Hu, Hassan Nawaz, Yi-Fan Hou, Lijie Chi, Xin-Yu Tong, Yuting Rui,
> Yuxinxin Chen, Arif Ullah, Pavlo O. Dral. *Aitomia: An Agentic Framework for
> AI-Driven Atomistic and Quantum Chemical Simulations.* **J. Chem. Theory Comput.** 2026.
> DOI: [10.1021/acs.jctc.6c00591](https://doi.org/10.1021/acs.jctc.6c00591)
> · Preprint: [ChemRxiv](https://doi.org/10.26434/chemrxiv-2025-gnf13)
> · [arXiv:2505.08195](https://arxiv.org/abs/2505.08195)

It is released under the [Apache License 2.0](LICENSE).

> ### Want a polished, ready-to-use experience?
> Meet **[Protomia](https://aitomistic.com/protomia)** — an AI research agent for
> computational chemistry. It's an online workbench where you can ask questions,
> run calculations, monitor jobs, inspect results, and turn them into reports and
> papers, all from your browser with nothing to install. Protomia runs on the
> **[Aitomistic Hub](https://aitomistic.com/en/product/hub)** — the smoothest way
> to experience AI-driven chemistry, and we'd love for you to try it.
>
> This repository is the open-source **Aitomia** framework, which you can install
> and build on yourself.

## How it works

```
        natural-language request
                  │
                  ▼
            planner agent ──────────────► routes to a task agent
                                              │
   ┌──────────────────────────────────────────┼───────────────────────────┐
   ▼              ▼                ▼            ▼            ▼               ▼
 single-point  geometry-opt   frequency    transition-   reaction /     UV-Vis
 energy                       / IR / Raman  state / IRC   thermochem     spectra
   │              │                │            │            │               │
   └──────────────┴────────────────┴────────────┴────────────┴───────────────┘
                                  │
                                  ▼
                          MLatom calculation
                    (method + program selection,
                     molecule preparation, run,
                       result file + summary)
```

The agents are LangGraph graphs. A typical task agent: resolves the working
directory, asks the LLM to choose the method/program, prepares the molecule (from
a file or by name via PubChem), renders an MLatom Python script, executes it, and
then produces a human-readable summary.

## Capabilities

Single-point energy · geometry optimization · transition-state search · IRC ·
frequencies / thermochemistry · IR and Raman spectra · UV-Vis absorption
spectra (excited-state single-point) · reaction profiles · molecule preparation ·
chemistry Q&A — driven by natural language and powered by
[MLatom](https://github.com/dralgroup/mlatom) (AIQM methods, ML potentials, and quantum-chemistry
programs such as PySCF, xTB, and Gaussian).

## Requirements

- **Python** ≥ 3.10
- **[MLatom](https://github.com/dralgroup/mlatom)** with the backends for the methods you want
  to use (e.g. AIQM2, B3LYP via PySCF, GFN2-xTB). MLatom and its quantum-chemistry
  dependencies are the heavy part of the install — see the MLatom docs.
- **An OpenAI-compatible LLM API** (OpenAI, DeepSeek, Alibaba DashScope/Qwen, a
  local vLLM/Ollama server, …). You supply the endpoint and key.

## Installation

```bash
git clone https://github.com/dralgroup/aitomia.git
cd aitomia

# (recommended) create an environment that already has MLatom available,
# then install the agent dependencies:
pip install -e .
#   or, without installing the package:
pip install -r requirements.txt
```

See [MLatom's installation guide](https://github.com/dralgroup/mlatom) for installing the
chemistry backends required by the methods you intend to run.

## Configuration

Configuration is read from `aitomia_agents/.env`. **This file must exist with
valid values before importing or running anything** — the settings are loaded at
import time.

```bash
cp aitomia_agents/.env.example aitomia_agents/.env
# then edit aitomia_agents/.env and set OPENAI_MODEL, OPENAI_API_KEY,
# OPENAI_API_BASE to your LLM provider.
```

Key settings (full list with comments in
[`aitomia_agents/.env.example`](aitomia_agents/.env.example)):

| Variable | Meaning |
| --- | --- |
| `OPENAI_MODEL` / `OPENAI_API_KEY` / `OPENAI_API_BASE` | Your OpenAI-compatible LLM endpoint (**required**). |
| `DEV_MODE` | `True` runs calculations locally via MLatom. |
| `RESULT_DIR` | Where calculation outputs are written. |
| `MEMORY_SAVER` | Unset → in-memory chat state. `postgres` → persist to a database (needs the `postgres` extra). |

> Never commit your real `.env` — it is git-ignored. Use placeholders in any
> shared examples.

## Quick start (CLI)

```bash
python main.py          # or: aitomia   (installed console script)
```

This starts an interactive session. Example request:

```
You      > calculate IR spectrum of hexanol with AIQM2
```

The agent confirms intent, selects the method and programs, fetches the molecule
from PubChem, optimizes the geometry, computes the vibrational frequencies and IR
intensities, and plots the spectrum — printing the result paths along the way.

## Programmatic example

`run_calculation.py` drives the same planner non-interactively (auto-accepting
the agent's confirmation questions), which makes it a convenient reproducible
example:

```bash
python examples/run_calculation.py
    # default: "calculate IR spectrum of hexanol with AIQM2"

python examples/run_calculation.py "optimize the geometry of ethanol with AIQM2"
python examples/run_calculation.py "calculate the single point energy of water with AIQM2"
```

A **complete captured run** — transcript, the agent-generated MLatom scripts, the
per-step summaries, and the output IR spectrum — is in
[`docs/example_session.md`](docs/example_session.md). Sample molecules are in
[`examples/molecules/`](examples/molecules/).

> **A capable LLM is required.** The agent makes many tool-routing decisions;
> weak models mis-route (e.g. the molecule-retrieval step) and can stall. The
> paper used `qwen3-max-preview`; any sufficiently capable OpenAI-compatible
> model works.

## Citing

If you use Aitomia in your work, please cite the paper above
(DOI [10.1021/acs.jctc.6c00591](https://doi.org/10.1021/acs.jctc.6c00591)).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
