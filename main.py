#!/usr/bin/env python
"""Convenience launcher for the Aitomia CLI.

Equivalent to running ``aitomia`` (the installed console script) or
``python -m aitomia_agents.cli``.

    python main.py
"""

from aitomia_agents.cli import main

if __name__ == "__main__":
    main()
