"""Entry point for running sigma as a module."""

from __future__ import annotations

import sys

from . import cli

if __name__ == "__main__":
    status = cli.run()
    sys.exit(status)
