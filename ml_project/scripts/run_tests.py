from __future__ import annotations

"""Run pytest suite."""

import os
import subprocess
import sys


def main() -> None:
    """Execute pytest with the project on PYTHONPATH."""
    env = dict(os.environ, PYTHONPATH=".")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"], env=env, check=False
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
