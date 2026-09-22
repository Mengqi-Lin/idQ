"""Locations for experiment outputs, separate from installed source code."""

import os
from pathlib import Path


def data_directory() -> Path:
    """Use IDQ_DATA_DIR, the source checkout's data/, or ./data for a wheel install."""
    configured = os.environ.get("IDQ_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "idQ").is_dir():
            return parent / "data"
    return Path.cwd() / "data"
