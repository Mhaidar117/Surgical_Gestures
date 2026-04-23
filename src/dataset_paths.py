"""Resolve the filesystem root that contains EEG/, Eye/, and Gestures/."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "SURGICAL_GESTURES_DATA_ROOT"

# Parent of EEG/, Eye/, Gestures/ on the primary development machine (iCloud Drive).
_DEFAULT_EXTERNAL_ROOT = Path(
    "/Users/michaelhaidar/Library/Mobile Documents/com~apple~CloudDocs/"
    "Documents/Vanderbilt/Fall_25/Surgical Robotics/Surgical_Gestures"
)


def resolve_dataset_root(
    cli_value: str | Path | None = None,
    *,
    fallback_repo_root: Path | None = None,
) -> Path:
    """
    Return the dataset root used for EEG, Eye, and Gestures.

    Resolution order:
    1. Explicit ``cli_value`` if non-empty
    2. Environment variable ``SURGICAL_GESTURES_DATA_ROOT``
    3. ``_DEFAULT_EXTERNAL_ROOT`` if that directory exists
    4. ``fallback_repo_root`` if provided
    5. Current working directory
    """
    if cli_value is not None and str(cli_value).strip() != "":
        return Path(cli_value).expanduser().resolve()

    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()

    ext = _DEFAULT_EXTERNAL_ROOT.expanduser().resolve()
    if ext.is_dir():
        return ext

    if fallback_repo_root is not None:
        return Path(fallback_repo_root).resolve()

    return Path(".").resolve()
