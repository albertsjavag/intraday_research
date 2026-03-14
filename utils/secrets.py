from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> None:
    """Load a .env file into os.environ.

    Falls back to python-dotenv if available, otherwise parses manually.
    """
    env_path = Path(path) if path else Path(".env")

    # Try python-dotenv first (installed with [alpaca] extra)
    try:
        from dotenv import load_dotenv as _load

        _load(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    # Minimal manual parser
    if not env_path.exists():
        return

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def require(key: str) -> str:
    """Return the value of an environment variable, raising if absent."""
    value = os.environ.get(key)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return value
