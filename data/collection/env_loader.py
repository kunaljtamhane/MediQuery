from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_env_file(env_path: Path) -> None:
    """Populate os.environ from a simple .env file without overriding existing values."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        cleaned = value.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1]

        os.environ[key] = cleaned


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def configure_requests_session(session: Any, trust_env_default: bool = False) -> Any:
    """
    Normalize requests.Session proxy behavior for collection scripts.

    These jobs run well in local environments that may export placeholder proxy
    variables. By default we bypass env-derived proxies unless the caller opts
    back in via COLLECTION_TRUST_ENV_PROXY=1.
    """
    if hasattr(session, "trust_env"):
        session.trust_env = env_flag("COLLECTION_TRUST_ENV_PROXY", trust_env_default)
    return session
