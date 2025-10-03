"""
Environment loader for HicXAI agent
Loads configuration from .env file securely
"""

import os
from pathlib import Path

def _load_env_file(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                k = key.strip()
                v = value.strip()
                # Do NOT override variables already set in the process env
                # This preserves values set by entrypoints (e.g., app_v1.py sets HICXAI_VERSION=v1)
                if k not in os.environ:
                    os.environ[k] = v
    return True


def load_env() -> bool:
    """Load environment variables from .env.local (preferred) and .env files."""
    root = Path(__file__).parent.parent
    loaded_any = False
    # Prefer .env.local for developer-specific overrides
    loaded_any = _load_env_file(root / '.env.local') or loaded_any
    # Then load .env as the shared defaults
    loaded_any = _load_env_file(root / '.env') or loaded_any
    return loaded_any

# Load .env on import
load_env()