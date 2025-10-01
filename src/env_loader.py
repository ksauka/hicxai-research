"""
Environment loader for HicXAI agent
Loads configuration from .env file securely
"""

import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        return True
    return False

# Load .env on import
load_env()