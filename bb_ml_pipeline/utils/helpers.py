"""Helper utility functions for BB ML Pipeline."""

import os
import json
from datetime import datetime
from typing import Dict, Any


def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory to ensure exists.
        
    Returns:
        The directory path.
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """
    Get a formatted timestamp string for directory naming.
    
    Returns:
        Formatted timestamp string.
    """
    return datetime.now().strftime("%m%d%Y%H%M%S")


def config_to_str(config: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert a configuration dictionary to a formatted string.
    
    Args:
        config: Configuration dictionary.
        indent: Number of spaces for indentation.
        
    Returns:
        Formatted string representation of the configuration.
    """
    return json.dumps(config, indent=indent) 