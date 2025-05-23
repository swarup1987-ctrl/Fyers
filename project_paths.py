from pathlib import Path

# The root is the current working directory from which the script is run.
PROJECT_ROOT = Path().resolve()

def data_path(filename):
    """Get the full path to a data file in the project root."""
    return PROJECT_ROOT / filename
