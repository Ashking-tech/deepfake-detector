"""Shared test helpers: path to the image fixtures folder."""

import pathlib

# Folder holding all test images (tests/fixtures/).
FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# Builds the full path to one fixture image by name.
def fixture_path(name: str) -> pathlib.Path:
    """Return tests/fixtures/<name> as a Path object."""
    full = FIXTURES / name
    return full
