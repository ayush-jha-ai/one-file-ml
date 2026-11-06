"""Make `from model import ...` resolve to the model.py next to each test file.

Every concept folder has a file called model.py. Before pytest imports a test module
we put that module's folder first on sys.path and forget any previously imported
`model`, so each test file sees its own implementation.
"""

import re
import sys
from pathlib import Path

import pytest


def _use_folder(folder: Path) -> None:
    sys.modules.pop("model", None)
    sys.path.insert(0, str(folder))


def pytest_pycollect_makemodule(module_path, parent):
    _use_folder(Path(module_path).parent)


def _run_readme_snippets(test_file: str) -> int:
    """Execute every ```python block in the README beside `test_file`.

    The "Where it breaks" examples contain asserts, so this test fails if a claim
    in a README stops being true. Returns the number of blocks run.
    """
    folder = Path(test_file).parent
    text = (folder / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    for block in blocks:
        _use_folder(folder)
        exec(compile(block, str(folder / "README.md"), "exec"), {"__name__": "__readme__"})
    return len(blocks)


@pytest.fixture
def run_readme():
    """Fixture form of _run_readme_snippets, so test files need no extra imports."""
    return _run_readme_snippets
