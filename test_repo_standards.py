"""Repository-wide checks that every concept meets the standards in CONTRIBUTING.md."""

import ast
import re
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).parent
CONCEPTS = sorted(p for p in ROOT.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name))
SECTIONS = ["## Intuition", "## The maths", "## Code walkthrough", "## Where it breaks",
            "## Common mistakes", "## Further reading"]
BANNED = ["revolutionary", "deep dive", "unlock", "game-changer", "game changer",
          "cutting-edge", "seamless", "supercharge", "leverage"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")


def test_twelve_concepts_present():
    assert len(CONCEPTS) == 12


@pytest.mark.parametrize("folder", CONCEPTS, ids=lambda p: p.name)
def test_model_is_short_and_numpy_only(folder):
    source = (folder / "model.py").read_text()
    assert len(source.splitlines()) <= 150
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy"}, f"model.py imports {imported - {'numpy'}}"


@pytest.mark.parametrize("folder", CONCEPTS, ids=lambda p: p.name)
def test_gif_size_and_length(folder):
    gif = folder / "demo.gif"
    assert gif.stat().st_size < 3 * 1024 * 1024
    with Image.open(gif) as im:
        total_ms = 0
        for i in range(im.n_frames):
            im.seek(i)
            total_ms += im.info.get("duration", 0)
    assert 0 < total_ms <= 8000


@pytest.mark.parametrize("folder", CONCEPTS, ids=lambda p: p.name)
def test_readme_structure(folder):
    text = (folder / "README.md").read_text()
    positions = [text.find(s) for s in SECTIONS]
    assert all(p >= 0 for p in positions), "missing section"
    assert positions == sorted(positions), "sections out of order"
    assert 0 <= text.find("(demo.gif)") < positions[0], "GIF must come before Intuition"
    intuition = text[positions[0] + len(SECTIONS[0]): positions[1]]
    assert len(intuition.split()) <= 120


def walkthrough_rows(text):
    start = text.index("## Code walkthrough")
    end = text.index("## Where it breaks")
    return [line for line in text[start:end].splitlines() if "model.py#L" in line]


def squash(s):
    return re.sub(r"\s+", "", s)


@pytest.mark.parametrize("folder", CONCEPTS, ids=lambda p: p.name)
def test_walkthrough_links_point_at_the_quoted_code(folder):
    """Each table row quotes code in backticks and links to line numbers. At least one
    quoted fragment (split on "...") must appear in the linked lines."""
    lines = (folder / "model.py").read_text().splitlines()
    for row in walkthrough_rows((folder / "README.md").read_text()):
        linked = []
        for a, b in re.findall(r"model\.py#L(\d+)(?:-L(\d+))?", row):
            linked += lines[int(a) - 1: int(b or a)]
        target = squash("".join(linked))
        fragments = [squash(f) for span in re.findall(r"`([^`]+)`", row)
                     for f in re.split(r"\.\.\.|\(\.\.\.\)", span) if len(squash(f)) >= 5]
        assert fragments, f"no code quoted in row: {row}"
        assert any(f in target for f in fragments), f"{folder.name}: {row[:90]}"


@pytest.mark.parametrize("path", sorted(ROOT.glob("**/*.md")), ids=lambda p: str(p.name))
def test_writing_style(path):
    if ".venv" in path.parts or ".pytest_cache" in path.parts:
        return
    text = path.read_text().lower()
    if path.name != "CONTRIBUTING.md":               # it lists the banned words
        for word in BANNED:
            assert word not in text, f"{path}: '{word}'"
    assert not EMOJI.search(path.read_text()), f"{path}: emoji"
