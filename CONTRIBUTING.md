# Contributing

Thank you for helping. This repo values correctness and clarity over coverage, so a new concept is accepted only when it meets the same standard as the existing twelve. This page is the complete checklist. `make all` checks most of it automatically.

## Kinds of contribution

- **A new concept.** Open an issue first, with a one-paragraph case for why it belongs among the core ideas, so effort is not wasted.
- **An extra "where it breaks" example** for an existing concept. It must be runnable, end in assertions, and show something the existing examples do not.
- **A translation** of a concept README. Put it in `translations/<language-code>/<concept-folder>.md`. Keep the code blocks unchanged so they still run.
- **A correction.** If a derivation, number or citation is wrong, open an issue or a pull request. These take priority over everything else.

## Setting up

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all          # ruff, every test (including README examples), all GIFs
```

## The per-concept template

A concept lives in a folder named `NN-short-name/` (two digits, lower case, hyphens) containing exactly these files.

### `model.py`

- NumPy only. No other imports, including the standard library's `math` (use `np`). The test suite enforces this.
- At most 150 lines, readable from top to bottom. Helper functions come before the code that uses them.
- Classes at most one level deep (no inheritance hierarchies).
- A module docstring that states the model, the loss and the update rule in plain text.
- Comments explain *why* or name the equation a line implements. Do not narrate the obvious.
- Deterministic: anything random takes a `seed` or a `numpy.random.Generator`.

### `test_model.py`

- Compare against scikit-learn or PyTorch (float64) on fixed seeds.
- The module docstring lists every tolerance and what it applies to.
- **Never loosen a tolerance to make a test pass.** If the implementation does not match, find out why:
  - a bug: fix it;
  - a documented difference in the reference (scikit-learn stores tree inputs as float32 and breaks exact ties in a random feature order, for example): handle it openly and test the difference itself, as concepts 08 and 10 do;
  - randomness that cannot be matched draw for draw: test statistically and justify the tolerance with a standard error, as concept 09 does.
- Include `test_readme_examples_run(run_readme)`, which executes every `python` code block in the README.

### `animate.py`

- Writes `demo.gif` next to itself using the shared `plotstyle.py` at the repository root (`ps.apply()` and `ps.save_gif(...)`).
- At most 8 seconds (`save_gif` refuses to write longer) and under 3 MB.
- Light background, large type, readable at phone width. Use the categorical colours in order: `ps.BLUE`, `ps.ORANGE`, `ps.AQUA`. More than three series in one scatter means the picture needs rethinking.
- It must show the algorithm learning or operating, not a static result.
- Look at a few frames yourself before submitting. Check for overlapping labels and anything that runs off the axes.

### `README.md`

These sections, in this order (the test suite checks the order):

1. `# NN · Title`, then **one sentence**: what this solves and why it matters.
2. The GIF, with alt text that describes what happens in it.
3. `## Intuition`: at most 120 words. Define every technical term you use.
4. `## The maths`: standard notation, written so a strong A-level student can follow it. Show every step a beginner would get stuck on: expand the square, state the chain rule, say why a cross term vanishes.
5. `## Code walkthrough`: a table with one row per equation. Each row has the equation, the exact code in backticks, and a link such as `[21](model.py#L21)`. The test suite checks that each link points at a line containing the quoted code. After the table, say what the tests compare against and to what tolerance.
6. `## Where it breaks`: 2 to 4 failure modes. Each one has a bold one-line claim, a short explanation, and a runnable code block that ends in `assert` statements confirming the claim. Every number in the prose must come from running the code.
7. `## Common mistakes`: 3 to 5 misconceptions learners actually have, each in the form **"the wrong belief"** followed by the correct mental model.
8. `## Further reading`: only real, well-known sources such as original papers and standard textbooks, with authors, title, venue and year. If you are not certain a source exists exactly as cited, leave it out. Never invent a citation.

## Writing style

- Plain UK English: *optimise*, *regularisation*, *behaviour*.
- Direct and specific. Prefer a number to an adjective.
- No hype ("revolutionary", "deep dive", "unlock" and similar are rejected by the tests), no emojis, no filler sentences.
- Every formula in the README must match the code. Read them side by side before submitting.

## Before opening a pull request

```bash
make all
```

This runs ruff, every test (including every README example and the repository standards checks), and regenerates every GIF. If a GIF changes that you did not mean to change, find out why. Generation is deterministic.
