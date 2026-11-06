PYTHON ?= python
CONCEPTS := $(sort $(wildcard [0-9][0-9]-*))

.PHONY: test lint gifs all

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

# Each animate.py writes demo.gif next to itself.
gifs:
	@for d in $(CONCEPTS); do echo "rendering $$d/demo.gif"; $(PYTHON) $$d/animate.py || exit 1; done

all: lint test gifs
