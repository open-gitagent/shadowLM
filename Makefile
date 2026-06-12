# ShadowLM Trainer — common workflows. Run `make` to list them.
#
# Layout: a Python package (shadowlm/) with an opt-in CLI/server, and a React
# studio in frontend/ that builds into shadowlm/_static (shipped in the wheel).

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
SHADOWLM:= $(VENV)/bin/shadowlm
PORT    ?= 8329

.DEFAULT_GOAL := help

# ---- help -------------------------------------------------------------------
.PHONY: help
help:  ## list the available targets
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-14s\033[0m %s\n",$$1,$$2}'

# ---- setup ------------------------------------------------------------------
$(PY):
	python3 -m venv $(VENV)

.PHONY: install
install: $(PY)  ## editable install with the CLI + a training backend (mlx)
	$(PIP) install -q -e '.[mlx,cli]'

.PHONY: install-torch
install-torch: $(PY)  ## editable install for CUDA / CPU boxes
	$(PIP) install -q -e '.[torch,cli]'

.PHONY: frontend
frontend:  ## install + build the React studio into shadowlm/_static
	cd frontend && npm install && npm run build

# ---- run --------------------------------------------------------------------
.PHONY: serve
serve:  ## run the studio + API on one port (make serve PORT=8329)
	$(SHADOWLM) serve --port $(PORT)

.PHONY: dev
dev:  ## serve with Vite hot-reload UI alongside the backend
	$(SHADOWLM) serve --port $(PORT) --dev

.PHONY: demo
demo:  ## end-to-end smoke: a tiny finetune through the CLI
	$(SHADOWLM) finetune examples/sample_dataset.jsonl \
	  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --method lora --max-steps 8

# ---- checks -----------------------------------------------------------------
.PHONY: check
check:  ## compile the package + typecheck the frontend
	$(PY) -m compileall -q shadowlm
	cd frontend && npx tsc -b

.PHONY: gpu-test
gpu-test:  ## the CUDA verification suite (run on a GPU box)
	$(PY) tests/gpu/test_cuda.py

# ---- build / release --------------------------------------------------------
.PHONY: build
build: frontend  ## build the wheel + sdist (with the studio inside) and validate
	rm -rf dist
	$(PIP) install -q build twine
	$(PY) -m build
	$(VENV)/bin/twine check dist/*

.PHONY: version
version:  ## print the version recorded in both source files
	@grep '^version' pyproject.toml
	@grep '^__version__' shadowlm/__init__.py

.PHONY: bump
bump:  ## set the version in both source files (make bump V=0.2.1)
ifndef V
	$(error set V, e.g. `make bump V=0.2.1`)
endif
	$(PY) -c "import re,pathlib; \
p=pathlib.Path('pyproject.toml'); \
p.write_text(re.sub(r'(?m)^version = \".*\"', 'version = \"$(V)\"', p.read_text())); \
i=pathlib.Path('shadowlm/__init__.py'); \
i.write_text(re.sub(r'__version__ = \".*\"', '__version__ = \"$(V)\"', i.read_text()))"
	@$(MAKE) --no-print-directory version

.PHONY: release
release:  ## bump, build, commit, tag, push — CI publishes to PyPI (make release V=0.2.1)
ifndef V
	$(error set V, e.g. `make release V=0.2.1`)
endif
	@git diff --quiet || { echo "working tree is dirty — commit or stash first"; exit 1; }
	@$(MAKE) --no-print-directory bump V=$(V)
	@$(MAKE) --no-print-directory build
	git add pyproject.toml shadowlm/__init__.py shadowlm/_static
	git commit -m "Release $(V)"
	git tag -a v$(V) -m "ShadowLM Trainer $(V)"
	git push && git push --tags
	@echo "pushed v$(V) — watch the publish: gh run watch"

# ---- clean ------------------------------------------------------------------
.PHONY: clean
clean:  ## remove build artifacts and caches (keeps the built _static)
	rm -rf dist build *.egg-info
	find shadowlm -name __pycache__ -type d -prune -exec rm -rf {} +
