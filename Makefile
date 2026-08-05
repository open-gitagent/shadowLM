# ShadowLM Trainer — common workflows. Run `make` to list them.
#
# Layout: a Python package (shadowlm/) with an opt-in CLI/server, and a React
# studio in frontend/ that builds into shadowlm/_static (shipped in the wheel).

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
SHADOWLM:= $(VENV)/bin/shadowlm
PORT    ?= 8329

# Version: read the current one, compute the next. `make release` bumps the
# patch; override with BUMP=minor / BUMP=major, or pin with V=1.2.3.
CURRENT := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
BUMP    ?= patch
V       ?= $(shell echo $(CURRENT) | awk -F. -v b=$(BUMP) '{ \
	if (b=="major") printf "%d.0.0", $$1+1; \
	else if (b=="minor") printf "%d.%d.0", $$1, $$2+1; \
	else printf "%d.%d.%d", $$1, $$2, $$3+1 }')

.DEFAULT_GOAL := help

# ---- help -------------------------------------------------------------------
.PHONY: help
help:  ## list the available targets
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-14s\033[0m %s\n",$$1,$$2}'

# ---- setup ------------------------------------------------------------------
$(PY):
	python3 -m venv $(VENV)

# Targets that run the CLI need the package *installed*, not just a venv. Guard
# them so a fresh clone gets told what to do instead of the bare
# "make: .venv/bin/shadowlm: No such file or directory".
$(SHADOWLM):
	@echo "shadowlm isn't installed in $(VENV) yet. Install it:"
	@echo "    make install         # Apple Silicon (adds the mlx backend)"
	@echo "    make install-torch   # CUDA or CPU"
	@echo ""
	@echo "Or run it from an environment you already have:"
	@echo "    python3 -m shadowlm.serve --port $(PORT)"
	@exit 1

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
serve: | $(SHADOWLM)  ## run the studio + API on one port (make serve PORT=8329)
	$(SHADOWLM) serve --port $(PORT)

.PHONY: dev
dev: | $(SHADOWLM)  ## serve with Vite hot-reload UI alongside the backend
	$(SHADOWLM) serve --port $(PORT) --dev

.PHONY: demo
demo: | $(SHADOWLM)  ## end-to-end smoke: a tiny finetune through the CLI
	$(SHADOWLM) finetune examples/sample_dataset.jsonl \
	  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --method lora --max-steps 8

# ---- checks -----------------------------------------------------------------
.PHONY: check
check: | $(PY)  ## compile the package + typecheck the frontend
	$(PY) -m compileall -q shadowlm
	cd frontend && npx tsc -b

.PHONY: gpu-test
gpu-test: | $(PY)  ## the CUDA verification suite (run on a GPU box)
	$(PY) tests/gpu/test_cuda.py

# ---- gpu (cloud demo box) ---------------------------------------------------
# Instance id + domain live in .env (gitignored): GPU_INSTANCE, GPU_DOMAIN.
# studio.shadowlm.sh runs over a Cloudflare Tunnel, so stop/start needs no DNS
# change — the box reconnects on boot and the URL just comes back.
_GPUENV = set -a; [ -f .env ] && . ./.env; set +a; \
	[ -n "$$GPU_INSTANCE" ] || { echo "set GPU_INSTANCE=i-... in .env"; exit 1; }

.PHONY: gpu-start
gpu-start:  ## start the cloud GPU box (auto-retries on capacity; URL back in ~1-2 min)
	@$(_GPUENV); \
	started=0; \
	for i in $$(seq 1 8); do \
	  if out=$$(aws ec2 start-instances --instance-ids $$GPU_INSTANCE \
	      --query "StartingInstances[0].CurrentState.Name" --output text 2>&1); then \
	    echo "starting ($$out)"; started=1; break; \
	  fi; \
	  case "$$out" in \
	    *InsufficientInstanceCapacity*) echo "AZ out of capacity — retry $$i/8 in 15s…"; sleep 15;; \
	    *) echo "$$out"; exit 1;; \
	  esac; \
	done; \
	[ "$$started" = 1 ] || { echo "no capacity after 8 tries — retry later, or ask me to migrate AZ (snapshot+relaunch)"; exit 1; }; \
	echo "waiting for running..."; aws ec2 wait instance-running --instance-ids $$GPU_INSTANCE; \
	IP=$$(aws ec2 describe-instances --instance-ids $$GPU_INSTANCE \
	  --query "Reservations[0].Instances[0].PublicIpAddress" --output text); \
	echo "state: running · IP $$IP$${GPU_DOMAIN:+ · https://$$GPU_DOMAIN} (URL back in ~1-2 min)"

.PHONY: gpu-stop
gpu-stop:  ## stop the cloud GPU box (halts GPU billing; EBS volume remains)
	@$(_GPUENV); \
	aws ec2 stop-instances --instance-ids $$GPU_INSTANCE \
	  --query "StoppingInstances[0].CurrentState.Name" --output text; \
	echo "waiting for stopped..."; aws ec2 wait instance-stopped --instance-ids $$GPU_INSTANCE; \
	echo "state: $$(aws ec2 describe-instances --instance-ids $$GPU_INSTANCE \
	  --query "Reservations[0].Instances[0].State.Name" --output text) · GPU billing off"

.PHONY: gpu-reload
gpu-reload:  ## restart the studio: frees all GPU VRAM + reloads it (box must be running)
	@$(_GPUENV); \
	IP=$$(aws ec2 describe-instances --instance-ids $$GPU_INSTANCE \
	  --query "Reservations[0].Instances[0].PublicIpAddress" --output text); \
	[ "$$IP" != "None" ] || { echo "box isn't running — run 'make gpu-start' first"; exit 1; }; \
	aws ec2-instance-connect send-ssh-public-key --instance-id $$GPU_INSTANCE \
	  --instance-os-user ubuntu --ssh-public-key "file://$$HOME/.ssh/id_rsa.pub" >/dev/null; \
	ssh -o StrictHostKeyChecking=no -i $$HOME/.ssh/id_rsa ubuntu@$$IP \
	  'sudo systemctl restart shadowlm; sleep 3; echo "shadowlm: $$(systemctl is-active shadowlm) · VRAM used: $$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"'

.PHONY: gpu-status
gpu-status:  ## show the cloud GPU box state + public IP
	@$(_GPUENV); \
	aws ec2 describe-instances --instance-ids $$GPU_INSTANCE \
	  --query "Reservations[0].Instances[0].[State.Name,InstanceType,PublicIpAddress]" --output text

.PHONY: gpu-ssh
gpu-ssh:  ## ssh in via EC2 Instance Connect (no static key/IP needed)
	@$(_GPUENV); \
	IP=$$(aws ec2 describe-instances --instance-ids $$GPU_INSTANCE \
	  --query "Reservations[0].Instances[0].PublicIpAddress" --output text); \
	aws ec2-instance-connect send-ssh-public-key --instance-id $$GPU_INSTANCE \
	  --instance-os-user ubuntu --ssh-public-key "file://$$HOME/.ssh/id_rsa.pub" >/dev/null; \
	ssh -o StrictHostKeyChecking=no -i $$HOME/.ssh/id_rsa ubuntu@$$IP

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
bump:  ## write version V into both source files (default: next patch)
	$(PY) -c "import re,pathlib; \
p=pathlib.Path('pyproject.toml'); \
p.write_text(re.sub(r'(?m)^version = \".*\"', 'version = \"$(V)\"', p.read_text())); \
i=pathlib.Path('shadowlm/__init__.py'); \
i.write_text(re.sub(r'__version__ = \".*\"', '__version__ = \"$(V)\"', i.read_text()))"
	@$(MAKE) --no-print-directory version

.PHONY: release
release:  ## cut a release: auto-bump patch (or BUMP=minor/major, or V=x.y.z)
	@git diff --quiet || { echo "working tree is dirty — commit or stash first"; exit 1; }
	@echo "releasing $(CURRENT) → $(V)"
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