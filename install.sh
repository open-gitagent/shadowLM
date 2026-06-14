#!/bin/sh
# ShadowLM Trainer — one-command install + studio.
#
#   curl -fsSL https://shadowlm.sh | sh
#
# Installs shadowlm into an isolated venv (~/.shadowlm/venv), picks the right
# training backend for your machine, and opens the studio. Re-run any time to
# upgrade. Override with env vars:
#   SHADOWLM_EXTRAS=cli      # lighter: just the UI/CLI, add a backend later
#   SHADOWLM_PORT=8329       # studio port
#   SHADOWLM_NO_SERVE=1      # install only, don't launch
set -eu

HOME_DIR="${SHADOWLM_HOME:-$HOME/.shadowlm}"
VENV="$HOME_DIR/venv"
PORT="${SHADOWLM_PORT:-8329}"

esc=$(printf '\033')  # real ESC so codes render inside printf %s
red="${esc}[38;5;203m"; dim="${esc}[2m"; off="${esc}[0m"
say() { printf "${red}slm♥${off} %s\n" "$1"; }
err() { printf "${red}✗${off} %s\n" "$1" >&2; exit 1; }

# 1. find a Python 3.10+ -------------------------------------------------------
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 \
     && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || err "Python 3.10+ is required. Install it (e.g. brew install python) and re-run."

# 2. pick the backend extras for this machine ---------------------------------
EXTRAS="${SHADOWLM_EXTRAS:-}"
if [ -z "$EXTRAS" ]; then
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    EXTRAS="mlx-all"      # Apple Silicon — the native mlx dev loop
  else
    EXTRAS="all"          # Linux/other — the CUDA/CPU torch stack
  fi
fi

# 3. install into an isolated venv (idempotent; upgrades on re-run) ------------
say "using $($PY --version 2>&1)"
say "installing ${dim}shadowlm[$EXTRAS]${off} into $VENV  (first run can take a few minutes)"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install --quiet --upgrade "shadowlm[$EXTRAS]"

# 4. put `shadowlm` on PATH if we can ----------------------------------------
BIN="$HOME/.local/bin"
if [ -d "$BIN" ] || mkdir -p "$BIN" 2>/dev/null; then
  ln -sf "$VENV/bin/shadowlm" "$BIN/shadowlm"
  case ":$PATH:" in
    *":$BIN:"*) ;;
    *) say "add to PATH:  ${dim}export PATH=\"$BIN:\$PATH\"${off}" ;;
  esac
fi

VER="$("$VENV/bin/python" -c 'import shadowlm; print(shadowlm.__version__)' 2>/dev/null || echo "?")"
say "installed ShadowLM Trainer $VER"

# 5. launch the studio --------------------------------------------------------
if [ "${SHADOWLM_NO_SERVE:-}" = "1" ]; then
  say "run the studio when ready:  ${dim}shadowlm serve${off}"
  exit 0
fi
URL="http://127.0.0.1:$PORT"
say "starting the studio → ${URL}  ${dim}(Ctrl-C to stop)${off}"
( sleep 2
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  fi ) >/dev/null 2>&1 &
exec "$VENV/bin/shadowlm" serve --port "$PORT"
