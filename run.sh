#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

usage() {
  cat <<'EOF'
Usage: ./run.sh <command>

  setup      create .venv, install requirements, check GROQ_API_KEY
  ingest     run the ingestion pipeline (stages 1-5)
  report     print the chunking report
  stats      print vector store statistics
  chat       start the terminal chat (stages 6-13)
  ui         start the Streamlit UI
  test       run the offline test suite
  selftest   run the pipeline self-test
EOF
}

activate_venv() {
  if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}

require_module() {
  local module="$1"
  if ! "$PYTHON" -c "import ${module}" >/dev/null 2>&1; then
    echo "run.sh: module '${module}' is not available yet (later phase or missing dependency)." >&2
    echo "run.sh: run './run.sh setup' first, or continue with the next implementation phase." >&2
    exit 2
  fi
}

case "${1:-}" in
  setup)
    "$PYTHON" -m venv "$VENV_DIR"
    activate_venv
    "$PYTHON" -m pip install --upgrade "pip>=26.2.2"
    "$PYTHON" -m pip install -r requirements.txt
    if [ -f .env ]; then
      chmod 600 .env 2>/dev/null || true
    fi
    if [ -f .env ] && grep -q '^GROQ_API_KEY=.\+' .env; then
      echo "run.sh: GROQ_API_KEY is set. Chat will call Groq."
    else
      echo "run.sh: set GROQ_API_KEY in .env before './run.sh chat'."
    fi
    ;;
  ingest)
    activate_venv
    require_module src.ingest
    "$PYTHON" -m src.ingest "${@:2}"
    ;;
  report)
    activate_venv
    require_module src.chunking
    "$PYTHON" -m src.chunking --report
    ;;
  stats)
    activate_venv
    require_module src.vectorstore
    "$PYTHON" -m src.vectorstore --stats
    ;;
  chat)
    activate_venv
    require_module src.answer
    "$PYTHON" -m src.answer --chat
    ;;
  ui)
    activate_venv
    require_module streamlit
    if [ ! -f app.py ]; then
      echo "run.sh: app.py does not exist yet (Phase 9)." >&2
      exit 2
    fi
    streamlit run app.py
    ;;
  test)
    activate_venv
    "$PYTHON" -m pytest "${@:2}"
    ;;
  selftest)
    activate_venv
    require_module src.answer
    "$PYTHON" -m src.answer --selftest
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "run.sh: unknown command '${1}'" >&2
    usage
    exit 1
    ;;
esac
