#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  echo "error: claude CLI not found in PATH" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/delegate_experiment.sh --cmd "python3 -u experiments/qwen36_full_rust.py 1.0 1 --warmup-tokens 0 --max-tokens 1"

Options:
  --cmd       Command to execute inside the workspace via Claude Code
  --label     Optional human-readable label for the experiment
  --timeout   Optional timeout in seconds for the shell command inside Claude (default: 900)
EOF
}

CMD=""
LABEL="workspace experiment"
TIMEOUT_SECS=900

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cmd)
      CMD="${2:-}"
      shift 2
      ;;
    --label)
      LABEL="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CMD" ]]; then
  echo "error: --cmd is required" >&2
  usage >&2
  exit 1
fi

SCHEMA=$(cat <<'EOF'
{
  "type": "object",
  "properties": {
    "status": { "type": "string", "enum": ["ok", "error"] },
    "label": { "type": "string" },
    "command": { "type": "string" },
    "summary": { "type": "string" },
    "generated_text": { "type": "string" },
    "stdout_tail": { "type": "string" },
    "stderr_tail": { "type": "string" },
    "suspicious": { "type": "boolean" },
    "exit_code": { "type": "integer" }
  },
  "required": [
    "status",
    "label",
    "command",
    "summary",
    "generated_text",
    "stdout_tail",
    "stderr_tail",
    "suspicious",
    "exit_code"
  ],
  "additionalProperties": false
}
EOF
)

PROMPT=$(cat <<EOF
You are running a single experiment inside this workspace.

Task:
1. Run this shell command exactly once with a timeout of ${TIMEOUT_SECS} seconds:
   timeout ${TIMEOUT_SECS} ${CMD}
2. Capture stdout and stderr.
3. If the output contains a line that starts with "  Output:", extract the text after that prefix into generated_text.
4. Return exactly one JSON object that matches the provided schema.

Rules:
- Do not modify any files.
- Do not run any extra heavy commands.
- Keep stdout_tail and stderr_tail concise: the last relevant ~40 lines max each.
- suspicious should be true if output looks semantically wrong, corrupted, crashes, or times out.
- summary should be one short sentence in English.
- label must be "${LABEL}".
- command must be "${CMD}".
EOF
)

printf '%s' "$PROMPT" | exec claude -p \
  --output-format json \
  --json-schema "$SCHEMA" \
  --permission-mode bypassPermissions \
  --add-dir "$(pwd)"
