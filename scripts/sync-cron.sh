#!/bin/bash
# zsearch incremental sync wrapper for cron / launchd.
#
# - Sources $HOME/.env so JINA_API_KEY (and any other backend creds) are set
#   even when launched without a login shell.
# - Logs to a per-month file under $LOG_DIR.
# - On macOS, sends a desktop notification when the underlying `zsearch sync`
#   exits non-zero.
# - Keeps log files for 30 days, deletes older ones at the end of each run.

set -uo pipefail

REPO_DIR="${ZSEARCH_REPO_DIR:-$HOME/Projects/zotero-cli-agent}"
LOG_DIR="${ZSEARCH_LOG_DIR:-$HOME/Library/Logs/zsearch}"
LOG_RETAIN_DAYS="${ZSEARCH_LOG_RETAIN_DAYS:-30}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync-$(date +%Y-%m).log"

run_sync() {
  echo "==========================================="
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting zsearch sync"

  # Zotero guard: only run if the desktop client is up. Otherwise the local
  # zotero.sqlite is whatever yesterday/last-launch left behind, missing items
  # that Zotero would have pulled from the cloud on next start. Skipping
  # cleanly (exit 0) lets launchd's later fire times retry without launchd
  # treating today as a failure.
  # macOS ships the Zotero binary as lowercase ``zotero``; we match
  # case-insensitively to be safe across past/future renames.
  if [ "${ZSEARCH_REQUIRE_ZOTERO:-1}" = "1" ] && ! pgrep -ix zotero >/dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') deferred — Zotero client not running"
    return 200  # special: treated as success at the end of the wrapper
  fi

  if [ -f "$HOME/.env" ]; then
    set +u
    # shellcheck disable=SC1091
    source "$HOME/.env"
    set -u
  fi

  cd "$REPO_DIR" || { echo "REPO_DIR $REPO_DIR not found"; return 2; }
  # shellcheck disable=SC1091
  source .venv/bin/activate

  zsearch sync
}

# --- main ----------------------------------------------------------------
run_sync >> "$LOG_FILE" 2>&1
exit_code=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') exit=$exit_code" >> "$LOG_FILE"

# Code 200 means the Zotero guard skipped today — not a failure; remap to 0.
if [ "$exit_code" -eq 200 ]; then
  exit_code=0
fi

# Failure notification (macOS only; osascript is a no-op elsewhere).
if [ "$exit_code" -ne 0 ] && command -v osascript >/dev/null 2>&1; then
  tail_msg=$(tail -n 5 "$LOG_FILE" | tr '"' "'" | tr '\n' ' ')
  osascript -e "display notification \"$tail_msg\" with title \"zsearch sync failed (exit $exit_code)\" sound name \"Basso\"" \
    >/dev/null 2>&1 || true
fi

# Log retention: delete monthly log files older than $LOG_RETAIN_DAYS.
find "$LOG_DIR" -maxdepth 1 -type f -name "sync-*.log" -mtime +"$LOG_RETAIN_DAYS" -delete 2>/dev/null

exit "$exit_code"
