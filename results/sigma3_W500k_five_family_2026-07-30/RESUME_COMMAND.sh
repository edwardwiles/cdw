#!/usr/bin/env bash
# RESUME_COMMAND.sh -- re-invoke all 5 family chains for the sigma3/W500k five-family production
# campaign (2026-07-30). Safe to run repeatedly: every cell checks its own DONE marker before
# doing any work, so already-completed cells are skipped, not re-solved or overwritten.
set -euo pipefail
cd "$(dirname "$0")"
./campaign_control.sh --resume
