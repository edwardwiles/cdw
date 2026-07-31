#!/usr/bin/env bash
# STOP_COMMAND.sh -- SIGTERM the recorded process groups for the sigma3/W500k five-family
# production campaign (2026-07-30). Cells that were mid-solve are left at their last checkpoint;
# RESUME_COMMAND.sh picks up cleanly from there (or re-attempts, up to the automatic retry cap,
# if no checkpoint had been written yet for that cell).
set -euo pipefail
cd "$(dirname "$0")"
./campaign_control.sh --stop
