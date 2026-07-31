#!/usr/bin/env bash
# STATUS_COMMAND.sh -- report per-family cell counts and process liveness for the
# sigma3/W500k five-family production campaign (2026-07-30).
set -euo pipefail
cd "$(dirname "$0")"
./campaign_control.sh --status
