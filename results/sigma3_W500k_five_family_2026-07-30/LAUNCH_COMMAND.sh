#!/usr/bin/env bash
# LAUNCH_COMMAND.sh -- sigma3/W500k five-family production campaign (2026-07-30)
#
# The audited, complete launch command. Defaults to outer strategy direct_sr1 (the optional BFGS
# polish stage is available but NOT enabled here -- see run_preflights.jl's strategy handoff smoke
# for how to opt in). Requires READY_TO_LAUNCH (written by --preflight-only after every gate
# passes) and refuses if campaign_config.json has changed since the preflight-verified checksum.
#
# This script does not launch anything by running it blind -- read PREFLIGHT_SUMMARY.json and the
# final verdict block in this directory's handoff report first.
set -euo pipefail
cd "$(dirname "$0")"
./campaign_control.sh --launch
