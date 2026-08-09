#!/bin/bash
# paper_upper_v1: records the exact reproducibility-relevant environment/source state into the
# protocol manifest BEFORE the real (non-toy) launch. Run once, from the protocol branch worktree,
# after committing the multistart_seed_generator.jl / cm_checkpoint.jl / cm_originzc_checkpoint.jl
# extensions.
#
# Usage:
#   ./freeze_protocol_source.sh <path-to-protocol-toml>
#
# Refuses to run against a dirty worktree (protocol section 21: never launch from uncommitted
# state) and refuses to overwrite a manifest whose protocol_sha is already frozen (not
# "PENDING_COMMIT") -- that would silently change what a running/completed campaign claims to be
# reproducing.

set -euo pipefail

TOML_PATH="$1"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$SRC_DIR"

if [ -n "$(git status --porcelain)" ]; then
    echo "REFUSING: worktree at $SRC_DIR is dirty. Commit the protocol-branch extensions first:" >&2
    git status --short >&2
    exit 1
fi

CURRENT_SHA=$(git rev-parse HEAD)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

EXISTING_SHA=$(grep '^protocol_sha' "$TOML_PATH" | head -1 | sed -E 's/.*"([^"]*)".*/\1/')
if [ "$EXISTING_SHA" != "PENDING_COMMIT" ] && [ "$EXISTING_SHA" != "$CURRENT_SHA" ]; then
    echo "REFUSING: $TOML_PATH already has a DIFFERENT frozen protocol_sha=$EXISTING_SHA" >&2
    echo "(current worktree HEAD is $CURRENT_SHA). This would silently redefine an already-frozen" >&2
    echo "protocol. If this is intentional, bump to a new protocol version (paper_upper_v2.toml)." >&2
    exit 1
fi

export PATH="$HOME/.juliaup/bin:$PATH"
JULIA_VERSION=$(julia --version | awk '{print $3}')

export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH="/opt/shared_sw/knitro/14.2.0/lib:${LD_LIBRARY_PATH:-}"
KNITRO_VERSION=$(julia --project="$SRC_DIR" -e 'using KNITRO; println(KNITRO.KN_get_release())' 2>/dev/null | tail -1)

echo "Recording:"
echo "  protocol_sha    = $CURRENT_SHA"
echo "  protocol_branch = $CURRENT_BRANCH"
echo "  julia_version   = $JULIA_VERSION"
echo "  knitro_version  = $KNITRO_VERSION"

python3 - "$TOML_PATH" "$CURRENT_SHA" "$CURRENT_BRANCH" "$JULIA_VERSION" "$KNITRO_VERSION" <<'PYEOF'
import sys, re
path, sha, branch, julia_v, knitro_v = sys.argv[1:6]
with open(path) as f:
    text = f.read()
text = re.sub(r'protocol_sha = "[^"]*"', f'protocol_sha = "{sha}"', text, count=1)
text = re.sub(r'protocol_branch = "[^"]*"', f'protocol_branch = "{branch}"', text, count=1)
text = re.sub(r'julia_version = "[^"]*"', f'julia_version = "{julia_v}"', text, count=1)
if knitro_v and knitro_v != "":
    if 'knitro_version' in text:
        text = re.sub(r'knitro_version = "[^"]*"', f'knitro_version = "{knitro_v}"', text, count=1)
    else:
        text = text.replace('knitro_dir = ', f'knitro_version = "{knitro_v}"\nknitro_dir = ', 1)
with open(path, "w") as f:
    f.write(text)
print("Wrote", path)
PYEOF

echo "Done. protocol_sha is now frozen at $CURRENT_SHA -- do not amend/rebase this commit."
