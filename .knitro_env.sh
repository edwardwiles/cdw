# NOTE: sourcing this alone does NOT change which KNITRO .so actually loads. KNITRO.jl's
# deps/deps.jl (in the shared Julia depot, `~/.julia/packages/KNITRO/<hash>/deps/deps.jl`)
# hardcodes an absolute libknitro.so path baked in at `Pkg.build` time; it is not re-resolved at
# runtime. If you change KNITRODIR here, you must also rebuild: source this file, then
#   export KNITRO_JL_USE_KNITRO_JLL=false
#   julia --project=. -e 'import Pkg; Pkg.build("KNITRO")'
# then verify with `verify_knitro_version()` (full_aod_diag/d4_exact/knitro_version_check.jl),
# called automatically by c10_d20_production_driver.jl at include time. This depot file is
# GLOBAL to the user account -- rebuilding affects every Julia process, including any other
# concurrent session on this shared host, so check for running `julia` processes first.
#
# PINNED TO 13.0.1 (2026-07-21): 14.2.0 and 14.0.0 are installed on this host but NOT covered by
# the current site Ziena license -- KN_new() returns -520 "Could not find a valid license" on
# both, confirmed live, while 13.0.1 succeeds under the identical license file
# (/etc/sharedsw_licenses/ziena.txt). Do not point KNITRODIR at 14.x until Artelys
# (licensing@artelys.com) confirms the license covers it; verify_knitro_version() will fail fast
# if this drifts out of sync with what's actually built.
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:$LD_LIBRARY_PATH
export PATH="$HOME/.juliaup/bin:$PATH"
