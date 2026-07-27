# Standalone Melitz-only smoke test: loads ONLY the Melitz include path + KNITRO, and
# NEVER `cc_algo/include_cc_algo.jl`. Designed to be run as its OWN process (never
# `include`d into `test/melitz/runtests.jl`, which always loads `cc_algo` for its own
# `KNITRO_AVAILABLE` guard and legacy dense-bundle cross-checks) -- see the
# "standalone (no cc_algo) subprocess" testset in `runtests.jl`, which launches this file
# with `julia --project=. test/melitz/standalone_no_cc_algo.jl` and asserts exit code 0.
#
# This test exists to catch a REAL incident found in the 2026-07-26 closure session
# (`docs/melitz_production_fast_backend_closure_2026-07-26.md` Phase 4 side finding):
# `MelitzCCBundle`'s own KNITRO driver (`cc_bundle.jl`) called the 2-arg convenience form
# `KNITRO.KN_add_vars(kc, n)`, which does not exist in the installed `KNITRO.jl` package --
# only the 3-arg core method does. The 2-arg form was silently supplied by
# `cc_algo/knitro_compat.jl`'s own monkey-patch of the KNITRO module, which every existing
# test run happened to load first (via `test/melitz/runtests.jl`'s own `KNITRO_AVAILABLE`
# guard), concealing the dependency. `MelitzCCBundle`'s own claimed "own functor...
# independent of cc_algo" was accurate for the economics/algorithm but NOT for this one
# KNITRO-version-compatibility shim. Fixed by `src/melitz/knitro_compat.jl` (Melitz-owned
# `melitz_kn_add_vars!`/`melitz_kn_add_cons!`/`melitz_kn_get_int_param`, no KNITRO-module
# type piracy). This script must FAIL (throw `MethodError`/`UndefVarError`) if that fix is
# ever reverted or a new `cc_algo`-monkey-patch-dependent call is reintroduced.

using Random
using LinearAlgebra: dot, norm
using KNITRO

const MELITZ_DIR = joinpath(@__DIR__, "..", "..", "src", "melitz")
include(joinpath(dirname(dirname(@__DIR__)), "misc", "doubleDiff.jl"))
include(joinpath(MELITZ_DIR, "include_melitz.jl"))
# NOTE: deliberately NOT including cc_algo/include_cc_algo.jl -- that is the entire point
# of this script. Any function call below that secretly needs
# `CounterfactualSensitivity`/`PsiObjectiveBundleDelta`/a `cc_algo`-monkey-patched KNITRO
# method will throw here, not silently succeed.

println(">>> standalone_no_cc_algo.jl: building a small D=4 fixture (Melitz-only)...")
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)

inner_opt = joinpath(dirname(dirname(@__DIR__)), "melitz_inner_loop_options.opt")
outer_opt = joinpath(dirname(dirname(@__DIR__)), "melitz_outer_finite_delta.opt")

println(">>> standalone_no_cc_algo.jl: constructing a production-fast (matrix-free) MelitzCCBundle...")
obj, theta_free = build_melitz_psi_bundle(data; backend=:matrix_free, forbid_dense_fallback=true,
    inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)

@assert obj isa MelitzCCBundle "standalone_no_cc_algo.jl: expected a MelitzCCBundle (matrix-free, cc_algo-independent), got $(typeof(obj))"

println(">>> standalone_no_cc_algo.jl: performing a real matrix-free inner KNITRO solve (melitz_recover_lfd)...")
lfd = melitz_recover_lfd(obj, theta_free)

println(">>> nStatus=", lfd.nStatus, "  Delta=", lfd.Delta, "  lfd_ok=", lfd.lfd_ok)

@assert lfd.nStatus == 0 "standalone_no_cc_algo.jl: expected nStatus == 0 (converged), got $(lfd.nStatus)"
@assert lfd.lfd_ok "standalone_no_cc_algo.jl: expected lfd_ok == true"
@assert isfinite(lfd.Delta) && lfd.Delta > 0 "standalone_no_cc_algo.jl: expected a finite, positive DeltaStar, got $(lfd.Delta)"

println(">>> standalone_no_cc_algo.jl: PASSED (no cc_algo ever loaded; MelitzCCBundle's own KNITRO driver is self-sufficient).")
