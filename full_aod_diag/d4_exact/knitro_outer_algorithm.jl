# ============================================================================
# Single source of truth for the OUTER KNITRO problem's `algorithm` (nlp_algorithm)
# and `hessopt` settings, when a caller explicitly opts in via `pin_outer_algorithm=true`
# on `run_profile_checkpointed`/`run_polish_checkpointed`/`run_cm_upper_checkpointed`.
#
# PRODUCTION DEFAULT BEHAVIOR IS UNCHANGED BY THIS FILE (pin_outer_algorithm defaults to
# false everywhere): each driver keeps its own pre-existing outer-algorithm choice (currently
# a per-family, mutually inconsistent mix -- run_profile_checkpointed hardcodes algorithm=3,
# the other two drivers leave the .opt file's algorithm=auto in effect). Changing that mix is
# explicitly OUT OF SCOPE for the 2026-07-25 allocation/Hessian production port (see that
# task's §1.3 "Do not port: a forced outer-algorithm change") -- this file exists only so
# matched benchmark comparisons (§4 of that task) can request one pinned, reproducible outer
# algorithm across both families for apples-to-apples A/B timing, without silently flipping
# what real production campaigns run by default.
#
# Background: every existing `.opt` file (csw_outer_wallclock_{sr1,lbfgs,productfd}.opt)
# hardcodes `algorithm auto`. KNITRO's live resolution of `auto` is NOT one fixed choice --
# the 2026-07-25 wall-clock audit found it resolves to Active-Set/CG for the unrestricted
# family's unconstrained profile formulation but to Interior-Point/Barrier-Direct for CM's
# constrained formulation, i.e. the two families silently run different outer algorithms
# despite identical `.opt` files. `run_profile_checkpointed` alone papers over this today by
# hardcoding `algorithm=3` (Active-Set/SLQP) -- but only in that one routine.
#
# The `experiment/fullA-outer-strategy-delta2-2026-07-23` branch (see memory
# outer-strategy-delta2-experiment-2026-07-23) is the only controlled, reproduced-at-every-scale
# comparison this project has run across outer-algorithm choices: `algorithm=2` (Interior/CG) +
# `hessopt=6` (L-BFGS) beat the `auto`(->Direct)+SR1 default by +9.8% relative kappa at a
# matched 12-minute budget, and reached the 60-minute baseline's answer in ~18 minutes. That
# result is real and reproducible, but a single controlled experiment at one delta is not, by
# itself, authorization to change what every production campaign runs by default -- hence
# opt-in here, not a forced default.
#
# NOTE: this governs the OUTER NLP solve over (A, g_p) only. It is unrelated to the INNER
# fixed-point solve's exact-Hessian-vs-HVP backend choice (CM Hessian architecture / BLAS
# threading), which is a completely separate computation selected via `cm_hessian_backend` /
# `blas_threads` / the production backend manifest.
# ============================================================================

"Interior/CG. See module docstring: +9.8% relative kappa vs auto+SR1 at matched budget, delta=2 experiment. Only applied when a caller passes pin_outer_algorithm=true."
const PRODUCTION_OUTER_ALGORITHM = 2

"L-BFGS. Paired with PRODUCTION_OUTER_ALGORITHM=2 in the winning delta=2 experiment config."
const PRODUCTION_OUTER_HESSOPT = 6

"KNITRO's own sentinel for `algorithm`/`hessopt` left unresolved to a specific choice."
const KNITRO_AUTO = 0

"""
    set_production_outer_algorithm!(kc)

Explicitly sets the outer KNITRO context's `algorithm` and `hessopt` parameters to the pinned
comparison config (see module docstring). Callers opt in via `pin_outer_algorithm=true` on the
relevant driver; this is NOT called unconditionally by any production driver's default path.
"""
function set_production_outer_algorithm!(kc)
    KNITRO.KN_set_param_by_name(kc, "algorithm", PRODUCTION_OUTER_ALGORITHM)
    KNITRO.KN_set_param_by_name(kc, "hessopt", PRODUCTION_OUTER_HESSOPT)
    return nothing
end

# ============================================================================
# sigma3/W500k campaign addendum (2026-07-30): the campaign brief requires the OFFICIAL KNITRO
# Direct interior-point algorithm with SR1 as the primary outer strategy, with an optional
# Direct+BFGS polish stage -- explicitly NOT the CG+L-BFGS combination pinned above (that is a
# different, opt-in-only experimental config from a single controlled 2026-07-23 comparison, not
# what this campaign asked for). Neither existed as a *forceable* choice before this addendum:
# every pre-existing `.opt` file leaves `algorithm auto`, and this file's own module docstring
# documents that `auto` resolves INCONSISTENTLY by family (Direct for the CM-family constrained
# formulations, but Active-Set/CG for the unrestricted family's unconstrained profile
# formulation) -- so "leave it at auto" does not reliably give Direct at all, let alone
# Direct+SR1 specifically. These two constants/functions mirror the existing CG+L-BFGS pattern
# exactly (same KN_set_param_by_name calls, same explicit-readback assertion pattern below),
# just with the KNITRO codes this campaign actually needs: algorithm=1 (Direct interior/barrier),
# hessopt=3 (SR1) or hessopt=6 (BFGS).
# ============================================================================

"Direct interior-point/barrier algorithm (KNITRO code 1)."
const KNITRO_ALGORITHM_DIRECT = 1

"SR1 Hessian approximation (KNITRO code 3) -- this campaign's primary outer strategy."
const KNITRO_HESSOPT_SR1 = 3

"BFGS Hessian approximation (KNITRO code 6) -- this campaign's optional polish stage."
const KNITRO_HESSOPT_BFGS = 6

"""
    set_outer_algorithm_direct!(kc, hessopt::Int)

Explicitly sets the outer KNITRO context's `algorithm` to Direct (1) and `hessopt` to the given
code (KNITRO_HESSOPT_SR1 or KNITRO_HESSOPT_BFGS). Mirrors set_production_outer_algorithm!'s
mechanism exactly; a distinct function (not a generalization of that one) so the CG+L-BFGS
experimental config above is never silently touched by this campaign's own wiring.
"""
function set_outer_algorithm_direct!(kc, hessopt::Int)
    hessopt in (KNITRO_HESSOPT_SR1, KNITRO_HESSOPT_BFGS) ||
        error("set_outer_algorithm_direct!: hessopt must be KNITRO_HESSOPT_SR1 (3) or KNITRO_HESSOPT_BFGS (6), got $hessopt")
    KNITRO.KN_set_param_by_name(kc, "algorithm", KNITRO_ALGORITHM_DIRECT)
    KNITRO.KN_set_param_by_name(kc, "hessopt", hessopt)
    return nothing
end

"""
    assert_outer_algorithm_direct!(kc, hessopt::Int; context::String = "")

Reads back BOTH `algorithm` and `hessopt` and throws unless algorithm==Direct(1) and hessopt
matches exactly. Stricter than assert_outer_algorithm_explicit! below (which only checks
algorithm!=auto) because this campaign's requirement is not merely "explicit", it is
specifically Direct+SR1 or Direct+BFGS -- a caller must not silently end up at, say,
Active-Set+SR1 and have this pass.
"""
function assert_outer_algorithm_direct!(kc, hessopt::Int; context::String = "")
    algo_ref = Ref{Cint}(KNITRO_AUTO)
    status_a = KNITRO.KN_get_int_param_by_name(kc, "algorithm", algo_ref)
    status_a == 0 || error("assert_outer_algorithm_direct!($context): KN_get_int_param_by_name(\"algorithm\") returned nonzero status $status_a")
    algo_ref[] == KNITRO_ALGORITHM_DIRECT || error(
        "$context: expected outer algorithm=Direct($KNITRO_ALGORITHM_DIRECT), got $(algo_ref[]) -- " *
        "set_outer_algorithm_direct! should have run right after KN_load_param_file. This is a bug in the calling driver.")
    hess_ref = Ref{Cint}(KNITRO_AUTO)
    status_h = KNITRO.KN_get_int_param_by_name(kc, "hessopt", hess_ref)
    status_h == 0 || error("assert_outer_algorithm_direct!($context): KN_get_int_param_by_name(\"hessopt\") returned nonzero status $status_h")
    hess_ref[] == hessopt || error(
        "$context: expected outer hessopt=$hessopt, got $(hess_ref[]) -- set_outer_algorithm_direct! should have set this exactly.")
    return nothing
end

"""
    assert_outer_algorithm_explicit!(kc; context::String = "")

Reads back the OUTER KNITRO context's resolved `algorithm` param and throws a loud error if it
is still `auto` (0). Only called by drivers when `pin_outer_algorithm=true` -- i.e. only when a
caller has explicitly asked for a pinned, reproducible outer algorithm and the setter should
therefore never have left it at auto. Not called on the default (pin_outer_algorithm=false)
path, where a family's own pre-existing algorithm choice (possibly auto) is expected and correct.
"""
function assert_outer_algorithm_explicit!(kc; context::String = "")
    ref = Ref{Cint}(KNITRO_AUTO)
    status = KNITRO.KN_get_int_param_by_name(kc, "algorithm", ref)
    status == 0 || error(
        "assert_outer_algorithm_explicit!($context): KN_get_int_param_by_name for " *
        "\"algorithm\" returned nonzero status $status -- cannot verify the outer algorithm " *
        "is explicit; refusing to proceed to KN_solve.",
    )
    ref[] == KNITRO_AUTO && error(
        "$context: pin_outer_algorithm=true was requested but outer KNITRO `algorithm` still " *
        "resolved to auto (0) -- set_production_outer_algorithm!(kc) should have run right " *
        "after KN_load_param_file. This is a bug in the calling driver, not expected behavior.",
    )
    return nothing
end
