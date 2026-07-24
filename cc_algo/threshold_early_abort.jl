# Part C (2026-07-23 release, addendum): immediate mid-solve early-abort from a certified
# dual lower bound on the minimum divergence Delta*(theta).
#
# --- Sign identity (verified directly from PsiObjectiveBundleImplicit's callable, PsiObjectiveBundle.jl) ---
# KNITRO's inner (zeta,lambda) solve MINIMIZES
#     f(zeta,lambda) = sum(Psi(arg0))/M + zeta,      arg0 = -(H_moments * [zeta;lambda])
# This f is the negative of an unconstrained convex-conjugate dual objective
#     D_dual(zeta,lambda) := -f(zeta,lambda).
# Because (zeta,lambda) ranges freely over R^{d+1} with no dual-feasibility constraint to
# violate, weak duality holds POINTWISE, not just at the minimizer:
#     D_dual(zeta,lambda) <= Delta*(theta)   for EVERY (zeta,lambda) KNITRO ever evaluates.
# At the minimizer, strong duality gives Delta*(theta) = -min f = sup D_dual (this is the
# "Delta_dual" reported throughout this codebase). Consequently, at ANY evaluated iterate:
#     f(zeta,lambda) <= -threshold   <=>   D_dual(zeta,lambda) >= threshold   =>   Delta*(theta) >= threshold.
# This licenses terminating the inner solve the instant f <= -threshold is observed -- no need
# to wait for KN_solve to converge to the minimizer. This is a certified lower-bound rejection,
# not a heuristic based on an unconverged primal estimate.
#
# --- Termination mechanism ---
# KNITRO.jl's C_wrapper.jl `_try_catch_handler` translates a callback that throws
# `InterruptException()` into `KN_RC_USER_TERMINATION` (-504), a clean solver-level abort that
# KN_solve honors immediately. A `DomainError` instead maps to `KN_RC_EVAL_ERR`, which only
# triggers a KNITRO backtrack/retry at the SAME iterate -- not a termination -- so it is not
# usable for this purpose. A generic programming error (any other exception type) is left to
# propagate through `_try_catch_handler` as `KN_RC_CALLBACK_ERR` with a `@warn`, i.e. it stays
# visible rather than being silently absorbed.

"""
Typed result: a certified LOWER BOUND on Delta*(theta), established mid-solve from a single
valid dual iterate. Does NOT mean Delta*(theta) equals `lower_bound`, and does NOT mean the
outer point is infeasible for every delta -- only for delta < lower_bound (minus tolerance).
A finite Delta*=12 is infeasible for delta=2 but feasible for delta=20; never conflate this
with a structural ExactInfeasible certificate.
"""
struct CertifiedDivergenceLowerBound
    lower_bound ::Float64   # D_dual(zeta,lambda) = -f at the aborting iterate
    threshold   ::Float64   # delta_auto_reject_threshold active at abort time
end

"""
Per-objective-bundle mutable state for the threshold-early-abort feature.
`threshold = Inf` disables the feature entirely (zero behavior change vs. pre-Part-C code).
Per-solve fields (`triggered`, `lower_bound`) are reset at the start of each `inner_loop_KNITRO`
call; the aggregate counters (`n_aborts`, `smallest_lower_bound`, `largest_lower_bound`) persist
across solves for campaign-level observability reporting (task Part B step 10 / Part C step 9).
"""
mutable struct ThresholdAbortState
    threshold             ::Float64
    triggered             ::Bool
    lower_bound           ::Float64
    n_aborts              ::Int
    smallest_lower_bound  ::Float64
    largest_lower_bound   ::Float64
end
ThresholdAbortState(threshold::Float64 = Inf) =
    ThresholdAbortState(threshold, false, NaN, 0, Inf, -Inf)

function reset_for_new_solve!(st::ThresholdAbortState)
    st.triggered = false
    st.lower_bound = NaN
    return st
end

"""
    maybe_abort_on_threshold!(st, f) -> Nothing

Called from inside the KNITRO inner-solve objective callback with the raw objective value `f`
just computed at the current iterate. If the feature is enabled (`st.threshold` finite) and the
canonical positive dual lower bound `-f` has reached `st.threshold`, records the certificate and
throws `InterruptException()` to request an immediate clean KNITRO termination. No-op otherwise.
"""
function maybe_abort_on_threshold!(st::ThresholdAbortState, f::Float64)
    isfinite(st.threshold) || return nothing
    isfinite(f) || return nothing   # never trigger from a non-finite/invalid evaluation
    lb = -f
    if lb >= st.threshold
        st.triggered = true
        st.lower_bound = lb
        st.n_aborts += 1
        st.smallest_lower_bound = min(st.smallest_lower_bound, lb)
        st.largest_lower_bound = max(st.largest_lower_bound, lb)
        throw(InterruptException())
    end
    return nothing
end

# Generic fallback: bundle types other than PsiObjectiveBundleImplicit do not carry a
# threshold_state field and are therefore always disabled (nothing) -- byte-identical behavior
# to before this feature existed.
_threshold_state(obj) = nothing
_reset_threshold_for_new_solve!(obj) = nothing

"""
    threshold_abort_result(obj) -> Union{Nothing, CertifiedDivergenceLowerBound}

If the most recent `inner_loop_KNITRO(obj)` call was terminated by the threshold-early-abort
mechanism, returns the typed certificate; otherwise `nothing` (including when the feature is
disabled, or the solve completed/failed for an unrelated reason).
"""
function threshold_abort_result(obj)
    st = _threshold_state(obj)
    (st === nothing || !st.triggered) && return nothing
    return CertifiedDivergenceLowerBound(st.lower_bound, st.threshold)
end

"""
    resolve_threshold_for_delta(requested_delta; base_threshold=10.0, safety_margin=1.0) -> Float64

Task Part C step 12/6 compatibility rule: the threshold-10 shortcut is valid only when the
requested neighborhood satisfies `requested_delta < base_threshold - safety_margin`. For runs
at or above that bound, the shortcut must be disabled (returns `Inf`) so a potentially feasible
point at large delta is never rejected from a threshold-10 certificate. Current paper campaigns
(`delta <= 2`) resolve to exactly `base_threshold` (10.0).
"""
function resolve_threshold_for_delta(requested_delta::Real; base_threshold::Float64 = 10.0,
                                      safety_margin::Float64 = 1.0)
    return requested_delta < (base_threshold - safety_margin) ? base_threshold : Inf
end

"""
    threshold_permits_reject(cert::CertifiedDivergenceLowerBound, requested_delta; safety_tolerance=1e-6) -> Bool

Cross-delta cache reuse rule (task Part C step 5 / addendum step 5): a cached certified lower
bound may reject a *different* requested delta without re-solving only when
`requested_delta < cert.lower_bound - safety_tolerance`. It may never reject when
`requested_delta >= cert.lower_bound`.
"""
function threshold_permits_reject(cert::CertifiedDivergenceLowerBound, requested_delta::Real;
                                   safety_tolerance::Float64 = 1e-6)
    return requested_delta < cert.lower_bound - safety_tolerance
end
