using KNITRO

# 2026-07-28 inner-solver architecture-consolidation session. Replaces
# `MelitzInnerSolveConfig`/`melitz_configure_lower_limit`/`melitz_assert_evaluation_cap_active`
# (the file this one supersedes, formerly `inner_solve_config.jl`) with a TYPED policy
# hierarchy, per the governing prompt's own Section 2.
#
# Why the symbol-mode config was not enough: `MelitzInnerSolveConfig` already REQUIRED an
# explicit mode/cap at ITS OWN construction site -- a genuine improvement over the raw
# `Float64 lower_limit` it replaced. But it was still a `Union{Nothing,MelitzInnerSolveConfig}`
# KEYWORD ARGUMENT on every wrapper constructor (`build_melitz_implicit_bundle`,
# `build_melitz_psi_bundle`, `build_melitz_psi_bundle_from_calibration`), each of which
# defaulted the keyword to `nothing` and then translated a missing config into
# `-KNITRO.KN_INFINITY` internally -- exactly the "translate a missing config into -Inf"
# failure mode this session exists to close (confirmed live,
# docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md Phase 2/6).
# `MelitzInnerSolveConfig` itself was never the bug; the `=nothing` DEFAULT wrapped around it,
# at every construction call site, was.
#
# This file's own types have NO analogous escape hatch: `CappedEvaluation`/`FullValueEvaluation`
# are the ONLY two ways to obtain a resolved policy, both fully validating at construction, and
# every bundle-constructing function this session touches now takes a mandatory
# `policy::MelitzInnerSolvePolicy` keyword with NO default at all -- omitting it is an
# `UndefKeywordError`, not a silently-resolved uncapped bundle.

"""
    MelitzInnerSolvePolicy

Abstract supertype for a fully-resolved Melitz inner-solve policy. Exactly two concrete
subtypes exist -- `CappedEvaluation` (the production routine-inner-solve mode) and
`FullValueEvaluation` (a deliberate, explicitly-named, uncapped mode) -- constructed via their
own validating outer constructors below, never via a raw positional call that skips validation.

There is deliberately no third "default"/"unspecified" policy and no `Union{Nothing,...}`
anywhere in this file: every function that needs a policy takes one as a mandatory
`policy::MelitzInnerSolvePolicy` argument.
"""
abstract type MelitzInnerSolvePolicy end

"""
    CappedEvaluation(cap; max_iterations=10_000, max_seconds=90.0) <: MelitzInnerSolvePolicy

The production routine-inner-solve policy (governing prompt Sections 1-2): every inner
KNITRO solve run under this policy has its KNITRO-native mid-solve bailout
(`lower_limit = -cap`, `melitz_policy_lower_limit` below) armed BEFORE `KN_solve` is ever
called, and its `AboveEvaluationCap`/`FiniteSolved` classification derives the cap from THIS
object (`melitz_policy_cap`), never from a second, independently-supplied number that could
drift out of sync with `lower_limit` (the exact anomaly this session's governing prompt
diagnoses).

`max_iterations`/`max_seconds` are applied to the live KNITRO instance by
`solve_melitz_delta!`/`melitz_cc_inner_loop_knitro!` (`cc_bundle.jl`) for the Melitz-owned
matrix-free bundle (`MelitzCCBundle`) -- `KN_set_int_param_by_name(kc, "maxit", ...)` and
`KN_set_double_param_by_name(kc, "maxtime_real", ...)`, applied AFTER the bundle's own
`inner_loop_opt` file is loaded, so this policy's values win over whatever the static option
file says. For the legacy dense bundles (`PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit`,
`cc_algo`, shared with the Ricardian model, never modified here), the underlying KNITRO
driver (`cc_algo/inner_loop_functions.jl`'s `inner_loop_KNITRO`) is Ricardian-owned code this
session does not touch -- `max_iterations`/`max_seconds` are therefore NOT dynamically
applied on that path; only the static `inner_loop_opt` file's own `maxit`/`maxtime_real`
govern it, exactly as before this session. This is a real, disclosed limitation of the
Ricardian ownership boundary, not an oversight -- see
`docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`. `cap` (the divergence
early-abort threshold, i.e. `lower_limit`) applies uniformly to BOTH bundle families
regardless, since `lower_limit` is a plain mutable field on both.
"""
struct CappedEvaluation <: MelitzInnerSolvePolicy
    cap::Float64
    max_iterations::Int
    max_seconds::Float64
end

function CappedEvaluation(cap::Real; max_iterations::Integer=10_000, max_seconds::Real=90.0)
    capf = Float64(cap)
    isfinite(capf) || throw(ArgumentError("CappedEvaluation: cap must be finite, got $capf"))
    capf > 0 || throw(ArgumentError("CappedEvaluation: cap must be strictly positive (it is a " *
        "divergence-magnitude threshold), got $capf"))
    Int(max_iterations) > 0 || throw(ArgumentError("CappedEvaluation: max_iterations must be " *
        "strictly positive, got $max_iterations"))
    max_secondsf = Float64(max_seconds)
    isfinite(max_secondsf) && max_secondsf > 0 || throw(ArgumentError(
        "CappedEvaluation: max_seconds must be finite and strictly positive, got $max_secondsf"))
    return CappedEvaluation(capf, Int(max_iterations), max_secondsf)
end

"""
    FullValueEvaluation(; max_iterations=10_000, max_seconds=1e8) <: MelitzInnerSolvePolicy

Deliberately uncapped (`lower_limit = -KNITRO.KN_INFINITY`, `melitz_policy_lower_limit`
below) -- valid ONLY when a caller explicitly wants the true, fully-optimized inner value
regardless of magnitude (a cold end-of-run reverification, a bounded D=4 unit-test fixture
where runaway divergence has never been observed, or a deliberate diagnostic comparison).
Constructing this type IS the explicit, named choice this session's governing prompt
requires -- never a default a caller falls into by omission (no function in this codebase
defaults its `policy` keyword to `FullValueEvaluation()` -- every production/test call site
must name it).

`melitz_policy_cap(::FullValueEvaluation) = Inf`: `AboveEvaluationCap` can never be produced
and the `FiniteSolved`-above-cap invariant (`inner_screening.jl`) is vacuously satisfied under
this policy, exactly as intended for an uncapped evaluation.
"""
struct FullValueEvaluation <: MelitzInnerSolvePolicy
    max_iterations::Int
    max_seconds::Float64
end

function FullValueEvaluation(; max_iterations::Integer=10_000, max_seconds::Real=1e8)
    Int(max_iterations) > 0 || throw(ArgumentError("FullValueEvaluation: max_iterations must be " *
        "strictly positive, got $max_iterations"))
    max_secondsf = Float64(max_seconds)
    isfinite(max_secondsf) && max_secondsf > 0 || throw(ArgumentError(
        "FullValueEvaluation: max_seconds must be finite and strictly positive, got $max_secondsf"))
    return FullValueEvaluation(Int(max_iterations), max_secondsf)
end

Base.:(==)(a::CappedEvaluation, b::CappedEvaluation) =
    a.cap == b.cap && a.max_iterations == b.max_iterations && a.max_seconds == b.max_seconds
Base.:(==)(a::FullValueEvaluation, b::FullValueEvaluation) =
    a.max_iterations == b.max_iterations && a.max_seconds == b.max_seconds
Base.:(==)(a::MelitzInnerSolvePolicy, b::MelitzInnerSolvePolicy) = false   # different concrete types never equal

function Base.show(io::IO, p::CappedEvaluation)
    print(io, "CappedEvaluation(cap=", p.cap, ", max_iterations=", p.max_iterations,
              ", max_seconds=", p.max_seconds, ")")
end
function Base.show(io::IO, p::FullValueEvaluation)
    print(io, "FullValueEvaluation(max_iterations=", p.max_iterations,
              ", max_seconds=", p.max_seconds, ")")
end

"""
    melitz_policy_lower_limit(policy::MelitzInnerSolvePolicy) -> Float64

The one authoritative computation of the KNITRO-native `lower_limit` early-bailout threshold
from a resolved policy -- `-cap` exactly under `CappedEvaluation` (no guard/margin, matching
this codebase's own established no-margin convention), `-KNITRO.KN_INFINITY` under
`FullValueEvaluation`.
"""
melitz_policy_lower_limit(policy::CappedEvaluation) = -policy.cap
melitz_policy_lower_limit(::FullValueEvaluation) = -KNITRO.KN_INFINITY

"""
    melitz_policy_cap(policy::MelitzInnerSolvePolicy) -> Float64

The cap value used by the classifier (`inner_screening.jl`) for the `AboveEvaluationCap`
screening threshold and the `FiniteSolved`-above-cap output invariant -- `policy.cap` under
`CappedEvaluation`, `Inf` under `FullValueEvaluation` (so no point can ever be classified
`AboveEvaluationCap` and the output invariant `Delta <= cap` is vacuous, exactly the intended
semantics of a deliberately uncapped policy).
"""
melitz_policy_cap(policy::CappedEvaluation) = policy.cap
melitz_policy_cap(::FullValueEvaluation) = Inf

"""
    melitz_apply_policy_to_knitro!(kc, policy::MelitzInnerSolvePolicy)

Applies `policy.max_iterations`/`policy.max_seconds` directly to a live KNITRO instance `kc`
via `KN_set_int_param_by_name`/`KN_set_double_param_by_name` -- call AFTER `KN_load_param_file`
so this policy's values win over whatever the static `.opt` file says. Used by the Melitz-owned
matrix-free driver (`melitz_cc_inner_loop_knitro!`, `cc_bundle.jl`) only -- the legacy dense
bundles' KNITRO instance is constructed inside `cc_algo/inner_loop_functions.jl`
(Ricardian-owned, never touched here), so this function is never called on that path; see
`CappedEvaluation`'s own docstring for the disclosed consequence.
"""
function melitz_apply_policy_to_knitro!(kc, policy::MelitzInnerSolvePolicy)
    KNITRO.KN_set_int_param_by_name(kc, "maxit", Cint(policy.max_iterations))
    KNITRO.KN_set_double_param_by_name(kc, "maxtime_real", Float64(policy.max_seconds))
    return nothing
end
