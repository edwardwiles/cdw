using Dates
using Serialization

# 2026-07-28 inner-solver architecture-consolidation session. The ONE Melitz-owned wrapper
# that pairs an already-constructed inner bundle (`PsiObjectiveBundleDelta`/
# `PsiObjectiveBundleImplicit` -- cc_algo, dense, shared with the Ricardian model, never
# modified here -- or `MelitzCCBundle` -- cc_bundle.jl, Melitz-owned, matrix-free) with the
# resolved `MelitzInnerSolvePolicy` (inner_solve_policy.jl) that governs it and its own
# dual-bank warm-start state.
#
# Placed in the include order AFTER inner_screening.jl (needs the concrete `MelitzDualBank`
# type for its own `bank` field) -- `session` itself is passed UNTYPED into every consumer
# function (this codebase's own established "zero function signatures type-annotate obj/
# session" duck-typing convention, cc_bundle.jl's file header), so there is no circular
# include-order dependency the other way: inner_screening.jl's own classifier function reads
# `session.obj`/`session.ctx`/`session.policy`/`session.bank` via plain field access, never
# `session::MelitzInnerSession` in its own signature.
#
# Deliberately NOT a deeper restructuring of `MelitzCCBundle`/the legacy dense bundles
# themselves into a separate "pure economic context" object -- `MelitzCCBundle.op`
# (`MelitzMomentOperator`) already IS that separated economic state (no solver-state field
# lives there), and the legacy dense bundles are a cc_algo (Ricardian) type this session
# cannot restructure at all. `MelitzInnerSession` is the new piece this architecture actually
# needed: a single object that owns the POLICY decision and the warm-start bank, decoupled
# from the fixture/calibration constructors that build `obj`/`ctx` (Section 4's genuine ask --
# "fixture and calibration constructors must not bake in a cap mode" -- satisfied by moving
# cap-mode ownership OUT of every one of those constructors' own silent defaults and into this
# one explicit object built alongside, never inside, a fixture).

"""
    MelitzInnerSession(obj, ctx, policy::MelitzInnerSolvePolicy; dual_bank_max_size=8, bank=nothing)

The one session object `solve_melitz_delta!` operates on. Asserts `obj.lower_limit ==
melitz_policy_lower_limit(policy)` at construction (Section 9's own "capped policy => raw
lower_limit == -cap" invariant, checked here in ADDITION to solve time so a session can never
even be constructed out of sync with its own declared policy) -- `obj` must already have been
built with `policy` (every constructor this session touches -- `build_melitz_cc_bundle`,
`build_melitz_implicit_bundle`, `build_melitz_psi_bundle`, `build_melitz_psi_bundle_from_calibration`
-- takes `policy` as a mandatory keyword and sets `lower_limit` from it, so this always holds
for any bundle built through this codebase's own sanctioned constructors).

`bank`, if given, lets a caller share a pre-existing `MelitzDualBank` across sessions/policies
deliberately (e.g. a benchmark comparing warm-start policies) -- default `nothing` allocates a
FRESH, empty bank sized `dual_bank_max_size`, matching every existing driver's own prior
per-outer-solve bank scoping convention (`finite_delta_outer.jl`'s `dual_bank =
MelitzDualBank(dual_bank_max_size)`).

**Never share one session (and never reuse the SAME dual bank) between a `CappedEvaluation`
and a `FullValueEvaluation` use.** A dual vector that made an uncapped pursuit run away to an
astronomical magnitude (exactly this session's own anomaly, `Delta~1.5e14`, `||x||~2.2e17`) is
a legitimate, weak-duality-valid bank entry for ANOTHER full-value solve, but would poison a
subsequent capped solve's warm start with a numerically catastrophic starting point (Phase 4
of `docs/melitz_finitesolved_anomaly_and_participation_diagnostic_2026-07-28.md` shows even a
freshly-capped re-solve from a stale runaway warm start returns `NumericalFailure`, not a clean
`AboveEvaluationCap`, purely from the poisoned starting dual) -- construct an independent
session (and let a fresh `MelitzDualBank` be allocated) per policy, never reuse one across
policy switches.
"""
mutable struct MelitzInnerSession
    obj::Any
    ctx::Any
    policy::MelitzInnerSolvePolicy
    bank::MelitzDualBank
end

function MelitzInnerSession(obj, ctx, policy::MelitzInnerSolvePolicy;
                             dual_bank_max_size::Int=8,
                             bank::Union{Nothing,MelitzDualBank}=nothing)
    _melitz_assert_session_policy_consistent(obj, policy)
    resolved_bank = bank === nothing ? MelitzDualBank(dual_bank_max_size) : bank
    return MelitzInnerSession(obj, ctx, policy, resolved_bank)
end

"""
    _melitz_assert_session_policy_consistent(obj, policy::MelitzInnerSolvePolicy)

Section 9's central invariant, callable independently of `MelitzInnerSession` construction
(also re-checked at the top of `solve_melitz_delta!` itself, defense-in-depth against a
session's `obj`/`policy` fields being mutated out of sync after construction -- Julia does not
make `MelitzInnerSession`'s fields immutable, since `obj`/`policy` legitimately need to be
swappable across a longer-lived session, e.g. a continuation script raising the cap between
stages).
"""
function _melitz_assert_session_policy_consistent(obj, policy::MelitzInnerSolvePolicy)
    expected = melitz_policy_lower_limit(policy)
    @assert obj.lower_limit == expected (
        "MelitzInnerSession: obj.lower_limit=$(obj.lower_limit) does not match " *
        "melitz_policy_lower_limit(policy)=$expected for policy=$policy -- obj was not built " *
        "with this policy (every sanctioned Melitz bundle constructor takes policy as a " *
        "mandatory keyword and derives lower_limit from it; this obj either predates this " *
        "session's API or had its lower_limit mutated directly, bypassing the policy).")
    return true
end

"""
    solve_melitz_delta!(session::MelitzInnerSession, theta, policy::MelitzInnerSolvePolicy;
        range_screen=true, matrix_free_range_screen=true, stored_dual_screen=true,
        dual_polish_screen=false, dual_polish_steps=3, origin_block_screen=false,
        screen_order=:A, warm_start_source=:previous, on_result=nothing) -> MelitzInnerResult

**The one public, authoritative Melitz inner-solve entry point (governing prompt Section 3).**
Every production and diagnostic caller in this codebase now routes through this function --
`_melitz_classified_inner_solve!` (`inner_screening.jl`), `melitz_bundle_inner_solve!`/
`melitz_cc_inner_loop_knitro!` (`cc_bundle.jl`) are internal from here on (Section 6), called
only from inside this function or from each other.

`policy` is passed AGAIN here, redundantly with `session.policy` by design (Section 9): this
function asserts `policy == session.policy` before doing anything else, so a caller cannot
silently solve a session under a DIFFERENT policy than the one it was built for (e.g. copy-
pasting a capped session into a spot that meant to request full-value evaluation) -- catching
that mismatch at the call site that got it wrong, not three functions downstream.

This function alone:
  1. asserts the Section 9 policy/session/lower_limit invariants (see
     `_melitz_assert_session_policy_consistent`/the inline checks below);
  2. resets `session.obj.threshold_crossed[]` (via `_melitz_classified_inner_solve!`, which
     performs this reset immediately before every attempt, unconditionally);
  3. selects/applies the requested warm start (`melitz_resolve_warm_start!`);
  4. runs the exact prescreens (range, stored-dual, dual-polish, origin-block, per
     `screen_order`) and the one (no-retry) low-level KNITRO attempt --
     `_melitz_classified_inner_solve!`'s own existing orchestration, unchanged in substance
     from the pre-existing `melitz_classified_inner_solve`, only rewired to read
     `delta_evaluation_cap`/`bank`/`obj`/`ctx` off `session`/`session.policy` instead of
     independent by-value arguments;
  5. recovers and verifies the LFD implicitly via the classifier's own `FiniteSolved`-above-cap
     output invariant (Section 7, `inner_screening.jl`);
  6. creates the typed `MelitzInnerResult`.

Deliberately NOT a new algorithm: this is the SAME classification/screening logic
`melitz_classified_inner_solve` already implemented, given a new mandatory entry point so the
cap can no longer be requested as one number (the old `delta_evaluation_cap` argument) while a
DIFFERENT number governs the live KNITRO threshold (`obj.lower_limit`, fixed by whichever of
several constructors built `obj`) -- the root cause of this session's governing anomaly. Under
this API there is only one number (`session.policy`'s own `cap`/`melitz_policy_lower_limit`),
read from the SAME object at both construction and solve time.
"""
function solve_melitz_delta!(session::MelitzInnerSession, theta::AbstractVector,
                              policy::MelitzInnerSolvePolicy;
                              range_screen::Bool=true,
                              matrix_free_range_screen::Bool=true,
                              stored_dual_screen::Bool=true,
                              dual_polish_screen::Bool=false,
                              dual_polish_steps::Int=3,
                              origin_block_screen::Bool=false,
                              screen_order::Symbol=:A,
                              warm_start_source::Symbol=:previous,
                              on_result=nothing)::MelitzInnerResult
    # Section 9: "session policy matches requested policy."
    policy == session.policy || throw(ArgumentError(
        "solve_melitz_delta!: requested policy ($policy) does not match session.policy " *
        "($(session.policy)) -- construct a NEW MelitzInnerSession for a different policy " *
        "rather than solving an existing session under a policy it was not built for " *
        "(a stale warm-start dual from one policy's own trajectory is not safe to reuse " *
        "under a different one -- see MelitzInnerSession's own docstring)."))
    _melitz_assert_session_policy_consistent(session.obj, session.policy)

    # Section 13: archive pathological (very slow) attempts, timed around the ONE call this
    # function ever makes into the low-level classifier -- adds a single `time_ns()` pair
    # (already this codebase's own established zero-cost-when-unused timing idiom,
    # `profiling.jl`) on every call, and the archive-write itself only runs on the rare
    # slow-path branch, so ordinary hot-path performance is unaffected.
    starting_dual = copy(session.obj.x)
    t0_solve = time_ns()
    result = _melitz_classified_inner_solve!(session, theta;
        range_screen=range_screen, matrix_free_range_screen=matrix_free_range_screen,
        stored_dual_screen=stored_dual_screen, dual_polish_screen=dual_polish_screen,
        dual_polish_steps=dual_polish_steps, origin_block_screen=origin_block_screen,
        screen_order=screen_order, warm_start_source=warm_start_source, on_result=on_result)
    elapsed_s = (time_ns() - t0_solve) / 1e9
    elapsed_s >= MELITZ_PATHOLOGICAL_SOLVE_THRESHOLD_S[] &&
        _melitz_archive_pathological_solve!(session, theta, policy, starting_dual, elapsed_s, result)
    return result
end

"""
    MELITZ_PATHOLOGICAL_SOLVE_THRESHOLD_S

Section 13's configurable threshold (seconds) -- any `solve_melitz_delta!` attempt taking at
least this long has its diagnostic state archived (`_melitz_archive_pathological_solve!`
below). Default `10.0` (the governing prompt's own example value). Set to `Inf` to disable
archiving entirely (e.g. for a benchmark script that does not want the archive directory
written to).
"""
const MELITZ_PATHOLOGICAL_SOLVE_THRESHOLD_S = Ref(10.0)

"""
    MELITZ_PATHOLOGICAL_ARCHIVE_DIR

Directory `_melitz_archive_pathological_solve!` writes to -- created on first use if it does
not already exist. Default: `<repo root>/melitz_pathological_solve_archive/`.
"""
const MELITZ_PATHOLOGICAL_ARCHIVE_DIR = Ref(joinpath(@__DIR__, "..", "..", "melitz_pathological_solve_archive"))

"""
    MelitzPathologicalSolveRecord

Section 13's archived record for one slow `solve_melitz_delta!` attempt: `theta`, `policy`,
`cap`/`lower_limit` (both derived from `policy` -- recorded explicitly, not merely
recoverable, so an archived record remains self-contained even if read back after `policy`'s
own definition later changes), `starting_dual` (a copy of `session.obj.x` taken BEFORE the
attempt -- the closest available proxy for an "objective trace start point"; see this
struct's own docstring continuation below for the disclosed limitation on a full per-iteration
trace), `context_fingerprint` (`melitz_context_fingerprint`), `elapsed_s`, `nStatus` (`-1` if
the result type carries none, e.g. `InfiniteDeltaCertified`), `classification` (the result
type's own name, e.g. `:FiniteSolved`), and `timestamp` (ISO-8601, wall-clock capture time).

**Disclosed limitation on "objective trace"**: the governing prompt asks for an "objective
trace" alongside the other fields. A genuine PER-ITERATION trace would require threading a
recording callback into the hot KNITRO objective/gradient/Hessian functor
(`(Q::MelitzCCBundle)`'s own callback body, `cc_bundle.jl`, or the shared
`cc_algo/PsiObjectiveBundle.jl` functor for the legacy dense bundles) -- exactly the
"without affecting ordinary hot-path performance materially" constraint this session's own
prompt places on this feature warns against modifying casually. This record instead captures
a cheap TWO-POINT proxy computed only on the rare slow-path branch (never inside the hot
callback): the raw functor value at `starting_dual` (before the attempt) and at the final
`objective_value` KNITRO reports at whatever iterate it stopped at (after) -- enough to see
the gross direction/magnitude of movement without touching the hot path at all. A full
per-iteration trace is a disclosed, not-implemented follow-up, not a silent omission.
"""
struct MelitzPathologicalSolveRecord
    theta::Vector{Float64}
    policy::MelitzInnerSolvePolicy
    cap::Float64
    lower_limit::Float64
    starting_dual::Vector{Float64}
    objective_at_start::Float64
    objective_at_end::Float64
    context_fingerprint::UInt
    elapsed_s::Float64
    nStatus::Int
    classification::Symbol
    timestamp::String
end

_melitz_result_nstatus(r::FiniteSolved) = r.nStatus
_melitz_result_nstatus(r::NumericalFailure) = r.nStatus
_melitz_result_nstatus(r) = -1   # AboveEvaluationCap/InfiniteDeltaCertified/BoundaryFeasible carry no nStatus

"""
    _melitz_archive_pathological_solve!(session, theta, policy, starting_dual, elapsed_s, result)

Writes one `MelitzPathologicalSolveRecord` (Julia `Serialization`, `.jls`) to
`MELITZ_PATHOLOGICAL_ARCHIVE_DIR[]` and prints a loud `@warn` naming the file -- called ONLY
from `solve_melitz_delta!`, ONLY when `elapsed_s >= MELITZ_PATHOLOGICAL_SOLVE_THRESHOLD_S[]`
(the rare slow-path branch this function exists to make visible/debuggable after the fact).
Never throws on a filesystem error (archiving is a best-effort diagnostic aid, never allowed
to turn a slow-but-real solve into a harder failure) -- catches and `@warn`s instead.
"""
function _melitz_archive_pathological_solve!(session, theta::AbstractVector, policy::MelitzInnerSolvePolicy,
                                              starting_dual::AbstractVector, elapsed_s::Real, result)
    try
        obj = session.obj
        objective_at_start = try
            all(isfinite, starting_dual) && !isempty(starting_dual) ? Float64(obj(starting_dual)) : NaN
        catch
            NaN
        end
        objective_at_end = result isa FiniteSolved ? -result.Delta :
                            (hasproperty(result, :x) && !isempty(result.x) && all(isfinite, result.x)) ?
                                (try Float64(obj(result.x)) catch; NaN end) : NaN
        rec = MelitzPathologicalSolveRecord(
            collect(Float64.(theta)), policy, melitz_policy_cap(policy), melitz_policy_lower_limit(policy),
            collect(Float64.(starting_dual)), objective_at_start, objective_at_end,
            melitz_context_fingerprint(session.ctx, hasproperty(obj, :U) ? obj.U : nothing),
            Float64(elapsed_s), _melitz_result_nstatus(result), Symbol(nameof(typeof(result))),
            string(Dates.now()))
        dir = MELITZ_PATHOLOGICAL_ARCHIVE_DIR[]
        isdir(dir) || mkpath(dir)
        fname = joinpath(dir, "pathological_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSSsss"))_$(objectid(rec)).jls")
        open(fname, "w") do io
            Serialization.serialize(io, rec)
        end
        @warn "melitz: pathological inner solve archived (elapsed_s=$elapsed_s >= $(MELITZ_PATHOLOGICAL_SOLVE_THRESHOLD_S[]))" fname
    catch e
        @warn "melitz: failed to archive pathological solve (non-fatal, the solve result itself is unaffected)" exception=(e, catch_backtrace())
    end
    return nothing
end
