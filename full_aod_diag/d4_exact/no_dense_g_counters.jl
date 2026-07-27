# ================================================================================================
# port/shared-inner-fg-operator-and-verification-2026-07-26, §12: runtime counters proving (or
# disproving) the "no full G materialization in production hot paths" invariant, PLUS an opt-in
# fail-fast guard that throws the instant a production hot path attempts one.
#
# SCOPE (honest, not oversold): these counters are wired at the call sites this branch itself
# built or touched -- economic_operator.jl's economic_forward!/economic_transpose!,
# zc_restriction_operator.jl's restriction_forward!/restriction_transpose!, and the operator-vs-
# dense-reference dispatch points for origin-ZC/CM+ZC (this branch's two NEW operator families) and
# flexible CM/common-Fréchet (the two ALREADY-lookup-capable families, dispatch points instrumented
# too). This is NOT a full-codebase audit (task §13's "search the repository for every dense
# moment builder/gemv!/select_G_from_H/generic verifier/diagnostic scorer" is a separate, larger,
# NOT-YET-DONE item -- see docs/NO_DENSE_G_RUNTIME_PROOF_2026-07-26.md's own explicit gap list).
# What these counters DO prove, honestly: every inner-FG call this branch's own operator code paths
# serve is accounted for as `operator_*`, and the pre-existing dense-reference paths (still the
# default for 3/5 families) are accounted for as `generic_dense_FG_calls`/`dense_economic_G_materializations`.
#
# Winner-aware H_ER phase (2026-07-27): extended with the cross-Hessian (H_ER) backend-use
# counters, `operator_economic_FG_calls`/`operator_restriction_FG_calls`, and
# `dense_Frechet_G_materializations` -- consolidated HERE (not a separate Ref elsewhere) so every
# "no dense G" counter this project tracks lives in one place, per that task's own Section 7 list.
# ================================================================================================

Base.@kwdef mutable struct NoDenseGCounters
    full_G_materializations::Int = 0
    dense_economic_G_materializations::Int = 0
    dense_CM_G_materializations::Int = 0
    dense_ZC_G_materializations::Int = 0
    dense_Frechet_G_materializations::Int = 0
    generic_dense_FG_calls::Int = 0
    operator_FG_calls::Int = 0
    operator_forward_calls::Int = 0
    operator_transpose_calls::Int = 0
    operator_economic_FG_calls::Int = 0
    operator_restriction_FG_calls::Int = 0
    operator_verification_calls::Int = 0
    dense_reference_verification_calls::Int = 0
    # winner-aware H_ER cross-Hessian counters. `winner_cross_hessian_calls` is kept numerically
    # identical to `operator_cross_hessian_calls` (both incremented together by
    # record_winner_cross_hessian_call!) -- the task brief's own §7 counter list names both
    # separately; there is no semantic difference, only two names for the same event.
    dense_cross_hessian_calls::Int = 0
    operator_cross_hessian_calls::Int = 0
    winner_cross_hessian_calls::Int = 0
end

const NO_DENSE_G_COUNTERS = Ref(NoDenseGCounters())

"Reset every counter to zero -- call at the start of a gate/benchmark run that wants a clean count."
reset_no_dense_g_counters!() = (NO_DENSE_G_COUNTERS[] = NoDenseGCounters(); nothing)

"""
    FAIL_FAST_ON_DENSE_G

Opt-in guard (task §12, "an opt-in fail-fast mode that throws if a production hot path attempts to
materialize full G"). `false` by default (counting only, never throws) -- flip to `true` in a gate
script that wants a hard assertion. Checked by `record_dense_economic_G!`/`record_generic_dense_fg!`
below, NOT by the Hessian's own legitimate dense-H reads (those are explicitly out of this
invariant's scope -- see economic_operator.jl's header for why the Hessian's H_EC/H_ER cross-terms
still require dense H, an out-of-scope Hessian-cross-block dependency, not a violation of this
invariant, which is about the FG *callback's own* consumption).
"""
const FAIL_FAST_ON_DENSE_G = Ref(false)

"Call from economic_forward!/economic_transpose! (economic_operator.jl) every invocation."
function record_operator_forward!()
    NO_DENSE_G_COUNTERS[].operator_forward_calls += 1
    NO_DENSE_G_COUNTERS[].operator_FG_calls += 1
    return nothing
end
function record_operator_transpose!()
    NO_DENSE_G_COUNTERS[].operator_transpose_calls += 1
    return nothing
end

"Call from a family's :dense_reference FG dispatch branch (e.g. inner_loop_internal_archgeneric callers) every inner solve."
function record_generic_dense_fg!()
    NO_DENSE_G_COUNTERS[].generic_dense_FG_calls += 1
    FAIL_FAST_ON_DENSE_G[] &&
        error("record_generic_dense_fg!: a production hot path dispatched to the dense-reference FG callback while FAIL_FAST_ON_DENSE_G[]=true")
    return nothing
end

"Call whenever an FG callback's economic (E) block reads dense obj.H directly (the tied-winner/compressed-unavailable fallback in OriginZCOperatorState/CMMeanZCOperatorState, or a family that hasn't been ported to the operator yet)."
function record_dense_economic_G!()
    NO_DENSE_G_COUNTERS[].dense_economic_G_materializations += 1
    FAIL_FAST_ON_DENSE_G[] &&
        error("record_dense_economic_G!: a production hot path materialized/read dense economic G while FAIL_FAST_ON_DENSE_G[]=true")
    return nothing
end

"Call whenever an FG callback's CM-grid block reads dense obj.H columns directly (the :dense_reference CM/Fréchet/CM+ZC path)."
record_dense_cm_g!() = (NO_DENSE_G_COUNTERS[].dense_CM_G_materializations += 1; nothing)

"Call whenever an FG callback's ZC (mean/pair) block reads dense obj.H columns directly (the :dense_reference origin-ZC/CM+ZC path)."
record_dense_zc_g!() = (NO_DENSE_G_COUNTERS[].dense_ZC_G_materializations += 1; nothing)

"Call whenever common-Fréchet's level-anchor block reads dense obj.H columns directly (the :dense_reference path)."
record_dense_frechet_g!() = (NO_DENSE_G_COUNTERS[].dense_Frechet_G_materializations += 1; nothing)

"Call from operator-based verification (verify_inner_solution_operator!)."
record_operator_verification!() = (NO_DENSE_G_COUNTERS[].operator_verification_calls += 1; nothing)

"Call from dense-reference verification (the explicit :dense_reference debug backend)."
record_dense_reference_verification!() = (NO_DENSE_G_COUNTERS[].dense_reference_verification_calls += 1; nothing)

"""
    record_winner_cross_hessian_call!() / record_dense_cross_hessian_call!()

Winner-aware H_ER phase (2026-07-27): call from a family's cross-Hessian (H_ER) backend dispatch
point every Hessian callback -- mirrors `record_operator_forward!`/`record_generic_dense_fg!`'s own
discipline, one call site per decision, no silent path.
"""
function record_winner_cross_hessian_call!()
    c = NO_DENSE_G_COUNTERS[]
    c.winner_cross_hessian_calls += 1
    c.operator_cross_hessian_calls += 1
    return nothing
end
function record_dense_cross_hessian_call!()
    NO_DENSE_G_COUNTERS[].dense_cross_hessian_calls += 1
    return nothing
end

"""
    no_dense_g_report() -> NamedTuple

Snapshot of every counter, for a gate script to print/assert against.
"""
function no_dense_g_report()
    c = NO_DENSE_G_COUNTERS[]
    return (full_G_materializations = c.full_G_materializations,
            dense_economic_G_materializations = c.dense_economic_G_materializations,
            dense_CM_G_materializations = c.dense_CM_G_materializations,
            dense_ZC_G_materializations = c.dense_ZC_G_materializations,
            dense_Frechet_G_materializations = c.dense_Frechet_G_materializations,
            generic_dense_FG_calls = c.generic_dense_FG_calls,
            operator_FG_calls = c.operator_FG_calls,
            operator_forward_calls = c.operator_forward_calls,
            operator_transpose_calls = c.operator_transpose_calls,
            operator_economic_FG_calls = c.operator_economic_FG_calls,
            operator_restriction_FG_calls = c.operator_restriction_FG_calls,
            operator_verification_calls = c.operator_verification_calls,
            dense_reference_verification_calls = c.dense_reference_verification_calls,
            dense_cross_hessian_calls = c.dense_cross_hessian_calls,
            operator_cross_hessian_calls = c.operator_cross_hessian_calls,
            winner_cross_hessian_calls = c.winner_cross_hessian_calls)
end
