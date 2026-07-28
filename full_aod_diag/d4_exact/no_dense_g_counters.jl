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
    # Final-architecture-closure task (2026-07-27), §11/§13: family-specific structured/direct
    # restriction-self-block (H_RR/H_CC) computations -- e.g. hessian_cm_structured!/_v2!'s own
    # H_CC raw+congruence block, the shared zc_restriction_gram! H_ZZ primitive when called for its
    # OWN self-block role rather than as an H_EZ/H_CZ cross term. These are ALWAYS a small dense/
    # structured calculation by design (never dependent on composite G, per this task's Goal 5
    # architecture table: "H_RR: family-specific structured/direct restriction method, never
    # dependent on composite G") -- this counter exists purely for visibility into how often that
    # family-owned path fires, not as a violation signal (there is no "should be zero" requirement
    # on it, unlike the composite-G/dense-reference counters above).
    direct_restriction_hessian_calls::Int = 0
    # No-moments/no-composite-G task (2026-07-28): the shared Hessian-weight prep
    # (operator_hessian_weights.jl::operator_prep_for_hessian!) replacing `_archC_prep_for_hessian!`'s
    # dense `H[:,2:1+outer_constr_index]` gemv for flexible-CM/common-Fréchet/CM+ZC/ZC-only. A "hit"
    # reuses the exact-same-point `r` the FG callback already published into `st.obj.arg0`; a "miss"
    # recomputes `r` fresh via the family's own operator forward kernel (`dual_index!`) -- both paths
    # are dense-G-free, `hessian_weight_dense_recomputes` is the ONLY one that is not (the retained,
    # explicit-opt-in `_archC_prep_for_hessian!` fallback under `moment_representation=:dense_reference`).
    hessian_weight_cache_hits::Int = 0
    hessian_weight_cache_misses::Int = 0
    hessian_weight_operator_recomputes::Int = 0
    hessian_weight_dense_recomputes::Int = 0
end

const NO_DENSE_G_COUNTERS = Ref(NoDenseGCounters())

"Reset every counter to zero -- call at the start of a gate/benchmark run that wants a clean count."
reset_no_dense_g_counters!() = (NO_DENSE_G_COUNTERS[] = NoDenseGCounters(); nothing)

"""
    MOMENT_REPRESENTATION

Default-flips task (2026-07-27), Task C: explicit dispatch for whether a family's generic
inner-solve SETUP (the once-per-inner-solve `obj.moments!(@view(H[:,1]), select_G_from_H(obj,H),
θ, obj.U, obj)` call in `inner_loop_internal_archgeneric`/`inner_loop_internal_cm*lookup_production`
-- NOT the per-Newton-iterate FG callback, which was already correctly operator-vs-dense-forked
before this task) is allowed to skip filling the family-specific "restriction" G columns (CM bins,
Fréchet level anchor, ZC mean/pair) that the operator FG callback never reads.

`:operator` (default): skip the restriction-column dense fill wherever it is SAFE to do so, i.e.
wherever nothing downstream in the SAME inner solve (Hessian callback included) still reads those
columns. `:dense_reference`: always fill everything, exactly as the pre-existing code did --
retained as an explicit diagnostic backend, not deleted.

**"Safe to skip" is decided per family, not globally** -- flipping this Ref to `:operator` does NOT
blindly skip every family's restriction fill:
  - Flexible CM (`cm_production_bundle.jl::archC_base_state`): SAFE, and now wired to this selector.
    `:cm_lookup` FG + `:winner_bin` H_EC cross-Hessian means NEITHER the FG callback NOR the Hessian
    reads the dense CM columns anymore (confirmed live, `docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_
    2026-07-27.md` C.2: `dense_cross_hessian_calls=0`, `winner_cross_hessian_calls>0`).
  - Common-Fréchet (`cm_frechet_cplus.jl::archC_frechet_base_state`): **NOT SAFE, deliberately NOT
    wired to this selector.** `archC_frechet_hess_cb_builder`'s Hessian callback reads the dense
    CM/level columns regardless of FG backend -- skipping their fill under `:cm_frechet_lookup` was
    tried once already (Phase 5.2, `skip_cm_fill_ref`) and produced a real, reproduced (4/4)
    `nStatus=-400` infeasible termination away from the calibration point, root-caused and fixed by
    REMOVING the skip (see the long comment at the top of `archC_frechet_base_state` and
    `docs/COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md`). Re-wiring this family to `:operator`
    without a corresponding operator/winner-bin Hessian cross-block first would silently reintroduce
    that exact bug -- do not do this without re-validating the Hessian side first.
  - CM+ZC / origin-ZC: out of this task's scope (a separate agent owns their Hessian internals;
    `production_backend_manifest.jl` already records `cross_hessian_backend=:dense_exact`/
    `restriction_hessian_backend=:dense_exact` for origin-ZC's H_ER/H_RR, and CM+ZC's H_EC is
    structurally excluded from `:winner_bin` whenever `ncore_core<NCORE`) -- not wired here.
  - Unrestricted: not applicable -- its compressed-only FG path never goes through
    `inner_loop_internal_archgeneric`/`select_G_from_H` at all (predates this counter set), so there
    is no composite-G setup call for this selector to gate in the first place.
"""
const MOMENT_REPRESENTATION = Ref{Symbol}(:operator)

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

"Call from a family's own structured/direct restriction-self-block (H_RR/H_CC) computation -- see the field's own docstring above."
record_direct_restriction_hessian_call!() = (NO_DENSE_G_COUNTERS[].direct_restriction_hessian_calls += 1; nothing)

"""
    record_hessian_weight_cache_hit!() / _cache_miss!() / _operator_recompute!() / _dense_recompute!()

No-moments/no-composite-G task (2026-07-28): call exactly once per Hessian callback invocation from
`operator_hessian_weights.jl::operator_prep_for_hessian!` (hit or miss+operator_recompute, mutually
exclusive) or from the retained `_archC_prep_for_hessian!` dense fallback (dense_recompute, only
reachable under explicit `moment_representation=:dense_reference`). Production requires
`hessian_weight_dense_recomputes == 0`.
"""
record_hessian_weight_cache_hit!() = (NO_DENSE_G_COUNTERS[].hessian_weight_cache_hits += 1; nothing)
record_hessian_weight_cache_miss!() = (NO_DENSE_G_COUNTERS[].hessian_weight_cache_misses += 1; nothing)
record_hessian_weight_operator_recompute!() = (NO_DENSE_G_COUNTERS[].hessian_weight_operator_recomputes += 1; nothing)
function record_hessian_weight_dense_recompute!()
    NO_DENSE_G_COUNTERS[].hessian_weight_dense_recomputes += 1
    FAIL_FAST_ON_DENSE_G[] &&
        error("record_hessian_weight_dense_recompute!: a production Hessian callback used the dense _archC_prep_for_hessian! fallback while FAIL_FAST_ON_DENSE_G[]=true")
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
            winner_cross_hessian_calls = c.winner_cross_hessian_calls,
            direct_restriction_hessian_calls = c.direct_restriction_hessian_calls,
            hessian_weight_cache_hits = c.hessian_weight_cache_hits,
            hessian_weight_cache_misses = c.hessian_weight_cache_misses,
            hessian_weight_operator_recomputes = c.hessian_weight_operator_recomputes,
            hessian_weight_dense_recomputes = c.hessian_weight_dense_recomputes)
end

"""
    no_dense_g_report_task_names() -> NamedTuple

Final-architecture-closure task (2026-07-27), §11: the SAME snapshot as `no_dense_g_report()`,
keyed by the exact 13 counter names that task's own §11 lists verbatim
(`composite_G_materializations`, `dense_economic_block_materializations`, etc.). Added as a
DISTINCT accessor rather than renaming the underlying struct fields, to avoid a repository-wide
rename across the ~30 existing call sites (gate scripts, docs, other counters' own cross-references)
that already read `no_dense_g_report()`'s current field names -- both accessors read the SAME
underlying `NO_DENSE_G_COUNTERS[]` state, there is no drift risk between them (verified by
construction: this function is a pure relabeling of `no_dense_g_report()`'s own return value, not a
second independent counter store).
"""
function no_dense_g_report_task_names()
    r = no_dense_g_report()
    return (composite_G_materializations = r.full_G_materializations,
            dense_economic_block_materializations = r.dense_economic_G_materializations,
            dense_CM_block_materializations = r.dense_CM_G_materializations,
            dense_Frechet_block_materializations = r.dense_Frechet_G_materializations,
            dense_ZC_block_materializations = r.dense_ZC_G_materializations,
            generic_dense_FG_calls = r.generic_dense_FG_calls,
            dense_reference_verification_calls = r.dense_reference_verification_calls,
            dense_cross_hessian_calls = r.dense_cross_hessian_calls,
            operator_economic_FG_calls = r.operator_economic_FG_calls,
            operator_restriction_FG_calls = r.operator_restriction_FG_calls,
            operator_verification_calls = r.operator_verification_calls,
            winner_cross_hessian_calls = r.winner_cross_hessian_calls,
            direct_restriction_hessian_calls = r.direct_restriction_hessian_calls)
end
