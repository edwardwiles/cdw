# ============================================================================
# Pure, KNITRO-free incumbent-tracking helpers factored out of
# run_profile_checkpointed / run_polish_checkpointed (c10_d20_production_driver.jl).
#
# Bug this fixes (see docs/fullA_driver_delta5_diagnostics_handoff.md §3): both driver
# functions cold-verify the caller-supplied start point as feasible (r_seed / r0) before
# calling KN_solve, but the pre-fix code discarded that result and initialized the
# best-feasible incumbent to `nothing` on every fresh (non-resumed) call. KNITRO's own
# first callback evaluation of that same point happens via a WARM solve against a
# freshly-initialized (effectively empty) dual slot, which can genuinely fail or simply
# never complete if the wall-time budget is exhausted first. A caller that chains stages
# by feeding the previous stage's incumbent forward (run_staged_delta5_continuation) then
# silently loses ground each stage, even though the true feasible set only grows with a
# larger delta budget.
#
# `seed_incumbent` and the two `is_better_*` predicates below are the exact decision
# logic used at incumbent-initialization time and at every accepted callback evaluation,
# extracted so they can be unit-tested (test_incumbent_seeding.jl) without building a
# real D=20 context or calling KNITRO at all.
# ============================================================================

"""
    seed_incumbent(resumed_best, cand_feasible::Bool, cand)

Decide the initial incumbent before `KN_solve` is ever called.

- If resuming from a checkpoint (`resumed_best !== nothing`), the checkpoint's own
  incumbent always wins -- a resumed run must never regress relative to what a prior
  segment already found.
- Otherwise, if the caller-supplied start point was itself cold-verified feasible
  (`cand_feasible`), it becomes the initial incumbent (`cand`). This is the fix: the old
  code returned `nothing` unconditionally here.
- If the start point was not feasible (or resuming is not in play and no candidate was
  computed), there is genuinely no better information available yet, so `nothing` is
  correct.
"""
function seed_incumbent(resumed_best, cand_feasible::Bool, cand)
    resumed_best !== nothing && return resumed_best
    return cand_feasible ? cand : nothing
end

"""
    is_better_polish(candidate_gp, best_gp_or_nothing, find_smallest::Bool)

Upper/lower-direction comparison used by `run_polish_checkpointed`'s `cb_F!`:
`find_smallest=true` (minimize gamma'_focal, which MAXIMIZES kappa -- the UPPER-bound search;
see direction_bounds.jl's three-way evidence and c10_d20_production_driver.jl's own
`find_smallest=true ⇔ :upper` convention) prefers a SMALLER `gp`; `find_smallest=false`
(the LOWER-bound search) prefers a LARGER `gp`. `best_gp_or_nothing === nothing` (no
incumbent yet) always accepts the candidate.

Remediation task Part E (finding F8): this docstring previously labeled `find_smallest=true` as
"lower-bound search" -- inverted. The LOGIC below was always correct (smaller gp preferred when
find_smallest=true); only the docstring's direction label was wrong, reproducing the exact
historical label-swap confusion the direction-box addendum (git 5fbfada) fixed elsewhere.
"""
function is_better_polish(candidate_gp::Real, best_gp_or_nothing, find_smallest::Bool)
    best_gp_or_nothing === nothing && return true
    return find_smallest ? candidate_gp < best_gp_or_nothing : candidate_gp > best_gp_or_nothing
end

"""
    is_better_profile(candidate_Delta, best_Delta_or_nothing)

Comparison used by `run_profile_checkpointed`'s `cb_F!`: this stage always minimizes the
divergence `Delta_dual` needed (there is no separate upper/lower direction -- both
find_smallest branches of the outer problem call this the same way). `nothing` (no
incumbent yet) always accepts the candidate.
"""
function is_better_profile(candidate_Delta::Real, best_Delta_or_nothing)
    best_Delta_or_nothing === nothing && return true
    return candidate_Delta < best_Delta_or_nothing
end
