# ============================================================================
# Validation for the successful-dual/KKT-scored bank (dual_bank.jl), ported
# from diag/fullA-d20-warmstart-replay's Policy P3.
#
# Checks:
#   1. record_success! bounds history to maxsize (recency-window eviction).
#   2. select_warm_start never proposes a candidate that wasn't either the
#      current production slot, a genuinely-recorded success, or neutral.
#   3. An organic failure (never recorded) does not appear in the bank, and
#      the bank survives a NaN-poisoned obj.x (falls back to neutral/bank
#      candidates only, matching production_actual_start's guard).
#   4. cheap_score is lower (or equal) at a converged solution's own dual
#      than at a neutral start, at the SAME point it was solved at -- a
#      converged dual should look better than neutral by its own KKT proxy,
#      sanity-checking the scoring direction (lower = better) is wired the
#      way select_warm_start assumes (argmin).
#   5. Selection is deterministic: repeated calls at the same bank/point
#      state return the identical candidate.
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_dual_bank.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
include(joinpath(@__DIR__, "dual_bank.jl"))

using Random, LinearAlgebra, Statistics

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

println("== 1. record_success! bounds history to maxsize (recency window) ==")
bank = DualBank(3)
for i in 1:5
    record_success!(bank, i, Float64[i, i], Float64[i, i, i])
end
check("history length == maxsize after exceeding it", length(bank.history) == 3)
check("history contains only the LAST maxsize entries (eval_id 3,4,5)", [h.eval_id for h in bank.history] == [3, 4, 5])

println("\n== 2/3/4/5: select_warm_start on a real D=4 context ==")
ctx = d4_exact_setup(find_smallest = true)
D = ctx.D
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# NOTE: `vcat(gp0, pivot_reduce(zeros(D,D),pe))` is NOT the calibration point -- it's
# gravity_elimination.jl's own pivot-reparametrization zero-reference (A_od==1 everywhere), a
# well-documented recurring trap in this repo (see memory
# feedback-gravity-elimination-zero-is-not-calibration.md). The REAL calibration
# (ctx.θ0_up's own A_od block, bypassing the pivot reparam) DOES solve cold here too
# (inner_status=0, Delta_dual=0.0010029941762691, matching δ_star_initial) -- re-verified
# directly for this D=4 ctx, consistent with that memory's D=20 finding. The pivot's zero-
# reference point itself, however, genuinely does NOT solve cold (verified: 4 repeated cold
# calls in one process all return inner_status=-300 instantly, ruling out JIT-timing;
# reproduced identically on the clean, unmodified 98983bd base -- not a regression). Since this
# test only needs a point that reliably converges cold (not specifically calibration), it uses
# test_infeasibility_screen.jl's OWN "lower_stalled_maxit15" feasible_ws registry point instead
# (confirmed here to solve cold: inner_status=0).
w_target = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]
zfree_target = w_target[2:end]
xf0 = x_free_from_w(w_target)
pc_shared = precompute_pairwise_M(ctx)   # this ctx type has no .pairwise field -- pass explicitly
r0, _ = evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed, cache = nothing, use_cache = false, warm = false, pairwise = pc_shared)
check("D=4 feasible_ws point solves feasibly (sanity precondition)", r0.inner_status in (0, -100, -101, -103))

θ_full0 = CS.reconstruct_full(xf0, ctx.m)
cf0 = build_compressed_factual(θ_full0, ctx; check_ties = true)
x_solved0 = vcat(r0.zeta, r0.lambda)

score_solved = cheap_score(ctx.obj, cf0, x_solved0)
score_neutral = cheap_score(ctx.obj, cf0, zeros(ctx.obj.outer_constr_index))
println("  cheap_score at converged solution's own dual: ", score_solved)
println("  cheap_score at neutral (zeros): ", score_neutral)
check("converged dual scores <= neutral at its OWN point (lower=better is the right direction)", score_solved <= score_neutral + 1e-9)

empty_bank = DualBank(8)
ctx.obj.x .= NaN   # simulate the post-(-300) NaN-poisoned single slot
x_sel, label = select_warm_start(empty_bank, ctx.obj, cf0, zfree_target)
check("empty bank + NaN-poisoned obj.x falls back to neutral", label == :neutral && all(iszero, x_sel))

bank2 = DualBank(8)
record_success!(bank2, 1, zfree_target, x_solved0)
x_sel2, label2 = select_warm_start(bank2, ctx.obj, cf0, zfree_target)
check("bank with one recorded success proposes a candidate from {last_accepted,nearest,neutral} (obj.x still NaN)",
      label2 in (:last_accepted, :nearest, :neutral))
check("selected candidate is either the recorded success or neutral (never fabricated)",
      x_sel2 == x_solved0 || all(iszero, x_sel2))

ctx.obj.x .= x_solved0   # simulate a valid production last-successful slot
x_sel3, label3 = select_warm_start(bank2, ctx.obj, cf0, zfree_target)
check("with a valid obj.x, :actual is a legal candidate label", label3 in (:actual, :last_accepted, :nearest, :neutral))

x_sel4, label4 = select_warm_start(bank2, ctx.obj, cf0, zfree_target)
check("selection is deterministic across repeated calls at identical state", label4 == label3 && x_sel4 == x_sel3)

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
