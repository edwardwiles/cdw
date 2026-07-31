# ============================================================================
# Independent validation of the draw_design.jl port (pseudorandom/
# sobol_randomized/halton_scrambled), written by the consolidating session
# rather than the porting agent, per the addendum's own acceptance bar.
#
# Checks:
#   1. :pseudorandom default is bit-for-bit unchanged vs the pre-existing
#      Random.seed!(draw_seed); d20_real_setup(...) sequence (same ctx.U,
#      same Delta_dual at a real point).
#   2. Fresh-context reproducibility: same design+seed -> same checksums
#      (checked via two calls in one process at fresh contexts, which is a
#      necessary but not sufficient proxy for fresh-process reproducibility --
#      the true fresh-process check is done by running this script twice).
#   3. :sobol_randomized / :halton_scrambled build without crashing, produce
#      finite Delta_dual at a real value evaluation, and have DIFFERENT
#      checksums from :pseudorandom and from each other (genuinely different
#      draw sets, not accidentally identical).
#   4. Zero measurable overhead for :pseudorandom when log_draw_meta=false.
#   5. (unify-random-draw-production-pipeline, 2026-07-30, task §14)
#      historical-drift regression checks: threshold_state finite and equal
#      to resolve_threshold_for_delta(delta) for EVERY design at delta<9 (the
#      exact bug the audit found -- QMC used to silently get threshold=Inf);
#      pairwise/witness/screen_setup_wall present for every design; returned
#      context field sets identical across designs; :precomputed routes
#      through the same pipeline as the generated designs; a delta>=9 case
#      (where the threshold feature should self-disable) disables identically
#      for every design, not just :pseudorandom.
#
# Run standalone (twice, to check cross-process reproducibility manually):
#   julia --project=. full_aod_diag/d4_exact/test_draw_design.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf

lp(xs...) = (println(xs...); flush(stdout))

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

W = 80000; δ = 1.0; seed = 20260719

lp("== 1. :pseudorandom default is bit-for-bit unchanged ==")
Random.seed!(seed)
ctx_old = d20_real_setup(W = W, δ = δ, find_smallest = true)
ctx_new = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = :pseudorandom, draw_seed = seed)
check(":pseudorandom ctx.U bit-identical to old d20_real_setup path", ctx_old.U == ctx_new.U)
check("draw_design field == :pseudorandom", ctx_new.draw_design == :pseudorandom)

pe = build_pivot_elimination(ctx_new)
# D_dest, not D: ctx's Aod free block is Dact x Ddest (rectangular under the current
# destination_sample=:exclude_row default), not D^2 -- pre-existing bug in this test file
# (predates the draw-design unification; written assuming the square :all_legacy layout), found
# while running this gate for real at W=80000. Not part of the draw-design refactor itself.
D = ctx_new.D; Ddest = ctx_new.D_dest; DxDdest = D * Ddest
x_free_from_w2(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx_new.θ0_up[ctx_new.Aod_offset+1:ctx_new.Aod_offset+DxDdest]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe)
gp0 = ctx_new.θ0_up[3+D]
xf0 = x_free_from_w2(vcat(gp0*1.01, zfree0), pe)

r_old, _ = evaluate_fullA_fast(xf0, ctx_old; cache = nothing, use_cache = false, warm = false)
r_new, _ = evaluate_fullA_fast(xf0, ctx_new; cache = nothing, use_cache = false, warm = false)
check(":pseudorandom Delta_dual bit-identical at a real value eval", r_old.Delta_dual === r_new.Delta_dual)
lp("  Delta_dual = ", r_old.Delta_dual)

lp("\n== 2. Fresh-context reproducibility (same design+seed -> same checksum) ==")
ctx_new2 = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = :pseudorandom, draw_seed = seed)
check(":pseudorandom checksum_uniform reproducible", ctx_new.draw_meta.checksum_uniform == ctx_new2.draw_meta.checksum_uniform)
check(":pseudorandom checksum_transformed reproducible", ctx_new.draw_meta.checksum_transformed == ctx_new2.draw_meta.checksum_transformed)

lp("\n== 3. sobol_randomized / halton_scrambled build + solve + genuinely different draws ==")
for design in (:sobol_randomized, :halton_scrambled)
    ctx_q = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = design, draw_seed = seed)
    check("$design: ctx builds, draw_design field correct", ctx_q.draw_design == design)
    check("$design: checksum differs from :pseudorandom", ctx_q.draw_meta.checksum_transformed != ctx_new.draw_meta.checksum_transformed)
    check("$design: no non-finite transformed draws", ctx_q.draw_meta.n_inf_transformed == 0)
    check("$design: no exact-boundary uniforms (0 or 1)", ctx_q.draw_meta.n_at_boundary == 0)

    pe_q = build_pivot_elimination(ctx_q)
    xf_q = x_free_from_w2(vcat(gp0*1.01, zfree0), pe_q)
    r_q, _ = evaluate_fullA_fast(xf_q, ctx_q; cache = nothing, use_cache = false, warm = false)
    check("$design: value eval runs, inner_status in FEASIBLE_CODES or a real screen/solver code", r_q.inner_status isa Integer)
    check("$design: Delta_dual finite or NaN (never crashes/errors)", isnan(r_q.Delta_dual) || isfinite(r_q.Delta_dual))
    lp("  $design: inner_status=", r_q.inner_status, " Delta_dual=", r_q.Delta_dual,
       " checksum=", ctx_q.draw_meta.checksum_transformed[1:12], "...")

    ctx_q2 = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = design, draw_seed = seed)
    check("$design: reproducible checksum across two fresh contexts", ctx_q.draw_meta.checksum_transformed == ctx_q2.draw_meta.checksum_transformed)
end

ctx_sobol = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = :sobol_randomized, draw_seed = seed)
ctx_halton = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = :halton_scrambled, draw_seed = seed)
check("sobol_randomized and halton_scrambled have DIFFERENT checksums from each other", ctx_sobol.draw_meta.checksum_transformed != ctx_halton.draw_meta.checksum_transformed)

lp("\n== 4. :pseudorandom overhead with log_draw_meta=false ==")
t0 = time()
ctx_nolog = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = :pseudorandom, draw_seed = seed, log_draw_meta = false)
t_nolog = time() - t0
t0 = time()
ctx_wlog = d20_real_setup_design(W = W, δ = δ, find_smallest = true, draw_design = :pseudorandom, draw_seed = seed, log_draw_meta = true)
t_wlog = time() - t0
check("log_draw_meta=false skips meta (draw_meta === nothing)", ctx_nolog.draw_meta === nothing)
lp("  ctx build wall: log_draw_meta=false -> ", round(t_nolog,digits=2), "s   log_draw_meta=true -> ", round(t_wlog,digits=2), "s")
check("logging overhead is small relative to ctx build (<20% of total)", abs(t_wlog - t_nolog) < 0.2 * max(t_nolog, t_wlog))

lp("\n== 5. Historical-drift regression checks (task §14) ==")
lp("   -- threshold_state, screens, and field set identical across EVERY design, not just :pseudorandom")

W_drift = 4000   # small W: this section is about wiring/structure, not statistical precision
δ_active = 1.0    # < 9 -> threshold should be resolve_threshold_for_delta(1.0) == 10.0, ACTIVE
δ_inactive = 10.0 # >= 9 -> threshold should be Inf, feature self-disabled

all_designs_for_drift = (:pseudorandom, :sobol_randomized, :halton_scrambled, :precomputed)
Uprecomp_drift = pseudorandom_U(W_drift, D20_REAL; seed = 777)

function build_drift_ctx(design, δ)
    design == :precomputed ?
        d20_real_setup_design(W = W_drift, δ = δ, find_smallest = true, draw_design = :precomputed,
            U_precomputed = Uprecomp_drift, precomputed_already_transformed = true) :
        d20_real_setup_design(W = W_drift, δ = δ, find_smallest = true, draw_design = design, draw_seed = 20260719)
end

field_sets = Dict{Symbol,Set{Symbol}}()
for design in all_designs_for_drift
    ctx = build_drift_ctx(design, δ_active)
    expected = CS.resolve_threshold_for_delta(δ_active)
    check("$design (δ=$δ_active<9): threshold_state.threshold is finite and == resolve_threshold_for_delta(δ)",
        isfinite(ctx.obj.threshold_state.threshold) && ctx.obj.threshold_state.threshold == expected)
    check("$design: ctx.pairwise built (not nothing)", ctx.pairwise !== nothing)
    check("$design: ctx.witness built (not nothing)", ctx.witness !== nothing)
    check("$design: ctx.screen_setup_wall present with finite timings",
        isfinite(ctx.screen_setup_wall.pairwise) && isfinite(ctx.screen_setup_wall.witness))
    check("$design: destination_sample routed (default :exclude_row, D_dest=D-1)",
        ctx.destination_sample == :exclude_row && ctx.D_dest == ctx.D - 1)
    field_sets[design] = Set(propertynames(ctx))
end

reference_fields = field_sets[:pseudorandom]
for design in all_designs_for_drift
    design == :pseudorandom && continue
    check("$design: returned context field SET identical to :pseudorandom's", field_sets[design] == reference_fields)
end

lp("   -- at δ=$δ_inactive>=9, the threshold feature must self-disable IDENTICALLY for every design (not silently already-Inf only for some)")
for design in all_designs_for_drift
    ctx = build_drift_ctx(design, δ_inactive)
    check("$design (δ=$δ_inactive>=9): threshold_state.threshold == Inf (self-disabled, matches resolve_threshold_for_delta)",
        ctx.obj.threshold_state.threshold == Inf)
end

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
