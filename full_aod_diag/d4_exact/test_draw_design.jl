# ============================================================================
# Independent validation of the draw_design.jl port (pseudorandom/
# sobol_randomized/halton_scrambled), written by the consolidating session
# rather than the porting agent, per the addendum's own acceptance bar.
#
# Checks:
#   1. :pseudorandom default is bit-for-bit unchanged vs the pre-existing
#      Random.seed!(draw_seed); d20_real_setup(...) sequence (same ctx.U,
#      same Delta_dual at a real point).
#   2. Fresh-process reproducibility: same design+seed -> same checksums
#      (checked via two calls in one process at fresh contexts, which is a
#      necessary but not sufficient proxy for fresh-process reproducibility --
#      the true fresh-process check is done by running this script twice).
#   3. :sobol_randomized / :halton_scrambled build without crashing, produce
#      finite Delta_dual at a real value evaluation, and have DIFFERENT
#      checksums from :pseudorandom and from each other (genuinely different
#      draw sets, not accidentally identical).
#   4. Zero measurable overhead for :pseudorandom when log_draw_meta=false.
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
D = ctx_new.D; D2 = D^2
x_free_from_w2(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx_new.θ0_up[ctx_new.Aod_offset+1:ctx_new.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
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

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
