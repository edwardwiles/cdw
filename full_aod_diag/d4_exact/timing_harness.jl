# ============================================================================
# Shared timing-regression harness (task brief Phase C / plan step 6-9).
#
# Reuses the EXACT point-construction methodology already validated by
# Continuation 10's own canonical benchmark (c10_canonical_benchmark.jl,
# commit cf74d89): genuine calibration (ctx.θ0_up's own A_od block --
# NOT the gravity-elimination pivot's z=0 reference point, see memory
# feedback-gravity-elimination-zero-is-not-calibration.md) perturbed by
# gp0*1.01, real D=20/W=80,000, delta=1, draw_seed=20260719.
#
# Measures the 7 distinct timing objects from the task brief:
#   1. inner KNITRO solve only (approximated: cold value eval minus screen time)
#   2. complete screened value callback (cold)
#   3. exact same-point warm re-solve
#   4. nearby changed-point warm solve (small zfree perturbation)
#   5. distant changed-point warm solve (larger zfree perturbation)
#   6. cold solve (same as 2, reported separately for clarity)
#   7. exact-point cache hit (only meaningful when exact_cache is enabled)
#
# This is the MODERN-architecture variant (evaluate_fullA_screened_ranged +
# composite_gradient_at_fast_buffered + optional exact_cache/dual_bank),
# used for worktree B (98983bd, caches off by construction -- no SafeExactCache
# exists there) and worktrees C/D (this integration branch, caches
# off/on via USE_EXACT_CACHE below). A separate timing_harness_legacy.jl
# (evaluate_fullA_screened + composite_gradient_at_fast, no ranged screens)
# is used for worktree A (cf74d89, pre-range-screen architecture).
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/timing_harness.jl [on|off]
# (optional first ARG: "on" enables exact_cache for this run, "off"/omitted
# leaves it disabled -- lets one script serve both worktree C and D roles.)
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, LinearAlgebra, Dates, Statistics

USE_EXACT_CACHE = length(ARGS) >= 1 && ARGS[1] == "on"

function rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Int, split(line)[2]) / 1024
            end
        end
    catch
    end
    return NaN
end

lp(xs...) = (println(xs...); flush(stdout))

lp("=== TIMING HARNESS (modern architecture) === ", Dates.now(), "  USE_EXACT_CACHE=", USE_EXACT_CACHE)
lp("RSS at start: ", round(rss_mb(), digits=1), " MB")

Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = 80000, find_smallest = true, δ = 1.0)
t_ctx = time() - t0
lp("ctx build wall=", round(t_ctx, digits=1), "s  RSS=", round(rss_mb(), digits=1), " MB  screen_setup_wall=", ctx.screen_setup_wall)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# genuine calibration (NOT the pivot zero-reference -- see file header)
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0 * 1.01, zfree0)
xf0 = x_free_from_w2(w0)

exact_cache = USE_EXACT_CACHE ? SafeExactCache() : nothing

# ---- 6/2. cold value eval (complete screened callback, cold) ----
t0 = time()
r_cold, meta_cold = evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed,
    cache = exact_cache, use_cache = exact_cache !== nothing, warm = false, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_cold_val = time() - t0
lp("[6/2. cold value eval, complete callback] wall=", round(t_cold_val, digits=3), "s  Delta_dual=", r_cold.Delta_dual,
   "  inner_status=", r_cold.inner_status, "  screen_status=", meta_cold.screen_status)

# ---- 3. exact same-point warm re-solve ----
t0 = time()
r_warm, meta_warm = evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = true, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_warm_same = time() - t0
lp("[3. exact same-point warm re-solve] wall=", round(t_warm_same, digits=3), "s  Delta_dual=", r_warm.Delta_dual, "  inner_status=", r_warm.inner_status)

# ---- 7. exact-point cache hit (only if enabled) ----
if exact_cache !== nothing
    t0 = time()
    r_hit, meta_hit = evaluate_fullA_screened_ranged(xf0, ctx, rsc; moment_representation = :compressed,
        cache = exact_cache, use_cache = true, warm = false, tag = "",
        pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    t_hit = time() - t0
    lp("[7. exact-point cache hit] wall=", round(t_hit, digits=5), "s  cache_hit=", r_hit.cache_hit, "  Delta_dual=", r_hit.Delta_dual)
end

# ---- 4. nearby changed-point warm solve (small zfree perturbation, ~0.05 step) ----
Random.seed!(777)
dir_near = randn(length(zfree0)); dir_near ./= sqrt(sum(abs2, dir_near))
zfree_near = zfree0 .+ 0.05 .* dir_near
xf_near = x_free_from_w2(vcat(gp0 * 1.01, zfree_near))
t0 = time()
r_near, meta_near = evaluate_fullA_screened_ranged(xf_near, ctx, rsc; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = true, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_near = time() - t0
lp("[4. nearby changed-point (|Δzfree|~0.05) warm solve] wall=", round(t_near, digits=3), "s  Delta_dual=", r_near.Delta_dual, "  inner_status=", r_near.inner_status)

# ---- 5. distant changed-point warm solve (larger perturbation, ~1.0 step) ----
dir_far = randn(length(zfree0)); dir_far ./= sqrt(sum(abs2, dir_far))
zfree_far = zfree0 .+ 1.0 .* dir_far
xf_far = x_free_from_w2(vcat(gp0 * 1.01, zfree_far))
t0 = time()
r_far, meta_far = evaluate_fullA_screened_ranged(xf_far, ctx, rsc; moment_representation = :compressed,
    cache = nothing, use_cache = false, warm = true, tag = "",
    pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
t_far = time() - t0
lp("[5. distant changed-point (|Δzfree|~1.0) warm solve] wall=", round(t_far, digits=3), "s  Delta_dual=", r_far.Delta_dual, "  inner_status=", r_far.inner_status, "  screen_status=", meta_far.screen_status)

# ---- full outer composite gradient evaluation (buffered, modern default) ----
base = BaseDualState(collect(xf0), r_warm.θ_full, r_warm.zeta, r_warm.lambda, copy(ctx.obj.arg1), r_warm.inner_status)
t0 = time()
gfull, gmeta = composite_gradient_at_fast_buffered(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_grad = time() - t0
lp("[full outer gradient, buffered] wall=", round(t_grad, digits=3), "s  norm(gfull)=", norm(gfull), "  tie_fallback=", gmeta.tie_fallback)

lp("\nSUMMARY: ctx_build=", round(t_ctx,digits=2), " cold=", round(t_cold_val,digits=3), " warm_same=", round(t_warm_same,digits=3),
   " nearby=", round(t_near,digits=3), " distant=", round(t_far,digits=3), " grad=", round(t_grad,digits=3),
   exact_cache !== nothing ? " cache_hit=$(round(t_hit,digits=5))" : "")
lp("\nDONE_TIMING_HARNESS")
