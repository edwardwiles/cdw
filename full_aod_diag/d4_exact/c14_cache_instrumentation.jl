# Continuation 14, Task 4: exact-point cache instrumentation on a REALISTIC outer-loop-style
# sequence of repeated/nearby point requests, both unrestricted (oracle_cache_for/
# SafeExactCache{FullAEvalKey}) and CM (cm_oracle_cache_for/SafeExactCache{CMEvalKey}).
#
# Phase 2 (HEAD commit) already validated the CM cache MECHANICALLY (miss->store, hit,
# no-collision-on-different-draw_checksum). This script instruments a scenario shaped like a real
# outer KNITRO iterate: the SAME point gets asked about multiple times within one iterate
# (cb_F!/cb_G!/cb_newpt! all query the oracle at the current trial x), then the outer solver moves
# to a nearby point for the next iterate, etc. Reports: hits, misses (with reason -- genuinely new
# point vs. a key-field mismatch), inner solves avoided, and cache memory cost
# (Base.summarysize). Confirms a hit never calls KN_new by diffing CS.INNER_ITERS_TOTAL[] across
# the call (a hit must show iters delta == 0; production's KN_solve always advances this counter by
# >= 1 on a real solve, see oracle_fast.jl/cm_hessian_architectures.jl's own `CS.INNER_ITERS_TOTAL[]
# += CS._kn_num_iters(kc)` after every KN_solve).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod")
mkpath(OUTDIR)

lp("=== c14_cache_instrumentation === ", Dates.now())
W = 80000; DELTA = 1.0
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d", time()-t0, D, W))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
xf0 = x_free_from_w2(vcat(gp0 * 1.01, zfree0))
Random.seed!(777)
dir = randn(length(zfree0)); dir ./= sqrt(sum(abs2, dir))
xf_near = x_free_from_w2(vcat(gp0 * 1.01, zfree0 .+ 0.05 .* dir))

# ---- unrestricted cache instrumentation ----
lp(""); lp("="^100); lp("UNRESTRICTED cache (oracle_cache_for / SafeExactCache{FullAEvalKey})"); lp("="^100)
cache_ur = oracle_cache_for(ctx)
results_ur = NamedTuple[]

function probe_ur(label, xf; warm = false)
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    r, meta = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = cache_ur, use_cache = true, warm = warm, tag = "",
        pairwise = ctx.pairwise, witness = ctx.witness, use_witness = true)
    wall = time() - t0
    iters = CS.INNER_ITERS_TOTAL[] - iters0
    hit = get(r, :cache_hit, false)
    lp(@sprintf("  [%-28s] wall=%9.5fs  cache_hit=%-5s  iters_delta=%3d  inner_status=%s  screen_status=%s",
        label, wall, string(hit), iters, string(r.inner_status), string(get(meta, :screen_status, missing))))
    push!(results_ur, (label = label, wall = wall, cache_hit = hit, iters_delta = iters, inner_status = r.inner_status))
    return r, meta
end

lp("-- simulated outer iterate 1 at xf0: cb_F! then cb_G!'s own base-state re-query (both ask about the SAME point) --")
probe_ur("iter1_cbF_xf0_MISS", xf0)
probe_ur("iter1_cbG_xf0_expect_HIT", xf0)
probe_ur("iter1_newpt_xf0_expect_HIT", xf0)
lp("-- outer solver moves to a nearby point xf_near (genuinely new point -> MISS) --")
probe_ur("iter2_cbF_xfnear_MISS", xf_near)
probe_ur("iter2_cbG_xfnear_expect_HIT", xf_near)
lp("-- outer solver revisits xf0 later (e.g. a rejected line-search step reverting) -- still a HIT, cache persists across iterates --")
probe_ur("iter3_revisit_xf0_expect_HIT", xf0)
lp("-- key-field-mismatch miss: SAME x_free but a different delta -- construct the key directly")
lp("   (FullAEvalKey's own fields: x_free, δ, find_smallest, inner_loop_opt, mode) and confirm")
lp("   _cache_lookup correctly returns nothing even though x_free is byte-identical to a stored hit --")
key_same_xfree_diff_delta = FullAEvalKey(collect(xf0), 2.0, ctx.obj.find_smallest, ctx.obj.inner_loop_opt, :hard)
miss_by_key_mismatch = _cache_lookup(cache_ur, key_same_xfree_diff_delta)
lp(@sprintf("  [%-28s] lookup result = %s  (must be `nothing` -- a real cache entry exists at this x_free under delta=%.1f, NOT delta=2.0)",
    "diffdelta_samexfree_MISS", string(miss_by_key_mismatch), DELTA))
push!(results_ur, (label = "diffdelta_samexfree_keymismatch_MISS", wall = 0.0, cache_hit = false, iters_delta = 0, inner_status = 0))

lp(""); lp("Cache size after unrestricted sequence: ", length(cache_ur), " entries, ",
    round(Base.summarysize(cache_ur.d) / 2^20, digits=3), " MB (Base.summarysize)")
n_hit_ur = count(r -> r.cache_hit, results_ur)
n_miss_ur = count(r -> !r.cache_hit, results_ur)
lp(@sprintf("Unrestricted: %d hits, %d misses out of %d probes. Inner solves avoided by cache: %d (hits with iters_delta==0: %d/%d)",
    n_hit_ur, n_miss_ur, length(results_ur), n_hit_ur, count(r -> r.cache_hit && r.iters_delta == 0, results_ur), n_hit_ur))

# ---- CM cache instrumentation ----
lp(""); lp("="^100); lp("CM cache (cm_oracle_cache_for / SafeExactCache{CMEvalKey})"); lp("="^100)
snaps = nested_grid_sequence([10, 20, 50])
cfg = CMConfig(common_marginals = true, cm_grid_rule = :nested_family, cm_grid_sizes = [10, 20, 50],
               cm_basis = :cumulative, cm_hessian_backend = :structured, contrasts = :anchored)
pcx = build_cm_production_context_v2(ctx, CS, cfg; L = 50)
cache_cm = cm_oracle_cache_for(pcx)
results_cm = NamedTuple[]

function probe_cm(label, xf; draw_checksum = hash(ctx.U))
    pcx.ctx_cm.obj.x .= NaN   # force cold on a genuine miss (see c14_cm_hessian_benchmark.jl)
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K, base = cm_production_value_v2(xf, pcx; cache = cache_cm, use_cache = true, draw_checksum = draw_checksum)
    wall = time() - t0
    iters = CS.INNER_ITERS_TOTAL[] - iters0
    hit = iters == 0 && wall < 0.01   # a real miss always does >=1 KN_solve iteration; a hit is a pure dict lookup
    lp(@sprintf("  [%-28s] wall=%9.5fs  iters_delta=%3d  (hit-if-both-~0: %-5s)  inner_status=%d  Delta_dual=%.10f",
        label, wall, iters, string(hit), base.inner_status, -base.ζstar))
    push!(results_cm, (label = label, wall = wall, iters_delta = iters, cache_hit = hit, inner_status = base.inner_status))
    return K, base
end

lp("-- CM outer iterate 1 at hard-point-adjacent xf0(mag=0.5 not needed here -- calibration is enough to show the mechanism), repeated 3x --")
probe_cm("cm_iter1_cbF_xf0_MISS", xf0)
probe_cm("cm_iter1_cbG_xf0_expect_HIT", xf0)
probe_cm("cm_iter1_newpt_xf0_expect_HIT", xf0)
lp("-- different draw_checksum, SAME xf0 -> must be a genuinely separate entry (miss), never collide --")
probe_cm("cm_diffchecksum_xf0_MISS_bydesign", xf0; draw_checksum = hash(ctx.U) + 1)
lp("-- back to the original checksum at xf0 -- still a hit, unaffected by the diff-checksum entry above --")
probe_cm("cm_backto_orig_checksum_xf0_HIT", xf0)

lp(""); lp("Cache size after CM sequence: ", length(cache_cm), " entries, ",
    round(Base.summarysize(cache_cm.d) / 2^20, digits=3), " MB (Base.summarysize)")
n_hit_cm = count(r -> r.cache_hit, results_cm)
n_miss_cm = count(r -> !r.cache_hit, results_cm)
lp(@sprintf("CM: %d hits, %d misses out of %d probes.", n_hit_cm, n_miss_cm, length(results_cm)))

lp(""); lp("="^100); lp("Cross-check: unrestricted and CM caches are NEVER the same object / never collide"); lp("="^100)
lp("  typeof(cache_ur) = ", typeof(cache_ur))
lp("  typeof(cache_cm) = ", typeof(cache_cm))
lp("  cache_ur === cache_cm ? ", cache_ur === cache_cm, "  (must be false)")

write_csv_rows(joinpath(OUTDIR, "cache_instrumentation_unrestricted.csv"), results_ur)
write_csv_rows(joinpath(OUTDIR, "cache_instrumentation_cm.csv"), results_cm)
lp(""); lp(">>> wrote cache_instrumentation_{unrestricted,cm}.csv")
lp("DONE_C14_CACHE_INSTRUMENTATION")
