# diag_allocation_stacktrace_2026-08-02.jl -- real Profile.Allocs allocation-site breakdown for
# ONE family's frozen-state Hessian callback, to explain (not assume) the per-callback byte gaps
# the canonical harness found:
#   common_frechet (561.7KB) vs flexible_cm (41.6KB) at similar n (1382 vs 1332)
#   cm_meanzc (1996KB) vs origin_zc (48.1KB) at n=1962 vs n=1012
# Reuses the SAME canonical include stack + per-family setup logic as
# production_all_hessian_audit_harness_2026-08-02.jl (copy-pasted setup functions, not a refactor
# of the already-validated harness, to avoid touching tested code mid-investigation).
#
# ENV: AUDIT_FAMILY, AUDIT_W (default 20000)
const D4X = @__DIR__
cd(D4X)

const _CM_FAMILY_LIST = [
    "draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
    "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
    "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
    "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
    "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
    "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
    "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
    "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
    "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
    "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
    "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
    "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
    "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
    "hcz_drawchunk_candidate_2026-07-29.jl",
    "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl",
]
for f in _CM_FAMILY_LIST
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random, Statistics, Dates, Profile
using Base.Threads: nthreads

lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = Symbol(get(ENV, "AUDIT_FAMILY", "cm_meanzc"))
const W = parse(Int, get(ENV, "AUDIT_W", "20000"))
const K_ZC = 3
const CM_L = 50

lp("=== allocation stacktrace diag: family=$FAMILY W=$W ===")
GRAV_EXCLUDE = default_gravity_exclude_cells_brazil_korea()
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV_EXCLUDE, σHat = 3.0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]

if FAMILY === :cm_meanzc
    ctx_cm, aug, cctx, bins = build_cm_meanzc_production_context(ctx0, CS; L = CM_L, K_mean = K_ZC, K_pair = K_ZC,
        contrasts = :orthonormal, meanzc_basis = :direct)
    νvec = Float64.(factorial.(1:K_ZC))
    base = archC_meanzc_base_state(x_free_calib, νvec, ctx_cm, cctx)
    global userParams = ctx_cm.obj
    global cb = archC_hess_cb_builder(cctx)
    global x_state = vcat(base.ζstar, base.λstar)
elseif FAMILY === :origin_zc
    layout = OriginByPowerLayout(ctx0.D, K_ZC, K_ZC)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for k in 1:K_ZC, o in 1:ctx0.D
        nu0[target_index(layout, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    pcx = build_originzc_production_context(ctx0, CS, layout; fg_backend = :operator, moment_representation = :operator)
    base = archOZ_base_state(x_free_calib, nu0, pcx.ctx_cm)
    global userParams = pcx.ctx_cm.obj
    global cb = archA_partitioned_hess_cb_builder(pcx.octx)
    global x_state = vcat(base.ζstar, base.λstar)
elseif FAMILY === :flexible_cm
    SNAPS = nested_grid_sequence([10, 20, 50])
    ctx_cm, aug, bins, cctx = build_cm_production_context(ctx0, CS; L = CM_L, contrasts = :orthonormal, probs = SNAPS[CM_L])
    base = archC_base_state(x_free_calib, ctx_cm, cctx)
    global userParams = ctx_cm.obj
    global cb = archC_hess_cb_builder(cctx)
    global x_state = vcat(base.ζstar, base.λstar)
elseif FAMILY === :common_frechet
    SNAPS = nested_grid_sequence([10, 20, 50])
    pcx = build_cm_frechet_production_context(ctx0, CS; L = CM_L, contrasts = :orthonormal, probs = SNAPS[CM_L], cm_hessian_backend = :structured)
    base = archC_frechet_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    global userParams = pcx.ctx_cm.obj
    global cb = archC_frechet_hess_cb_builder(pcx.cctx, pcx.aug.level_targets)
    global x_state = vcat(base.ζstar, base.λstar)
else
    error("unknown FAMILY=$FAMILY")
end

n = length(x_state)
hlen = n * (n + 1) ÷ 2
h = Vector{Float64}(undef, hlen)
fake_req = (x = x_state,)
fake_res = (hess = h,)

# warm-up (JIT)
cb(nothing, nothing, fake_req, fake_res, userParams)
b0 = @allocated cb(nothing, nothing, fake_req, fake_res, userParams)
lp(">> post-warmup single-call bytes = ", b0)

# Profile.Allocs: sample_rate=1.0 (every allocation, not sampled) since callback is short/cheap
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 begin
    cb(nothing, nothing, fake_req, fake_res, userParams)
end
results = Profile.Allocs.fetch()
lp(">> total allocs recorded = ", length(results.allocs))

# Aggregate by (type, top non-Base/stdlib frame)
using Base: StackTraces
agg = Dict{String,Tuple{Int,Int}}()   # key -> (count, bytes)
for a in results.allocs
    tstr = string(a.type)
    # first frame from THIS repo's own source (skip Base/stdlib internals) for attribution
    site = "unknown"
    for fr in a.stacktrace
        fname = string(fr.file)
        if occursin("full_aod_diag", fname)
            site = string(basename(fname), ":", fr.line, " (", fr.func, ")")
            break
        end
    end
    key = tstr * " @ " * site
    c, b = get(agg, key, (0, 0))
    agg[key] = (c + 1, b + a.size)
end
sorted = sort(collect(agg), by = x -> -x[2][2])
lp(">> TOP 25 allocation sites by total bytes (family=", FAMILY, ", W=", W, "):")
for (i, (key, (c, b))) in enumerate(sorted[1:min(25, length(sorted))])
    lp(@sprintf("  %2d. %10d bytes  (n=%5d)  %s", i, b, c, key))
end
total_b = sum(x -> x[2][2], sorted)
lp(">> TOTAL attributed bytes (Profile.Allocs, sample_rate=1.0) = ", total_b, "  (cf. @allocated single-call = ", b0, ")")

# Dump the packed Hessian for a before/after correctness A/B (git-stash-based, see caller script).
dumpfile = get(ENV, "AUDIT_H_DUMP", "")
if !isempty(dumpfile)
    open(dumpfile, "w") do io
        for v in h
            println(io, v)
        end
    end
    lp(">> wrote packed Hessian (", length(h), " values) to ", dumpfile)
end
