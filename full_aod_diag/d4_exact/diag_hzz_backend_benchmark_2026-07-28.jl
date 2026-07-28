# D=20 profiling task (Section 9): H_ZZ backend benchmark at production width + K-width
# sensitivity + BLAS-thread sweep, for cm_meanzc and origin_zc. Reuses the SAME real-driver-based
# warm-handle pattern as diag_cmzc_originzc_d20_profile_2026-07-28.jl (see that file's header for
# the KNITRO-callback-err rationale). Run at Julia threads=1 (BLAS threads swept at runtime via
# BLAS.set_num_threads, no process relaunch needed) -- the Julia-threads dimension of this same
# sweep (workers in {1,4,8,10,20} at Julia-thread level) is already covered by
# diag_cmzc_originzc_d20_profile_2026-07-28.jl's own t1/t4/t8/t10/t20 sweep at K_mean=1/K_pair=1
# (production width) -- this script's job is the BLAS-thread axis plus K-width sensitivity, which
# that script does not cover.
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t 1 diag_hzz_backend_benchmark_2026-07-28.jl
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics, Dates
lp(xs...) = (println(xs...); flush(stdout))

const NT = Threads.nthreads()
lp("Threads.nthreads() = ", NT, "  (BLAS threads swept at runtime; Julia threads should be 1 for this script)")

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "hzz_backend_benchmark_2026-07-28")
mkpath(OUTROOT)
const REPS = 5
function timeit(f::Function; reps::Int = REPS)
    f()
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        f()
        ts[i] = (time_ns() - t0) / 1e9
    end
    return (minimum(ts), sum(ts) / reps)
end

rows = NamedTuple[]
function record!(family, kconfig, backend, blas_threads, nx, tmin, tmean; maxdiff = NaN)
    push!(rows, (family = family, kconfig = kconfig, backend = backend, julia_threads = NT,
        blas_threads = blas_threads, nx = nx, t_min_s = tmin, t_mean_s = tmean, maxdiff_vs_reference = maxdiff))
    @printf("  [%-10s|%-10s] backend=%-16s blas_t=%-3d nx=%-4d min=%.4fs mean=%.4fs maxdiff=%.3e\n",
        family, kconfig, backend, blas_threads, nx, tmin, tmean, maxdiff)
    flush(stdout)
end

"Sweep BLAS threads for :blas_syrk/:blas_gemm at the given (op, w, M) state (nx = restriction width)."
function sweep_hzz_backends!(family, kconfig, op, w, M; blas_threads_list = [1, 4, 8, 10, 20])
    nx = n_restriction(op)
    hzz_ws = ZCRestrictionWorkspace(op)
    # need a centered scratch + targets to get the :reference baseline once (correctness anchor)
    zws = hzz_ws
    HZZ_ref = Matrix{Float64}(undef, nx, nx)
    cs = ensure_zc_centered_scratch!(nothing, op, length(w))
    refresh_zc_centered!(cs, op, zws, w; fill_S = true)
    zc_restriction_gram!(HZZ_ref, cs, op, M)
    old_blas = BLAS.get_num_threads()

    raw_ws = build_zc_raw_weighted_workspace(op, length(w))
    refresh_zc_raw_target_vector!(raw_ws, zws, op)
    HZZ = Matrix{Float64}(undef, nx, nx)
    for bt in blas_threads_list
        bt > Sys.CPU_THREADS && continue
        BLAS.set_num_threads(bt)
        tmin, tmean = timeit(() -> zc_gram_blas_syrk!(HZZ, raw_ws, w, M))
        record!(family, kconfig, "blas_syrk", bt, nx, tmin, tmean; maxdiff = maximum(abs.(HZZ .- HZZ_ref)))
        tmin, tmean = timeit(() -> zc_gram_blas_gemm!(HZZ, raw_ws, w, M))
        record!(family, kconfig, "blas_gemm", bt, nx, tmin, tmean; maxdiff = maximum(abs.(HZZ .- HZZ_ref)))
    end
    BLAS.set_num_threads(old_blas)
    # reference + threaded_packed at Julia-threads=NT for completeness of this K config (not swept
    # over BLAS threads -- threaded_packed doesn't use BLAS; reference uses BLAS but at nx this
    # small the reference gemm cost is dominated by centering-prep, not the gemm itself, per the
    # production-width sweep already on record)
    tmin, tmean = timeit(() -> zc_restriction_gram!(HZZ, cs, op, M))
    record!(family, kconfig, "reference", old_blas, nx, tmin, tmean; maxdiff = maximum(abs.(HZZ .- HZZ_ref)))
    tmin, tmean = timeit(() -> zc_gram_threaded_packed!(HZZ, raw_ws, w, M; workers = NT))
    record!(family, kconfig, "threaded_packed", old_blas, nx, tmin, tmean; maxdiff = maximum(abs.(HZZ .- HZZ_ref)))
end

# ---------------- origin_zc: K_mean/K_pair sensitivity (reliable real driver) ----------------
lp("\n", "="^100, "\n=== origin_zc: K-width sensitivity + BLAS-thread sweep ===")
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)

function originzc_w0_K(K_mean::Int, K_pair::Int)
    # n_eta(layout) = K_mean*D regardless of K_pair -- pair targets are DERIVED from the same
    # K_mean nu_{o,k} values (nu_{o,k}*nu_{p,k}, see pair_targets/cm_originzc_target_layout.jl),
    # not independent free parameters. K_pair only changes what the driver computes internally.
    layout0 = OriginByPowerLayout(D, K_mean, K_pair)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:K_mean, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    return vcat(w_a_calib, log.(nu0))
end

for (K_mean, K_pair, tag) in [(1, 0, "K1_K0"), (1, 1, "K1_K1_prod"), (2, 1, "K2_K1_wide")]
    lp("\n--- origin_zc K_mean=$K_mean K_pair=$K_pair ($tag) ---")
    out = joinpath(OUTROOT, "originzc_$(tag)")
    rm(out; force = true, recursive = true); mkpath(out)
    w0 = originzc_w0_K(K_mean, K_pair)
    local result
    try
        result = run_originzc_upper_checkpointed(w0;
            W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
            distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = K_mean, K_pair = K_pair,
            ckpt_dir = out, run_id = "hzz_$(tag)", label = "hzz_$(tag)",
            checkpoint_interval_s = 3600.0, maxtime_real = 15.0, verbose = false)
        handle = ORIGINZC_LIVE_PCX_STASH[]
        octx = handle.octx; obj_o = handle.ctx_cm.obj
        w_o = copy(obj_o.arg2); M_o = obj_o.M
        op_o = octx.hzz_zc_op
        refresh_zc_targets!(octx.hzz_zc_ws, op_o, octx.hzz_zc_layout, octx.nu_ref[])
        sweep_hzz_backends!("origin_zc", tag, op_o, w_o, M_o)
    catch e
        println("!!! origin_zc/$tag FAILED -- ", sprint(showerror, e)); flush(stdout)
    end
end

# ---------------- cm_meanzc: production width only (its driver is known-broken, see main profile
# script's header -- reuse the SAME unsolved-timing-only fallback, shape/algebra still meaningful) ----------------
lp("\n", "="^100, "\n=== cm_meanzc: K1_K1 production width (UNSOLVED_TIMING_ONLY, see caveat) ===")
try
    out = joinpath(OUTROOT, "cmzc_K1_K1_prod")
    rm(out; force = true, recursive = true); mkpath(out)
    w0c = vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
    result = run_cm_upper_checkpointed(w0c;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
        probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = out, run_id = "hzz_cmzc", label = "hzz_cmzc",
        checkpoint_interval_s = 3600.0, maxtime_real = 15.0, verbose = false)
    handle = CMZC_LIVE_PCX_STASH[]
    cctx = handle.cctx; obj_z = handle.ctx_cm.obj
    w_z = copy(obj_z.arg2); M_z = obj_z.M
    op_z = cctx.hzz_zc_op
    refresh_zc_targets!(cctx.hzz_zc_ws, op_z, cctx.hzz_zc_layout, cctx.nu_ref[])
    sweep_hzz_backends!("cm_meanzc", "K1_K1_prod_UNSOLVED", op_z, w_z, M_z)
catch e
    println("!!! cm_meanzc FAILED -- ", sprint(showerror, e)); flush(stdout)
end

outpath = joinpath(D4X, "..", "..", "docs", "HZZ_BACKEND_PRODUCTION_WIDTH_BENCHMARK_2026-07-28.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,kconfig,backend,julia_threads,blas_threads,nx,t_min_s,t_mean_s,maxdiff_vs_reference")
    for r in rows
        println(io, "$(r.family),$(r.kconfig),$(r.backend),$(r.julia_threads),$(r.blas_threads),$(r.nx),$(r.t_min_s),$(r.t_mean_s),$(r.maxdiff_vs_reference)")
    end
end
lp("Wrote ", outpath)
lp("DONE.")
