# ZC-centering cache D=20 gate (2026-07-28 selective-release continuation).
#
# Real D=20/W=100,000 gate for ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] (zc_restriction_operator.jl):
# on/off comparison for origin_zc (ZC-only) and cm_meanzc (CM+ZC), through the real public
# checkpointed drivers ONLY (run_originzc_upper_checkpointed / run_cm_upper_checkpointed) -- never
# a low-level helper, per this task's own discipline. Run ALONE on this host (no other concurrent
# KNITRO process) -- cm_meanzc's real driver is known-sensitive to host KNITRO-concurrency
# contention (docs/HANDOVER_NOTE_2026-07-28.md), and this gate needs a clean signal for both
# families.
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=<worktree-root> -t 20 \
#            zc_centering_d20_gate_2026-07-28.jl
# ONLY_FAMILY=cm_meanzc|origin_zc restricts to one family.
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
lp("Threads.nthreads() = ", NT, "  BLAS default threads = ", BLAS.get_num_threads())

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "zc_centering_d20_gate_2026-07-28")
mkpath(OUTROOT)
const ONLY_FAMILY = get(ENV, "ONLY_FAMILY", "")
const CALIB_BUDGET = parse(Float64, get(ENV, "CALIB_BUDGET", "20.0"))
const NONCALIB_BUDGET = parse(Float64, get(ENV, "NONCALIB_BUDGET", "20.0"))
const TRAJ_BUDGET = parse(Float64, get(ENV, "TRAJ_BUDGET", "90.0"))

rows = NamedTuple[]
function record!(family, point, cache, wall_s, knitro_status, n_eval, n_grad, kappa, rebuilds, cache_hits, maxdiff, notes)
    push!(rows, (family = family, point = point, cache = cache, wall_s = wall_s, knitro_status = knitro_status,
        n_eval = n_eval, n_grad = n_grad, kappa = kappa, zc_centered_rebuilds = rebuilds,
        zc_centered_cache_hits = cache_hits, hessian_maxdiff_vs_cache_off = maxdiff, notes = notes))
    @printf("  [%-10s|%-18s|cache=%-5s] wall=%.1fs status=%s n_eval=%d n_grad=%d rebuilds=%d hits=%d maxdiff=%.3e\n",
        family, point, string(cache), wall_s, knitro_status, n_eval, n_grad, rebuilds, cache_hits, maxdiff)
    flush(stdout)
end

counters_snapshot() = (NO_DENSE_G_COUNTERS[].zc_centered_rebuilds, NO_DENSE_G_COUNTERS[].zc_centered_cache_hits)

# ================================================================================================
# origin_zc
# ================================================================================================
function originzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    return vcat(w_a_calib, log.(nu0))
end

function run_originzc_driver(w0::Vector{Float64}, maxtime_real::Float64, run_id::String)
    ORIGINZC_LIVE_PCX_STASH[] = nothing
    out = joinpath(OUTROOT, "originzc_$(run_id)_t$(NT)")
    rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    result = run_originzc_upper_checkpointed(w0;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        ckpt_dir = out, run_id = "originzc_$(run_id)_t$(NT)", label = "originzc_$(run_id)_t$(NT)",
        checkpoint_interval_s = 3600.0, maxtime_real = maxtime_real, verbose = true)
    wall = time() - t0
    handle = ORIGINZC_LIVE_PCX_STASH[]
    handle === nothing && error("run_originzc_driver($run_id): ORIGINZC_LIVE_PCX_STASH is empty after driver run")
    return (handle = handle, result = result, wall = wall)
end

"Recompute the packed Hessian on the FROZEN dual state (obj.arg2) a real driver call already left
behind, toggling the cache flag -- isolates the cache's algebraic effect from any dual-state
trajectory difference between separate driver calls (matches the CM+ZC isolated gate's own
within-run recompute methodology)."
function packed_hessian_originzc(handle; cache::Bool)
    ctx_cm = handle.ctx_cm; octx = handle.octx; obj_o = ctx_cm.obj
    n = octx.NCORE + octx.n_eta
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _prep_dual_index_for_archA!(octx, obj_o, obj_o.arg2)
    old = ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = cache
    try
        archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = obj_o.arg2,), (hess = h,), obj_o)
    finally
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = old
    end
    return h
end

# ================================================================================================
# cm_meanzc
# ================================================================================================
function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

function run_cmzc_driver(w0::Vector{Float64}, maxtime_real::Float64, run_id::String)
    CMZC_LIVE_PCX_STASH[] = nothing
    out = joinpath(OUTROOT, "cmzc_$(run_id)_t$(NT)")
    rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    result = run_cm_upper_checkpointed(w0;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
        probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = out, run_id = "cmzc_$(run_id)_t$(NT)", label = "cmzc_$(run_id)_t$(NT)",
        checkpoint_interval_s = 3600.0, maxtime_real = maxtime_real, verbose = true)
    wall = time() - t0
    handle = CMZC_LIVE_PCX_STASH[]
    handle === nothing && error("run_cmzc_driver($run_id): CMZC_LIVE_PCX_STASH is empty after driver run")
    return (handle = handle, result = result, wall = wall)
end

function packed_hessian_cmzc(handle; cache::Bool)
    cctx = handle.cctx; ctx_cm = handle.ctx_cm; obj_z = ctx_cm.obj
    n = cctx.NCORE + cctx.ncm
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    old = ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = cache
    try
        archC_hess_cb_builder(cctx)(nothing, nothing, (x = obj_z.arg2,), (hess = h,), obj_z)
    finally
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = old
    end
    return h
end

# ================================================================================================
# Orchestration: for each family, 3 points x {cache off, cache on}, sequential, isolated.
# ================================================================================================
function run_family_gate!(family::String, w0_calib, w0_noncalib, driver_fn, packed_fn)
    for (point_label, w0_, budget_) in [("calibration", w0_calib, CALIB_BUDGET),
                                          ("non_calibration", w0_noncalib, NONCALIB_BUDGET),
                                          ("solver_trajectory", w0_calib, TRAJ_BUDGET)]
        local correctness_maxdiff = NaN
        local correctness_notes = "not computed"
        local first_handle = nothing

        # Two separate full driver calls (cache off / cache on) for the real wall-time and
        # rebuild/cache-hit-count evidence -- this is where the actual performance claim comes from.
        for cache in (false, true)
            ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = cache
            NO_DENSE_G_COUNTERS[] = NoDenseGCounters()
            local run, status_str, kappa_str, rebuilds, hits
            try
                run = driver_fn(w0_, budget_, "$(point_label)_cache$(cache)")
            catch e
                msg = sprint(showerror, e)
                lp("!!! DRIVER FAILED family=", family, " point=", point_label, " cache=", cache, " -- ", msg)
                for (i, fr) in enumerate(stacktrace(catch_backtrace()))
                    i > 8 && break
                    lp("      at ", fr)
                end
                record!(family, point_label, cache, NaN, "DRIVER_ERROR", -1, -1, "n/a", -1, -1, NaN, msg[1:min(end, 400)])
                continue
            end
            r = run.result
            status_str = string(r.knitro_status)
            kappa_str = hasproperty(r, :kappa) ? string(r.kappa) : "n/a"
            rebuilds, hits = counters_snapshot()

            # Correctness check, done ONCE per point using the cache=false run's own frozen dual
            # state (reused here, no extra driver call needed): recompute the packed Hessian TWICE
            # on that SAME frozen state (cache off, cache on) -- isolates the cache's algebraic
            # effect from any KNITRO-trajectory difference a second driver call would introduce.
            if !cache
                first_handle = run.handle
                try
                    h_off = packed_fn(first_handle; cache = false)
                    h_on = packed_fn(first_handle; cache = true)
                    correctness_maxdiff = maximum(abs.(h_off .- h_on))
                    correctness_notes = "recomputed on ONE frozen dual state from this cache=false driver call"
                    lp("  [CORRECTNESS ", family, "|", point_label, "] recompute maxdiff=", correctness_maxdiff)
                catch e
                    correctness_maxdiff = NaN
                    correctness_notes = "correctness recompute FAILED: " * sprint(showerror, e)[1:min(end, 300)]
                    lp("!!! CORRECTNESS RECOMPUTE FAILED family=", family, " point=", point_label, " -- ", sprint(showerror, e))
                    for (i, fr) in enumerate(stacktrace(catch_backtrace()))
                        i > 12 && break
                        lp("      at ", fr)
                    end
                end
            end

            record!(family, point_label, cache, run.wall, status_str, r.n_eval, r.n_grad, kappa_str, rebuilds, hits,
                cache ? correctness_maxdiff : 0.0,
                cache ? correctness_notes : "reference (cache off) -- see cache=true row for the recompute-based correctness check")
        end
    end
end

if isempty(ONLY_FAMILY) || ONLY_FAMILY == "origin_zc"
    lp("\n", "="^100, "\n=== origin_zc: building ctx0 for w0 construction ===")
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
    pe0 = build_pivot_elimination(ctx0)
    theta0 = cm_fixed_theta(ctx0)
    xy0 = precompute_cm_aspace_xy(ctx0)
    w0_calib = originzc_w0(ctx0, pe0, theta0, xy0)
    rng = MersenneTwister(2026_0728)
    w0_noncalib = w0_calib .* (1.0 .+ 1e-2 .* (rand(rng, length(w0_calib)) .- 0.5))
    run_family_gate!("origin_zc", w0_calib, w0_noncalib, run_originzc_driver, packed_hessian_originzc)
end

if isempty(ONLY_FAMILY) || ONLY_FAMILY == "cm_meanzc"
    lp("\n", "="^100, "\n=== cm_meanzc: building ctx0 for w0 construction ===")
    ctx0c = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
    pe0c = build_pivot_elimination(ctx0c)
    theta0c = cm_fixed_theta(ctx0c)
    xy0c = precompute_cm_aspace_xy(ctx0c)
    w0c_calib = cmzc_w0(ctx0c, pe0c, theta0c, xy0c)
    rngc = MersenneTwister(2026_0729)
    w0c_noncalib = w0c_calib .* (1.0 .+ 1e-2 .* (rand(rngc, length(w0c_calib)) .- 0.5))
    run_family_gate!("cm_meanzc", w0c_calib, w0c_noncalib, run_cmzc_driver, packed_hessian_cmzc)
end

csvpath = joinpath(D4X, "..", "..", "docs", "ZC_CENTERING_D20_GATE_2026-07-28.csv")
mkpath(dirname(csvpath))
open(csvpath, "w") do io
    println(io, "family,point,cache,wall_s,knitro_status,n_eval,n_grad,kappa,zc_centered_rebuilds,zc_centered_cache_hits,hessian_maxdiff_vs_cache_off,notes")
    for r in rows
        notes_escaped = replace(r.notes, "\"" => "'", "\n" => " ")
        println(io, "$(r.family),$(r.point),$(r.cache),$(r.wall_s),$(r.knitro_status),$(r.n_eval),$(r.n_grad),$(r.kappa),$(r.zc_centered_rebuilds),$(r.zc_centered_cache_hits),$(r.hessian_maxdiff_vs_cache_off),\"$(notes_escaped)\"")
    end
end
lp("Wrote ", csvpath)
lp("DONE.")
