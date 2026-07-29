# Final four-family production gate (2026-07-28 selective-release).
#
# One real D=20/W=100,000 call per restricted family (flexible_cm, common_frechet, cm_meanzc,
# origin_zc) through the actual public checkpointed drivers, WITH THE NEWLY-MERGED DEFAULTS ACTIVE
# (threaded H_EC/H_EZ on for all four, ZC-centering cache on, H_ZZ backend :reference) -- confirms
# the whole merged release branch works end-to-end per family and that the architecture invariants
# (no legacy-H, no composite-G, no moments!/select_G_from_H materializations) still hold with every
# new default engaged together. Run ALONE (no concurrent KNITRO) per this release's own established
# convention.
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

lp("Defaults in effect: CROSS_HESSIAN_THREADED_DEFAULT[]=", CROSS_HESSIAN_THREADED_DEFAULT[],
   " CROSS_HESSIAN_WORKERS_DEFAULT[]=", CROSS_HESSIAN_WORKERS_DEFAULT[],
   " ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]=", ZC_CENTERED_CACHE_ACROSS_CALLBACKS[],
   " ZC_GRAM_BACKEND_DEFAULT[]=", ZC_GRAM_BACKEND_DEFAULT[])

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "final_four_family_gate_2026-07-28")
mkpath(OUTROOT)

rows = NamedTuple[]
function record!(family, wall_s, status, n_eval, n_grad, kappa, counters, notes)
    push!(rows, (family = family, wall_s = wall_s, knitro_status = status, n_eval = n_eval, n_grad = n_grad,
        kappa = kappa, full_G_materializations = counters.full_G_materializations,
        dense_economic_G = counters.dense_economic_G_materializations,
        dense_CM_G = counters.dense_CM_G_materializations, dense_ZC_G = counters.dense_ZC_G_materializations,
        dense_Frechet_G = counters.dense_Frechet_G_materializations,
        zc_centered_rebuilds = counters.zc_centered_rebuilds, zc_centered_cache_hits = counters.zc_centered_cache_hits,
        notes = notes))
    @printf("  [%-14s] wall=%.1fs status=%s n_eval=%d n_grad=%d kappa=%s | full_G=%d dense_econ=%d dense_CM=%d dense_ZC=%d dense_Fr=%d | zc_rebuilds=%d zc_hits=%d\n",
        family, wall_s, status, n_eval, n_grad, kappa, counters.full_G_materializations,
        counters.dense_economic_G_materializations, counters.dense_CM_G_materializations,
        counters.dense_ZC_G_materializations, counters.dense_Frechet_G_materializations,
        counters.zc_centered_rebuilds, counters.zc_centered_cache_hits)
    flush(stdout)
end

function calib_w0_cm(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    return vcat(gp_calib, a_calib)
end

function calib_w0_originzc(ctx0, pe0, theta0, xy0)
    w_a = calib_w0_cm(ctx0, pe0, theta0, xy0)
    D = ctx0.D
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    return vcat(w_a, log.(nu0))
end

ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)

# Parallel-launch support (2026-07-28, user requested running all 4 families concurrently as
# separate processes to save wall time): ONLY_FAMILY restricts this process to one family, and
# each process appends its own row to the shared CSV rather than the whole script writing it once
# at the end -- matches the append pattern already proven in hzz_resource_gate_worker_2026-07-28.jl.
# cm_meanzc's own real driver is KNOWN-sensitive to concurrent KNITRO load on this host (see
# docs/HANDOVER_NOTE_2026-07-28.md and this release's own CM+ZC isolated gate) -- if it fails here
# under concurrency, that is an expected host-contention risk, not treated as a code regression;
# rerun it alone if so.
const ONLY_FAMILY = get(ENV, "ONLY_FAMILY", "")
run_family(name) = isempty(ONLY_FAMILY) || ONLY_FAMILY == name
const CSV_APPEND_PATH = joinpath(D4X, "..", "..", "docs", "FINAL_FOUR_FAMILY_PRODUCTION_GATE_2026-07-28.csv")
function append_row!(r)
    open(CSV_APPEND_PATH, "a") do io
        println(io, "$(r.family),$(r.wall_s),$(r.knitro_status),$(r.n_eval),$(r.n_grad),$(r.kappa),$(r.full_G_materializations),$(r.dense_economic_G),$(r.dense_CM_G),$(r.dense_ZC_G),$(r.dense_Frechet_G),$(r.zc_centered_rebuilds),$(r.zc_centered_cache_hits),\"$(r.notes)\"")
    end
end

# ---- flexible_cm ----
if run_family("flexible_cm")
try
    reset_no_dense_g_counters!()
    w0 = calib_w0_cm(ctx0, pe0, theta0, xy0)
    out = joinpath(OUTROOT, "flexible_cm"); rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    r = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_only, marginal_restriction = :common_flexible,
        ckpt_dir = out, run_id = "final_flexcm", label = "final_flexcm",
        checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
    wall = time() - t0
    record!("flexible_cm", wall, string(r.knitro_status), r.n_eval, r.n_grad,
        hasproperty(r, :kappa) ? string(r.kappa) : "n/a", NO_DENSE_G_COUNTERS[], "ok")
    append_row!(rows[end])
catch e
    lp("!!! flexible_cm FAILED -- ", sprint(showerror, e))
    append_row!((family = "flexible_cm", wall_s = NaN, knitro_status = "ERROR", n_eval = -1, n_grad = -1,
        kappa = "n/a", full_G_materializations = -1, dense_economic_G = -1, dense_CM_G = -1, dense_ZC_G = -1,
        dense_Frechet_G = -1, zc_centered_rebuilds = -1, zc_centered_cache_hits = -1,
        notes = replace(sprint(showerror, e)[1:min(end,200)], "\"" => "'", "\n" => " ")))
end
end # run_family("flexible_cm")

# ---- common_frechet ----
if run_family("common_frechet")
try
    reset_no_dense_g_counters!()
    w0 = calib_w0_cm(ctx0, pe0, theta0, xy0)
    out = joinpath(OUTROOT, "common_frechet"); rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    r = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_only, marginal_restriction = :common_frechet,
        ckpt_dir = out, run_id = "final_frechet", label = "final_frechet",
        checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
    wall = time() - t0
    record!("common_frechet", wall, string(r.knitro_status), r.n_eval, r.n_grad,
        hasproperty(r, :kappa) ? string(r.kappa) : "n/a", NO_DENSE_G_COUNTERS[], "ok")
    append_row!(rows[end])
catch e
    lp("!!! common_frechet FAILED -- ", sprint(showerror, e))
    append_row!((family = "common_frechet", wall_s = NaN, knitro_status = "ERROR", n_eval = -1, n_grad = -1,
        kappa = "n/a", full_G_materializations = -1, dense_economic_G = -1, dense_CM_G = -1, dense_ZC_G = -1,
        dense_Frechet_G = -1, zc_centered_rebuilds = -1, zc_centered_cache_hits = -1,
        notes = replace(sprint(showerror, e)[1:min(end,200)], "\"" => "'", "\n" => " ")))
end
end # run_family("common_frechet")

# ---- cm_meanzc (CM+ZC) ----
if run_family("cm_meanzc")
try
    reset_no_dense_g_counters!()
    w0 = vcat(calib_w0_cm(ctx0, pe0, theta0, xy0), log.(Float64.(factorial.(1:1))))
    out = joinpath(OUTROOT, "cm_meanzc"); rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    r = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[50],
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = out, run_id = "final_cmzc", label = "final_cmzc",
        checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
    wall = time() - t0
    record!("cm_meanzc", wall, string(r.knitro_status), r.n_eval, r.n_grad,
        hasproperty(r, :kappa) ? string(r.kappa) : "n/a", NO_DENSE_G_COUNTERS[], "ok")
    append_row!(rows[end])
catch e
    lp("!!! cm_meanzc FAILED -- ", sprint(showerror, e))
    append_row!((family = "cm_meanzc", wall_s = NaN, knitro_status = "ERROR", n_eval = -1, n_grad = -1,
        kappa = "n/a", full_G_materializations = -1, dense_economic_G = -1, dense_CM_G = -1, dense_ZC_G = -1,
        dense_Frechet_G = -1, zc_centered_rebuilds = -1, zc_centered_cache_hits = -1,
        notes = replace(sprint(showerror, e)[1:min(end,200)], "\"" => "'", "\n" => " ")))
end
end # run_family("cm_meanzc")

# ---- origin_zc (ZC-only) ----
if run_family("origin_zc")
try
    reset_no_dense_g_counters!()
    w0 = calib_w0_originzc(ctx0, pe0, theta0, xy0)
    out = joinpath(OUTROOT, "origin_zc"); rm(out; force = true, recursive = true); mkpath(out)
    t0 = time()
    r = run_originzc_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        ckpt_dir = out, run_id = "final_originzc", label = "final_originzc",
        checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
    wall = time() - t0
    record!("origin_zc", wall, string(r.knitro_status), r.n_eval, r.n_grad,
        hasproperty(r, :kappa) ? string(r.kappa) : "n/a", NO_DENSE_G_COUNTERS[], "ok")
    append_row!(rows[end])
catch e
    lp("!!! origin_zc FAILED -- ", sprint(showerror, e))
    append_row!((family = "origin_zc", wall_s = NaN, knitro_status = "ERROR", n_eval = -1, n_grad = -1,
        kappa = "n/a", full_G_materializations = -1, dense_economic_G = -1, dense_CM_G = -1, dense_ZC_G = -1,
        dense_Frechet_G = -1, zc_centered_rebuilds = -1, zc_centered_cache_hits = -1,
        notes = replace(sprint(showerror, e)[1:min(end,200)], "\"" => "'", "\n" => " ")))
end
end # run_family("origin_zc")

lp("Appended result row(s) to ", CSV_APPEND_PATH)
lp("DONE.")
