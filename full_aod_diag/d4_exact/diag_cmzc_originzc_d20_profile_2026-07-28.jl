# D=20 profiling task (cm_meanzc/origin_zc H_EC/H_EZ/H_CZ/H_ZZ), 2026-07-28.
#
# Real D=20/Ddest=19/W=100,000 sub-block timing profile + H_ZZ backend benchmark + correctness
# gates for cm_meanzc (CM+ZC) and origin_zc (ZC-only), routed EXCLUSIVELY through the real public
# checkpointed drivers (run_cm_upper_checkpointed / run_originzc_upper_checkpointed) per this task's
# brief -- calling the low-level inner-solve helpers directly is confirmed to reliably raise
# KNITRO's KN_RC_CALLBACK_ERR for cm_meanzc (STRUCTURED_CROSS_HESSIAN_MASTER_REPORT_2026-07-28.md
# Sec 5). After each short driver call, the opt-in CMZC_LIVE_PCX_STASH/ORIGINZC_LIVE_PCX_STASH
# (cross_hessian_live_stash_2026-07-28.jl) gives direct read access to the SAME live cctx/octx and
# obj (with obj.arg2 = S already filled by a REAL KNITRO Hessian callback) the driver run itself
# used -- safe to feed into the pure, KNITRO-free standalone sub-block kernels for timing/
# correctness microbenchmarks, exactly the pattern diag_subblock_profile_2026-07-28.jl already used
# successfully for origin_zc.
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t <N> \
#            diag_cmzc_originzc_d20_profile_2026-07-28.jl
# ONLY_FAMILY=cm_meanzc|origin_zc restricts to one family (recommended: run cm_meanzc alone first,
# since its own KNITRO behavior is the less-tested path).
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
const OUTROOT = joinpath(D4X, "..", "..", "results", "cmzc_originzc_d20_profile_2026-07-28")
mkpath(OUTROOT)

const REPS = 6
"min/mean seconds over REPS calls, after one untimed warmup call."
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

rows = NamedTuple[]        # family,block,nthreads,t_min_s,t_mean_s,point
gate_rows = NamedTuple[]   # family,point,check,workers_or_backend,maxdiff,pass
function record!(family, block, tmin, tmean, point)
    push!(rows, (family = family, block = block, nthreads = NT, t_min_s = tmin, t_mean_s = tmean, point = point))
    @printf("  [%-10s|%-18s] %-32s  min=%.4fs  mean=%.4fs\n", family, point, block, tmin, tmean)
    flush(stdout)
end
function record_gate!(family, point, check, key, maxdiff; tol = 1e-9)
    ok = maxdiff <= tol
    push!(gate_rows, (family = family, point = point, check = check, key = key, maxdiff = maxdiff, tol = tol, pass = ok))
    @printf("  [GATE %-10s|%-18s] %-28s %-20s maxdiff=%.3e tol=%.1e %s\n", family, point, check, key, maxdiff, tol, ok ? "PASS" : "FAIL")
    flush(stdout)
end

# ================================================================================================
# origin_zc: build w0 (calibration) exactly as smoke_delta1_originzc.jl does.
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
    @printf("  [DRIVER origin_zc/%s] wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d n_hess=%s kappa=%s\n",
        run_id, wall, string(result.knitro_status), result.n_eval, result.n_grad,
        hasproperty(result, :n_hess) ? string(result.n_hess) : "n/a", string(result.kappa))
    flush(stdout)
    handle = ORIGINZC_LIVE_PCX_STASH[]
    handle === nothing && error("run_originzc_driver($run_id): ORIGINZC_LIVE_PCX_STASH is empty after driver run")
    obj_o = handle.ctx_cm.obj
    normS = norm(obj_o.arg2)
    (isfinite(normS) && normS > 0) || error("run_originzc_driver($run_id): obj.arg2 (S) looks uninitialized (norm=$normS) -- no real Hessian callback fired in this budget, increase maxtime_real")
    lp("  [DRIVER origin_zc/$run_id] captured live handle, norm(S)=", normS)
    return (handle = handle, result = result, wall = wall)
end

function profile_originzc_point!(handle, point_label)
    ctx_cm = handle.ctx_cm; octx = handle.octx; obj_o = ctx_cm.obj
    w_o = copy(obj_o.arg2); M_o = obj_o.M
    NCORE_o = octx.NCORE; n_eta_o = octx.n_eta
    lp("  dims origin_zc: n_E=", NCORE_o, " n_Z=", n_eta_o, " (no C block)")

    HEE_o = Matrix{Float64}(undef, NCORE_o, NCORE_o)
    tmin, tmean = timeit(() -> fill_core_hessian_upper!(HEE_o, w_o, obj_o, octx.core_ws;
        backend = octx.core_hessian_backend, workers = octx.core_hessian_workers, storage = octx.core_hessian_storage))
    record!("origin_zc", "H_EE", tmin, tmean, point_label)

    wctx_o = serial_ctx(octx.core_ws)
    op_o = octx.hzz_zc_op
    refresh_zc_targets!(octx.hzz_zc_ws, op_o, octx.hzz_zc_layout, octx.nu_ref[])
    octx.hzz_centered = ensure_zc_centered_scratch!(octx.hzz_centered, op_o, size(w_o, 1))
    tmin, tmean = timeit(() -> refresh_zc_centered!(octx.hzz_centered, op_o, octx.hzz_zc_ws, w_o))
    record!("origin_zc", "H_ZZ_centering_prep(fill_S=true)", tmin, tmean, point_label)

    octx.zc_cross_scratch = _ensure_originzc_zc_cross_scratch!(octx, wctx_o.W, n_eta_o)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_prep!(octx.zc_cross_scratch, wctx_o, w_o))
    record!("origin_zc", "H_EZ_Snu_prep(=H_ER prep)", tmin, tmean, point_label)

    nx_o = n_restriction(op_o)
    Zview_o = @view octx.hzz_centered.Zc[:, 1:nx_o]
    HER_o = Matrix{Float64}(undef, wctx_o.ncolI + 1, nx_o)
    winner_pair_cross_hessian_zc_block!(HER_o, wctx_o, octx.zc_cross_scratch, w_o, Zview_o, M_o)
    HER_o_ref = copy(HER_o)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block!(HER_o, wctx_o, octx.zc_cross_scratch, w_o, Zview_o, M_o))
    record!("origin_zc", "H_EZ(=HER)", tmin, tmean, point_label)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_threaded!(HER_o, wctx_o, octx.zc_cross_scratch, w_o, Zview_o, M_o; workers = NT))
    record!("origin_zc", "H_EZ(=HER)_threaded", tmin, tmean, point_label)
    record_gate!("origin_zc", point_label, "H_EZ_threaded_vs_serial", "workers=$NT", maximum(abs.(HER_o .- HER_o_ref)); tol = 1e-12)

    HRR_o = Matrix{Float64}(undef, nx_o, nx_o)
    tmin, tmean = timeit(() -> zc_restriction_gram!(HRR_o, octx.hzz_centered, op_o, M_o))
    record!("origin_zc", "H_ZZ(=HRR)_reference", tmin, tmean, point_label)
    HRR_o_ref = copy(HRR_o)

    raw_ws_o = build_zc_raw_weighted_workspace(op_o, wctx_o.W)
    refresh_zc_raw_target_vector!(raw_ws_o, octx.hzz_zc_ws, op_o)
    tmin, tmean = timeit(() -> zc_gram_blas_syrk!(HRR_o, raw_ws_o, w_o, M_o))
    record!("origin_zc", "H_ZZ(=HRR)_blas_syrk", tmin, tmean, point_label)
    record_gate!("origin_zc", point_label, "H_ZZ_blas_syrk_vs_reference", "K1_1", maximum(abs.(HRR_o .- HRR_o_ref)))
    tmin, tmean = timeit(() -> zc_gram_blas_gemm!(HRR_o, raw_ws_o, w_o, M_o))
    record!("origin_zc", "H_ZZ(=HRR)_blas_gemm", tmin, tmean, point_label)
    record_gate!("origin_zc", point_label, "H_ZZ_blas_gemm_vs_reference", "K1_1", maximum(abs.(HRR_o .- HRR_o_ref)))
    tmin, tmean = timeit(() -> zc_gram_threaded_packed!(HRR_o, raw_ws_o, w_o, M_o; workers = NT))
    record!("origin_zc", "H_ZZ(=HRR)_threaded_packed", tmin, tmean, point_label)
    record_gate!("origin_zc", point_label, "H_ZZ_threaded_packed_vs_reference", "K1_1_workers$NT", maximum(abs.(HRR_o .- HRR_o_ref)))
end

# ================================================================================================
# cm_meanzc: build w0 (calibration) exactly as smoke_delta1_cmzc.jl does. cm_extension=:cm_plus_moments,
# K_mean=1, K_pair=1 (production width per that smoke test's own call pattern).
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
    @printf("  [DRIVER cm_meanzc/%s] wall=%.1fs knitro_status=%s n_eval=%d n_grad=%d kappa=%s\n",
        run_id, wall, string(result.knitro_status), result.n_eval, result.n_grad, string(result.kappa))
    flush(stdout)
    handle = CMZC_LIVE_PCX_STASH[]
    handle === nothing && error("run_cmzc_driver($run_id): CMZC_LIVE_PCX_STASH is empty after driver run")
    obj_z = handle.ctx_cm.obj
    normS = norm(obj_z.arg2)
    # Confirmed live 2026-07-28 (5 independent repro attempts: default config, verification_backend=
    # :dense_reference, K_pair=0, 1% w0 jitter, cm_gradient_backend=:reference -- all IDENTICAL
    # nStatus=-500 inside _meanzc_fg_dispatch/archC_meanzc_verified_state's own inner KNITRO solve,
    # at the VERY FIRST evaluation) -- run_cm_upper_checkpointed itself, not just the low-level
    # helpers the master report already flagged, currently fails for cm_meanzc's real inner solve on
    # THIS branch base (5b4f9da). This is a genuine, reproducible, pre-existing regression (git diff
    # fb6ad2e..5b4f9da touches cm_hessian_architectures.jl/cm_hessian_threaded.jl/
    # cm_meanzc_production.jl/operator_verification.jl -- the harmonization merge window), not a bug
    # in this task's own kernel work and not fixable within this task's scope (explicitly out of
    # scope per the task brief). obj.arg2 (S) never gets a real ddPsi! write in this case and stays
    # at its preallocated default (all-ones -> norm=sqrt(W)) -- WARN, don't hard-error: the resulting
    # sub-block TIMING is still meaningful (pure function of array shapes/threading, not of S's
    # values) and the correctness GATES (threaded-vs-serial / backend-vs-reference diffs) are still
    # valid (algebraic-equivalence checks on whatever S is, real or not) -- only the "this is a
    # genuinely solved economic state" claim is invalid, and every row/point downstream is labeled
    # accordingly (see UNSOLVED_STATE below).
    unsolved = !(isfinite(normS) && normS > 0 && result.n_eval >= 1)
    if unsolved
        lp("  [DRIVER cm_meanzc/$run_id] *** UNSOLVED_STATE *** knitro_status=", result.knitro_status,
           " n_eval=", result.n_eval, " -- obj.arg2 was never written by a real Hessian callback " *
           "(norm(S)=", normS, "). Sub-block TIMING/correctness-GATE data below is still meaningful " *
           "(shape/algebra-only) but is NOT from a converged economic state -- see driver_summary CSV.")
    else
        lp("  [DRIVER cm_meanzc/$run_id] captured live handle, norm(S)=", normS)
    end
    return (handle = handle, result = result, wall = wall, unsolved = unsolved)
end

function profile_cmzc_point!(handle, point_label)
    ctx_cm = handle.ctx_cm; cctx = handle.cctx; obj_z = ctx_cm.obj
    w_z = copy(obj_z.arg2); M_z = obj_z.M
    NCORE_z = cctx.NCORE; ncore_core_z = cctx.ncore_core; Lb_z = cctx.L; nO_z = cctx.nO; ncm_z = cctx.ncm
    n_Z = NCORE_z - ncore_core_z
    lp("  dims cm_meanzc: n_E=", ncore_core_z, " n_Z=", n_Z, " n_C(bin cols)=", ncm_z, " L=", Lb_z, " nO=", nO_z)

    HEE_z = @view cctx.Hfull[1:NCORE_z, 1:NCORE_z]
    tmin, tmean = timeit(() -> fill_core_hessian_upper!((@view HEE_z[1:ncore_core_z, 1:ncore_core_z]), w_z, obj_z, cctx.core_ws;
        backend = cctx.core_hessian_backend, workers = cctx.core_hessian_workers, storage = cctx.core_hessian_storage))
    record!("cm_meanzc", "H_EE(core-only)", tmin, tmean, point_label)

    H_z = _dense_H_or_nothing(obj_z)
    tmin, tmean = timeit(() -> begin
        build_bin_tables_threaded!(cctx, cctx.tls, H_z, w_z; fill_S = false)
        prefix_sum_tables_threaded!(cctx; fill_S = false)
    end)
    record!("cm_meanzc", "bintables(threaded)", tmin, tmean, point_label)

    wctx_z = serial_ctx(cctx.core_ws)
    op_z = cctx.hzz_zc_op
    refresh_zc_targets!(cctx.hzz_zc_ws, op_z, cctx.hzz_zc_layout, cctx.nu_ref[])
    cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op_z, size(w_z, 1))
    tmin, tmean = timeit(() -> refresh_zc_centered!(cctx.hzz_centered, op_z, cctx.hzz_zc_ws, w_z))
    record!("cm_meanzc", "H_ZZ_centering_prep(fill_S=true)", tmin, tmean, point_label)

    n_restr_z = NCORE_z - ncore_core_z
    cctx.zc_cross_scratch = _ensure_zc_cross_scratch!(cctx, wctx_z.W, n_restr_z)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_prep!(cctx.zc_cross_scratch, wctx_z, w_z))
    record!("cm_meanzc", "H_EZ_Snu_prep", tmin, tmean, point_label)

    nx_z = n_restriction(op_z)
    Zview_z = @view cctx.hzz_centered.Zc[:, 1:nx_z]
    HEM_z = Matrix{Float64}(undef, wctx_z.ncolI + 1, nx_z)
    winner_pair_cross_hessian_zc_block!(HEM_z, wctx_z, cctx.zc_cross_scratch, w_z, Zview_z, M_z)
    HEM_z_ref = copy(HEM_z)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block!(HEM_z, wctx_z, cctx.zc_cross_scratch, w_z, Zview_z, M_z))
    record!("cm_meanzc", "H_EZ(=HEM)", tmin, tmean, point_label)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_threaded!(HEM_z, wctx_z, cctx.zc_cross_scratch, w_z, Zview_z, M_z; workers = NT))
    record!("cm_meanzc", "H_EZ(=HEM)_threaded", tmin, tmean, point_label)
    record_gate!("cm_meanzc", point_label, "H_EZ_threaded_vs_serial", "workers=$NT", maximum(abs.(HEM_z .- HEM_z_ref)); tol = 1e-12)

    HMM_z = Matrix{Float64}(undef, nx_z, nx_z)
    tmin, tmean = timeit(() -> zc_restriction_gram!(HMM_z, cctx.hzz_centered, op_z, M_z))
    record!("cm_meanzc", "H_ZZ(=HMM)_reference", tmin, tmean, point_label)
    HMM_z_ref = copy(HMM_z)

    raw_ws_z = build_zc_raw_weighted_workspace(op_z, wctx_z.W)
    refresh_zc_raw_target_vector!(raw_ws_z, cctx.hzz_zc_ws, op_z)
    tmin, tmean = timeit(() -> zc_gram_blas_syrk!(HMM_z, raw_ws_z, w_z, M_z))
    record!("cm_meanzc", "H_ZZ(=HMM)_blas_syrk", tmin, tmean, point_label)
    record_gate!("cm_meanzc", point_label, "H_ZZ_blas_syrk_vs_reference", "K1_1", maximum(abs.(HMM_z .- HMM_z_ref)))
    tmin, tmean = timeit(() -> zc_gram_blas_gemm!(HMM_z, raw_ws_z, w_z, M_z))
    record!("cm_meanzc", "H_ZZ(=HMM)_blas_gemm", tmin, tmean, point_label)
    record_gate!("cm_meanzc", point_label, "H_ZZ_blas_gemm_vs_reference", "K1_1", maximum(abs.(HMM_z .- HMM_z_ref)))
    tmin, tmean = timeit(() -> zc_gram_threaded_packed!(HMM_z, raw_ws_z, w_z, M_z; workers = NT))
    record!("cm_meanzc", "H_ZZ(=HMM)_threaded_packed", tmin, tmean, point_label)
    record_gate!("cm_meanzc", point_label, "H_ZZ_threaded_packed_vs_reference", "K1_1_workers$NT", maximum(abs.(HMM_z .- HMM_z_ref)))

    nz_z = n_restriction(op_z)
    cctx.bin_zc_cross = ensure_bin_zc_cross_scratch!(cctx.bin_zc_cross, cctx.D, Lb_z, nz_z)
    bin_zc_ws_z = cctx.bin_zc_cross
    bin_zc_cross_hessian_fill!(bin_zc_ws_z, cctx.Bidx, cctx.hzz_centered.ZcS)
    ZBinCScum_ref = copy(bin_zc_ws_z.ZBinCScum)
    tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill!(bin_zc_ws_z, cctx.Bidx, cctx.hzz_centered.ZcS))
    record!("cm_meanzc", "H_CZ_prep(fill)", tmin, tmean, point_label)
    tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill_threaded!(bin_zc_ws_z, cctx.Bidx, cctx.hzz_centered.ZcS; workers = NT))
    record!("cm_meanzc", "H_CZ_prep(fill)_threaded", tmin, tmean, point_label)
    record_gate!("cm_meanzc", point_label, "H_CZ_threaded_vs_serial", "workers=$NT", maximum(abs.(bin_zc_ws_z.ZBinCScum .- ZBinCScum_ref)); tol = 1e-12)

    cross_ws_z = _ensure_cm_cross_scratch!(cctx, wctx_z.ncolI, cctx.D, Lb_z)
    winner_pair_cross_hessian_fill!(wctx_z, cross_ws_z, obj_z, cctx.Bidx)
    QCScum_ref = copy(cross_ws_z.QCScum)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill!(wctx_z, cross_ws_z, obj_z, cctx.Bidx))
    record!("cm_meanzc", "H_EC_prep(crossprep)", tmin, tmean, point_label)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill_threaded!(wctx_z, cross_ws_z, obj_z, cctx.Bidx; workers = NT))
    record!("cm_meanzc", "H_EC_prep(crossprep,threaded)", tmin, tmean, point_label)
    record_gate!("cm_meanzc", point_label, "H_EC_threaded_vs_serial", "workers=$NT", maximum(abs.(cross_ws_z.QCScum .- QCScum_ref)); tol = 1e-12)

    Hraw_EC_z = cctx.Hraw_EC
    Hraw_EC_core_z = @view Hraw_EC_z[1:ncore_core_z, :]
    Hraw_EC_zpart_z = @view Hraw_EC_z[ncore_core_z+1:NCORE_z, :]
    tmin, tmean = timeit(() -> begin
        for l in 1:Lb_z
            winner_pair_cross_hessian_cm_block!(Hraw_EC_core_z, wctx_z, cross_ws_z, l, cctx.origins, cctx.refIndex1, M_z)
            bin_zc_cross_hessian_block!(Hraw_EC_zpart_z, bin_zc_ws_z, l, cctx.origins, cctx.refIndex1, M_z)
        end
    end)
    record!("cm_meanzc", "H_EC+H_CZ_asm", tmin, tmean, point_label)

    # complete real Hessian callback + full inner solve wall time, direct from the driver's own
    # printed n_hess/wall numbers -- recorded separately, see the *_driver_summary.csv this script
    # also writes.
end

# ================================================================================================
# Orchestration: 3 points per family (calibration / non_calibration / solver_trajectory), each via
# a SEPARATE short/medium real driver call so no low-level helper is ever invoked directly.
# ================================================================================================
const ONLY_FAMILY = get(ENV, "ONLY_FAMILY", "")
const CALIB_BUDGET = parse(Float64, get(ENV, "CALIB_BUDGET", "12.0"))
const NONCALIB_BUDGET = parse(Float64, get(ENV, "NONCALIB_BUDGET", "12.0"))
const TRAJ_BUDGET = parse(Float64, get(ENV, "TRAJ_BUDGET", "60.0"))

driver_summary = NamedTuple[]
function record_driver!(family, point, run)
    r = run.result
    push!(driver_summary, (family = family, point = point, nthreads = NT, wall_s = run.wall,
        knitro_status = string(r.knitro_status), n_eval = r.n_eval, n_grad = r.n_grad,
        kappa = hasproperty(r, :kappa) ? string(r.kappa) : "n/a"))
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

    for (w0_, budget_, label_) in [(w0_calib, CALIB_BUDGET, "calibration"),
                                    (w0_noncalib, NONCALIB_BUDGET, "non_calibration"),
                                    (w0_calib, TRAJ_BUDGET, "solver_trajectory")]
        try
            r = run_originzc_driver(w0_, budget_, label_)
            record_driver!("origin_zc", label_, r)
            profile_originzc_point!(r.handle, label_)
        catch e
            println("\n!!! origin_zc/$label_ FAILED -- ", sprint(showerror, e))
            flush(stdout)
        end
    end
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

    for (w0_, budget_, label_) in [(w0c_calib, CALIB_BUDGET, "calibration"),
                                    (w0c_noncalib, NONCALIB_BUDGET, "non_calibration"),
                                    (w0c_calib, TRAJ_BUDGET, "solver_trajectory")]
        try
            c = run_cmzc_driver(w0_, budget_, label_)
            record_driver!("cm_meanzc", label_, c)
            plabel = c.unsolved ? label_ * "_UNSOLVED_TIMING_ONLY" : label_
            profile_cmzc_point!(c.handle, plabel)
        catch e
            println("\n!!! cm_meanzc/$label_ FAILED -- ", sprint(showerror, e))
            flush(stdout)
        end
    end
end

famtag = isempty(ONLY_FAMILY) ? "all" : ONLY_FAMILY
outpath = joinpath(D4X, "..", "..", "results", "cmzc_originzc_subblock_profile_t$(NT)_$(famtag)_2026-07-28.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,block,nthreads,t_min_s,t_mean_s,point")
    for r in rows
        println(io, "$(r.family),$(r.block),$(r.nthreads),$(r.t_min_s),$(r.t_mean_s),$(r.point)")
    end
end
lp("Wrote ", outpath)

gatepath = joinpath(D4X, "..", "..", "results", "cmzc_originzc_gates_t$(NT)_$(famtag)_2026-07-28.csv")
open(gatepath, "w") do io
    println(io, "family,point,check,key,maxdiff,tol,pass")
    for r in gate_rows
        println(io, "$(r.family),$(r.point),$(r.check),$(r.key),$(r.maxdiff),$(r.tol),$(r.pass)")
    end
end
lp("Wrote ", gatepath)

driverpath = joinpath(D4X, "..", "..", "results", "cmzc_originzc_driver_summary_t$(NT)_$(famtag)_2026-07-28.csv")
open(driverpath, "w") do io
    println(io, "family,point,nthreads,wall_s,knitro_status,n_eval,n_grad,kappa")
    for r in driver_summary
        println(io, "$(r.family),$(r.point),$(r.nthreads),$(r.wall_s),$(r.knitro_status),$(r.n_eval),$(r.n_grad),$(r.kappa)")
    end
end
lp("Wrote ", driverpath)
lp("DONE.")
