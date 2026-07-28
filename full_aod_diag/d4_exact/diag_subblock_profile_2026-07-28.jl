# optimize/structured-cross-hessian-ZC-CM-2026-07-28, Step 1: re-profile the CURRENT production
# head's cross-Hessian sub-blocks at real D=20/Ddest=19/W=100,000, directly calling the already-
# validated shared kernels (winner_pair_cross_hessian.jl / zc_restriction_operator.jl) against the
# warm state left behind by one real production solve per family -- not a re-derived clone of the
# KNITRO callback body (avoids a second, potentially-drifting copy of the branch logic).
#
# Each family's block is wrapped independently (try/catch) so one family's solve failure does not
# block profiling the others -- resilience added after flexible_cm's calibration solve was found to
# raise a KNITRO KN_RC_CALLBACK_ERR (-500) on this exact host/build for reasons under separate
# investigation (see STRUCTURED_CROSS_HESSIAN_PROVENANCE_2026-07-28.md addendum), independent of
# this task's own cross-Hessian kernel work.
#
# Usage: julia --project=. -t 1 diag_subblock_profile_2026-07-28.jl  (and again with -t 20)
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl", "instrumentation.jl",
          "oracle_fast.jl", "gravity_elimination.jl", "structured_moment_build.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian_threaded.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, SHA

const NT = Threads.nthreads()
println("Threads.nthreads() = $NT"); flush(stdout)

const PROFILE_W = parse(Int, get(ENV, "PROFILE_W", "100000"))
println("Building real D=20 context (W=$PROFILE_W, delta=1.0, destination_sample=:exclude_row)..."); flush(stdout)
ctx = d20_real_setup(W = PROFILE_W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
println("checksum_U = $(bytes2hex(sha256(reinterpret(UInt8, vec(ctx.U)))))")
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
flush(stdout)

const REPS = 8
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

rows = NamedTuple[]
function record!(family, block, tmin, tmean)
    push!(rows, (family = family, block = block, nthreads = NT, t_min_s = tmin, t_mean_s = tmean))
    @printf("  [%-15s] %-28s  min=%.4fs  mean=%.4fs\n", family, block, tmin, tmean)
    flush(stdout)
end

function run_flexible_cm()
    println("\n=== flexible_cm ==="); flush(stdout)
    pcx = build_cm_production_context(ctx, CS; L = 50, contrasts = :anchored)
    cctx = pcx.cctx; obj = pcx.ctx_cm.obj
    archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
    obj.ddPsi!(obj.arg2, obj.arg0)
    w = obj.arg2
    H = _dense_H_or_nothing(obj)
    M = obj.M
    NCORE = cctx.NCORE; Lb = cctx.L; nO = cctx.nO; origins = cctx.origins; refIndex1 = cctx.refIndex1
    println("dims: NCORE=$NCORE ncore_core=$(cctx.ncore_core) L=$Lb nO=$nO M=$M")

    HEE = @view cctx.Hfull[1:NCORE, 1:NCORE]
    tmin, tmean = timeit(() -> fill_core_hessian_upper!(HEE, w, obj, cctx.core_ws;
        backend = cctx.core_hessian_backend, workers = cctx.core_hessian_workers, storage = cctx.core_hessian_storage))
    record!("flexible_cm", "H_EE", tmin, tmean)

    tmin, tmean = timeit(() -> begin
        build_bin_tables_threaded!(cctx, cctx.tls, H, w; fill_S = false)
        prefix_sum_tables_threaded!(cctx; fill_S = false)
    end)
    record!("flexible_cm", "bintables", tmin, tmean)

    wctx = serial_ctx(cctx.core_ws)
    cross_ws = _ensure_cm_cross_scratch!(cctx, wctx.ncolI, D, Lb)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill!(wctx, cross_ws, obj, cctx.Bidx))
    record!("flexible_cm", "H_EC_prep(crossprep)", tmin, tmean)

    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill_threaded!(wctx, cross_ws, obj, cctx.Bidx; workers = NT))
    record!("flexible_cm", "H_EC_prep(crossprep,threaded)", tmin, tmean)

    Hraw_EC = cctx.Hraw_EC
    tmin, tmean = timeit(() -> begin
        for l in 1:Lb
            winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, cross_ws, l, origins, refIndex1, M)
        end
    end)
    record!("flexible_cm", "H_EC_asm", tmin, tmean)
end

function run_common_frechet()
    println("\n=== common_frechet ==="); flush(stdout)
    pcx_f = build_cm_frechet_production_context(ctx, CS; L = 50, contrasts = :anchored, cm_hessian_backend = :structured)
    cctx_f = pcx_f.cctx; obj_f = pcx_f.ctx_cm.obj
    archC_frechet_base_state(x_free_calib, pcx_f.ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)
    obj_f.ddPsi!(obj_f.arg2, obj_f.arg0)
    w_f = obj_f.arg2
    H_f = _dense_H_or_nothing(obj_f)
    M_f = obj_f.M
    NCORE_f = cctx_f.NCORE; Lb_f = cctx_f.L

    HEE_f = @view cctx_f.Hfull[1:NCORE_f, 1:NCORE_f]
    tmin, tmean = timeit(() -> fill_core_hessian_upper!(HEE_f, w_f, obj_f, cctx_f.core_ws;
        backend = cctx_f.core_hessian_backend, workers = cctx_f.core_hessian_workers, storage = cctx_f.core_hessian_storage))
    record!("common_frechet", "H_EE", tmin, tmean)

    tmin, tmean = timeit(() -> begin
        build_bin_tables_threaded!(cctx_f, cctx_f.tls, H_f, w_f; fill_S = false)
        prefix_sum_tables_threaded!(cctx_f; fill_S = false)
    end)
    record!("common_frechet", "bintables", tmin, tmean)

    wctx_f = serial_ctx(cctx_f.core_ws)
    cross_ws_f = _ensure_cm_cross_scratch!(cctx_f, wctx_f.ncolI, D, Lb_f)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill!(wctx_f, cross_ws_f, obj_f, cctx_f.Bidx))
    record!("common_frechet", "H_EC_prep(crossprep)", tmin, tmean)

    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill_threaded!(wctx_f, cross_ws_f, obj_f, cctx_f.Bidx; workers = NT))
    record!("common_frechet", "H_EC_prep(crossprep,threaded)", tmin, tmean)
end

function run_cm_meanzc()
    println("\n=== cm_meanzc (CM+ZC) ==="); flush(stdout)
    νvec0 = [1.0]
    pcx_z = build_cm_meanzc_production_context(ctx, CS; L = 50, K_mean = 1, K_pair = 0, contrasts = :anchored)
    cctx_z = pcx_z.cctx; obj_z = pcx_z.ctx_cm.obj
    cm_meanzc_production_value_verified(x_free_calib, νvec0, pcx_z)
    obj_z.ddPsi!(obj_z.arg2, obj_z.arg0)
    w_z = obj_z.arg2
    H_z = _dense_H_or_nothing(obj_z)
    M_z = obj_z.M
    NCORE_z = cctx_z.NCORE; ncore_core_z = cctx_z.ncore_core; Lb_z = cctx_z.L; nO_z = cctx_z.nO
    println("dims: NCORE=$NCORE_z ncore_core=$ncore_core_z (n_Z=$(NCORE_z-ncore_core_z)) L=$Lb_z nO=$nO_z")

    HEE_z = @view cctx_z.Hfull[1:NCORE_z, 1:NCORE_z]
    tmin, tmean = timeit(() -> fill_core_hessian_upper!((@view HEE_z[1:ncore_core_z, 1:ncore_core_z]), w_z, obj_z, cctx_z.core_ws;
        backend = cctx_z.core_hessian_backend, workers = cctx_z.core_hessian_workers, storage = cctx_z.core_hessian_storage))
    record!("cm_meanzc", "H_EE(core-only)", tmin, tmean)

    tmin, tmean = timeit(() -> begin
        build_bin_tables_threaded!(cctx_z, cctx_z.tls, H_z, w_z; fill_S = false)
        prefix_sum_tables_threaded!(cctx_z; fill_S = false)
    end)
    record!("cm_meanzc", "bintables", tmin, tmean)

    wctx_z = serial_ctx(cctx_z.core_ws)
    op_z = cctx_z.hzz_zc_op
    refresh_zc_targets!(cctx_z.hzz_zc_ws, op_z, cctx_z.hzz_zc_layout, cctx_z.nu_ref[])
    cctx_z.hzz_centered = ensure_zc_centered_scratch!(cctx_z.hzz_centered, op_z, size(w_z, 1))
    tmin, tmean = timeit(() -> refresh_zc_centered!(cctx_z.hzz_centered, op_z, cctx_z.hzz_zc_ws, w_z))
    record!("cm_meanzc", "H_ZZ_centering_prep(fill_S=true)", tmin, tmean)

    n_restr_z = NCORE_z - ncore_core_z
    cctx_z.zc_cross_scratch = _ensure_zc_cross_scratch!(cctx_z, wctx_z.W, n_restr_z)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_prep!(cctx_z.zc_cross_scratch, wctx_z, w_z))
    record!("cm_meanzc", "H_EZ_Snu_prep", tmin, tmean)

    nx_z = n_restriction(op_z)
    Zview_z = @view cctx_z.hzz_centered.Zc[:, 1:nx_z]
    HEM_z = Matrix{Float64}(undef, wctx_z.ncolI + 1, nx_z)
    winner_pair_cross_hessian_zc_block!(HEM_z, wctx_z, cctx_z.zc_cross_scratch, w_z, Zview_z, M_z)
    HEM_z_ref = copy(HEM_z)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block!(HEM_z, wctx_z, cctx_z.zc_cross_scratch, w_z, Zview_z, M_z))
    record!("cm_meanzc", "H_EZ(=HEM)", tmin, tmean)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_threaded!(HEM_z, wctx_z, cctx_z.zc_cross_scratch, w_z, Zview_z, M_z; workers = NT))
    record!("cm_meanzc", "H_EZ(=HEM,threaded)", tmin, tmean)
    println("  [CORRECTNESS] cm_meanzc H_EZ threaded vs serial maxdiff = $(maximum(abs.(HEM_z .- HEM_z_ref)))")

    HMM_z = Matrix{Float64}(undef, nx_z, nx_z)
    tmin, tmean = timeit(() -> zc_restriction_gram!(HMM_z, cctx_z.hzz_centered, op_z, M_z))
    record!("cm_meanzc", "H_ZZ(=HMM)_gram_reference", tmin, tmean)
    HMM_z_ref = copy(HMM_z)

    raw_ws_z = build_zc_raw_weighted_workspace(op_z, wctx_z.W)
    refresh_zc_raw_target_vector!(raw_ws_z, cctx_z.hzz_zc_ws, op_z)
    tmin, tmean = timeit(() -> zc_gram_blas_syrk!(HMM_z, raw_ws_z, w_z, M_z))
    record!("cm_meanzc", "H_ZZ(=HMM)_blas_syrk", tmin, tmean)
    println("  [CORRECTNESS] cm_meanzc H_ZZ blas_syrk vs reference maxdiff = $(maximum(abs.(HMM_z .- HMM_z_ref)))")
    tmin, tmean = timeit(() -> zc_gram_blas_gemm!(HMM_z, raw_ws_z, w_z, M_z))
    record!("cm_meanzc", "H_ZZ(=HMM)_blas_gemm", tmin, tmean)
    println("  [CORRECTNESS] cm_meanzc H_ZZ blas_gemm vs reference maxdiff = $(maximum(abs.(HMM_z .- HMM_z_ref)))")
    tmin, tmean = timeit(() -> zc_gram_threaded_packed!(HMM_z, raw_ws_z, w_z, M_z; workers = NT))
    record!("cm_meanzc", "H_ZZ(=HMM)_threaded_packed", tmin, tmean)
    println("  [CORRECTNESS] cm_meanzc H_ZZ threaded_packed vs reference maxdiff = $(maximum(abs.(HMM_z .- HMM_z_ref)))")

    nz_z = n_restriction(op_z)
    cctx_z.bin_zc_cross = ensure_bin_zc_cross_scratch!(cctx_z.bin_zc_cross, D, Lb_z, nz_z)
    bin_zc_ws_z = cctx_z.bin_zc_cross
    bin_zc_cross_hessian_fill!(bin_zc_ws_z, cctx_z.Bidx, cctx_z.hzz_centered.ZcS)
    ZBinCScum_ref = copy(bin_zc_ws_z.ZBinCScum)
    tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill!(bin_zc_ws_z, cctx_z.Bidx, cctx_z.hzz_centered.ZcS))
    record!("cm_meanzc", "H_CZ_prep(fill)", tmin, tmean)
    tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill_threaded!(bin_zc_ws_z, cctx_z.Bidx, cctx_z.hzz_centered.ZcS; workers = NT))
    record!("cm_meanzc", "H_CZ_prep(fill,threaded)", tmin, tmean)
    println("  [CORRECTNESS] cm_meanzc H_CZ threaded vs serial maxdiff = $(maximum(abs.(bin_zc_ws_z.ZBinCScum .- ZBinCScum_ref)))")

    cross_ws_z = _ensure_cm_cross_scratch!(cctx_z, wctx_z.ncolI, D, Lb_z)
    winner_pair_cross_hessian_fill!(wctx_z, cross_ws_z, obj_z, cctx_z.Bidx)
    QCScum_ref = copy(cross_ws_z.QCScum)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill!(wctx_z, cross_ws_z, obj_z, cctx_z.Bidx))
    record!("cm_meanzc", "H_EC_prep(crossprep, core-cols)", tmin, tmean)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_fill_threaded!(wctx_z, cross_ws_z, obj_z, cctx_z.Bidx; workers = NT))
    record!("cm_meanzc", "H_EC_prep(crossprep, core-cols,threaded)", tmin, tmean)
    println("  [CORRECTNESS] cm_meanzc H_EC threaded vs serial maxdiff = $(maximum(abs.(cross_ws_z.QCScum .- QCScum_ref)))")

    Hraw_EC_z = cctx_z.Hraw_EC
    Hraw_EC_core_z = @view Hraw_EC_z[1:ncore_core_z, :]
    Hraw_EC_zpart_z = @view Hraw_EC_z[ncore_core_z+1:NCORE_z, :]
    tmin, tmean = timeit(() -> begin
        for l in 1:Lb_z
            winner_pair_cross_hessian_cm_block!(Hraw_EC_core_z, wctx_z, cross_ws_z, l, origins_of(cctx_z), refIndex1_of(cctx_z), M_z)
            bin_zc_cross_hessian_block!(Hraw_EC_zpart_z, bin_zc_ws_z, l, origins_of(cctx_z), refIndex1_of(cctx_z), M_z)
        end
    end)
    record!("cm_meanzc", "H_EC+H_CZ_asm", tmin, tmean)
end
origins_of(cctx) = cctx.origins
refIndex1_of(cctx) = cctx.refIndex1

function run_origin_zc()
    println("\n=== origin_zc (ZC-only) ==="); flush(stdout)
    layout_o = OriginByPowerLayout(D, 1, 0)
    nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)
    νfull0 = nu0_origin(1, D)
    pcx_o = build_originzc_production_context(ctx, CS, layout_o)
    octx = pcx_o.octx
    obj_o = pcx_o.ctx_cm.obj
    cm_originzc_production_value_verified(x_free_calib, νfull0, pcx_o)
    obj_o.ddPsi!(obj_o.arg2, obj_o.arg0)
    w_o = obj_o.arg2
    M_o = obj_o.M
    NCORE_o = octx.NCORE; n_eta_o = octx.n_eta
    println("dims: NCORE=$NCORE_o n_eta(Z)=$n_eta_o")

    HEE_o = Matrix{Float64}(undef, NCORE_o, NCORE_o)
    tmin, tmean = timeit(() -> fill_core_hessian_upper!(HEE_o, w_o, obj_o, octx.core_ws;
        backend = octx.core_hessian_backend, workers = octx.core_hessian_workers, storage = octx.core_hessian_storage))
    record!("origin_zc", "H_EE", tmin, tmean)

    wctx_o = serial_ctx(octx.core_ws)
    op_o = octx.hzz_zc_op
    refresh_zc_targets!(octx.hzz_zc_ws, op_o, octx.hzz_zc_layout, octx.nu_ref[])
    octx.hzz_centered = ensure_zc_centered_scratch!(octx.hzz_centered, op_o, size(w_o, 1))
    tmin, tmean = timeit(() -> refresh_zc_centered!(octx.hzz_centered, op_o, octx.hzz_zc_ws, w_o))
    record!("origin_zc", "H_ZZ_centering_prep(fill_S=true)", tmin, tmean)

    octx.zc_cross_scratch = _ensure_originzc_zc_cross_scratch!(octx, wctx_o.W, n_eta_o)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_prep!(octx.zc_cross_scratch, wctx_o, w_o))
    record!("origin_zc", "H_EZ_Snu_prep(=H_ER prep)", tmin, tmean)

    nx_o = n_restriction(op_o)
    Zview_o = @view octx.hzz_centered.Zc[:, 1:nx_o]
    HER_o = Matrix{Float64}(undef, wctx_o.ncolI + 1, nx_o)
    winner_pair_cross_hessian_zc_block!(HER_o, wctx_o, octx.zc_cross_scratch, w_o, Zview_o, M_o)
    HER_o_ref = copy(HER_o)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block!(HER_o, wctx_o, octx.zc_cross_scratch, w_o, Zview_o, M_o))
    record!("origin_zc", "H_EZ(=HER)", tmin, tmean)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_threaded!(HER_o, wctx_o, octx.zc_cross_scratch, w_o, Zview_o, M_o; workers = NT))
    record!("origin_zc", "H_EZ(=HER,threaded)", tmin, tmean)
    println("  [CORRECTNESS] origin_zc H_EZ threaded vs serial maxdiff = $(maximum(abs.(HER_o .- HER_o_ref)))")

    HRR_o = Matrix{Float64}(undef, nx_o, nx_o)
    tmin, tmean = timeit(() -> zc_restriction_gram!(HRR_o, octx.hzz_centered, op_o, M_o))
    record!("origin_zc", "H_ZZ(=HRR)_gram_reference", tmin, tmean)
    HRR_o_ref = copy(HRR_o)

    raw_ws_o = build_zc_raw_weighted_workspace(op_o, wctx_o.W)
    refresh_zc_raw_target_vector!(raw_ws_o, octx.hzz_zc_ws, op_o)
    tmin, tmean = timeit(() -> zc_gram_blas_syrk!(HRR_o, raw_ws_o, w_o, M_o))
    record!("origin_zc", "H_ZZ(=HRR)_blas_syrk", tmin, tmean)
    println("  [CORRECTNESS] origin_zc H_ZZ blas_syrk vs reference maxdiff = $(maximum(abs.(HRR_o .- HRR_o_ref)))")
    tmin, tmean = timeit(() -> zc_gram_blas_gemm!(HRR_o, raw_ws_o, w_o, M_o))
    record!("origin_zc", "H_ZZ(=HRR)_blas_gemm", tmin, tmean)
    tmin, tmean = timeit(() -> zc_gram_threaded_packed!(HRR_o, raw_ws_o, w_o, M_o; workers = NT))
    record!("origin_zc", "H_ZZ(=HRR)_threaded_packed", tmin, tmean)
end

# Pre-existing KNITRO-Julia-wrapper flakiness (KN_RC_CALLBACK_ERR / MethodError(convert,(Task,0.0))
# in KNITRO.jl's own _try_catch_handler, C_wrapper.jl:287), confirmed reproducible with a completely
# unmodified reference script (no_dense_g_full_family_audit_2026-07-27.jl) and even for the
# "unrestricted" family (touches none of this task's files) -- unrelated to this task's own kernel
# work. Empirically, running MULTIPLE families' KNITRO solves sequentially in one Julia process
# raises the chance of hitting it; ONLY_FAMILY restricts a given process to a single family so each
# gets a clean KNITRO/Julia environment (run this script once per family, `ONLY_FAMILY=<name>`).
const ONLY_FAMILY = get(ENV, "ONLY_FAMILY", "")
const ALL_FAMILY_FNS = [("flexible_cm", run_flexible_cm), ("common_frechet", run_common_frechet),
                         ("cm_meanzc", run_cm_meanzc), ("origin_zc", run_origin_zc)]
for (label, fn) in (isempty(ONLY_FAMILY) ? ALL_FAMILY_FNS : filter(p -> p[1] == ONLY_FAMILY, ALL_FAMILY_FNS))
    try
        fn()
    catch e
        println("\n!!! FAMILY_FAILED: $label -- $(sprint(showerror, e))")
        flush(stdout)
    end
end

famtag = isempty(ONLY_FAMILY) ? "all" : ONLY_FAMILY
outpath = joinpath(D4X, "..", "..", "results", "cross_hessian_subblock_profile_t$(NT)_$(famtag)_2026-07-28.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,block,nthreads,t_min_s,t_mean_s")
    for r in rows
        println(io, "$(r.family),$(r.block),$(r.nthreads),$(r.t_min_s),$(r.t_mean_s)")
    end
end
println("\nWrote $outpath")
