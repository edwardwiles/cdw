# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Sections 4/8: direct true-cold SINGLE
# inner-solve A/B. Each invocation of this script is meant to run in its OWN fresh Julia process
# (no shared state with any other arm/rep) -- the bash driver `zc_direct_true_cold_inner_ab_run_2026-08-01.sh`
# launches one fresh `julia` process per (family, arm, rep) combination. Within THIS process:
# fresh context, fresh obj (obj.x starts at KNITRO's own uninitialized default -- never touched
# before the ONE solve below), exact-point cache and dual bank never built at all (this harness
# never constructs them), same cold dual/outer point/solver settings across every arm.
#
# ENV vars: ZC_FAMILY (cm_meanzc|origin_zc), ZC_ARM (reference|hzz_only|hcz_only|hez_only|all_optimized),
# ZC_W (default 100000), ZC_OUT (output line file, appended).
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_config.jl",
          "cm_originzc_moments.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_originzc_production.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl",
          "hez_drawmajor_v2_candidate_2026-08-01.jl", "hzz_chunked_syrk_candidate_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random, Statistics, Dates

lp(xs...) = (println(xs...); flush(stdout))

FAMILY = Symbol(get(ENV, "ZC_FAMILY", "cm_meanzc"))
ARM = Symbol(get(ENV, "ZC_ARM", "reference"))
W = parse(Int, get(ENV, "ZC_W", "100000"))
REP = get(ENV, "ZC_REP", "1")
# Concurrent-CSV-append race under parallelism (known repo pitfall, see feedback memory of the same
# name): each fresh process writes its OWN uniquely-named per-point file, never appends to a shared
# CSV -- the bash driver assembles the final CSV afterward from these per-point files.
OUTDIR = get(ENV, "ZC_OUTDIR", joinpath(D4X, "..", "..", "results", "zc_direct_true_cold_inner_ab_2026-08-01"))
mkpath(OUTDIR)

lp("=== direct true-cold inner solve: family=$FAMILY arm=$ARM W=$W pid=$(getpid()) ===")

# Backend assignment per arm -- (hzz, hcz, hez); hcz is inapplicable to origin_zc (H_CZ has no
# origin_zc analog -- correctly excluded per task brief Section 5).
ARM_BACKENDS = Dict(
    :reference      => (:reference, :draw_chunk_thread_local, :winner_bin),
    :hzz_only       => (:blas_syrk, :draw_chunk_thread_local, :winner_bin),
    :hcz_only       => (:reference, :draw_chunk_reordered,    :winner_bin),
    :hez_only       => (:reference, :draw_chunk_thread_local, :drawmajor_v2),
    :all_optimized  => (:blas_syrk, :draw_chunk_reordered,    :drawmajor_v2),
)
haskey(ARM_BACKENDS, ARM) || error("unknown ZC_ARM=$ARM")
hzz_be, hcz_be, hez_be = ARM_BACKENDS[ARM]

t_ctx0 = time()
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
t_ctx = time() - t_ctx0
K_mean, K_pair, L = 3, 3, 50
x_free_calib = ctx.θ0_up[ctx.free_idx]

n_fg = -1; n_hess = -1; nStatus = -9999; t_solve = NaN; delta_dual = NaN

if FAMILY === :cm_meanzc
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
    cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    cctx.zc_gram_backend = hzz_be
    cctx.hcz_prep_backend = hcz_be
    cctx.zc_ez_backend = hez_be
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    νvec = Float64.(factorial.(1:K_mean))   # NOT ones(K_mean) -- confirmed live 2026-08-01 that ones(3) is
    # infeasible (nStatus=-300) for cm_meanzc K_mean=3 direct construction; diag_hzz_backend_benchmark_k3_2026-08-01.jl's
    # own working w0 uses exactly this factorial-based nu0 (see ORIGIN_ZC_HARNESS_ROOT_CAUSE_2026-08-01.md
    # for the analogous origin-ZC finding this mirrors).
    t0 = time()
    try
        base = archC_meanzc_base_state(x_free_calib, νvec, ctx_cm, cctx)
        global nStatus = base.inner_status
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        global nStatus = -300   # convention: uncaught expected-failure exception -> record as infeasible
    end
    global t_solve = time() - t0
    global n_fg = CS.INNER_SOLVE_COUNT[]
elseif FAMILY === :origin_zc
    layout = OriginByPowerLayout(ctx.D, K_mean, K_pair)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for k in 1:K_mean, o in 1:ctx.D
        nu0[target_index(layout, o, k)] = mean(@view (ctx.U .^ k)[:, o])
    end
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, moment_representation = :operator)
    pcx.octx.zc_gram_backend = hzz_be
    pcx.octx.zc_ez_backend = hez_be
    t0 = time()
    try
        base = archOZ_base_state(x_free_calib, nu0, pcx.ctx_cm)
        global nStatus = base.inner_status
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        global nStatus = -300
    end
    global t_solve = time() - t0
else
    error("unknown ZC_FAMILY=$FAMILY")
end

lp(@sprintf("RESULT family=%s arm=%s W=%d pid=%d ctx_build_s=%.3f solve_s=%.3f nStatus=%d", FAMILY, ARM, W, getpid(), t_ctx, t_solve, nStatus))

outfile = joinpath(OUTDIR, "$(FAMILY)_$(ARM)_rep$(REP)_pid$(getpid()).csv")
open(outfile, "w") do io
    println(io, "family,arm,rep,W,hzz_backend,hcz_backend,hez_backend,pid,ctx_build_s,solve_s,nStatus,timestamp")
    println(io, "$FAMILY,$ARM,$REP,$W,$hzz_be,$hcz_be,$hez_be,$(getpid()),$t_ctx,$t_solve,$nStatus,$(now())")
end
lp("Wrote per-point file ", outfile)
