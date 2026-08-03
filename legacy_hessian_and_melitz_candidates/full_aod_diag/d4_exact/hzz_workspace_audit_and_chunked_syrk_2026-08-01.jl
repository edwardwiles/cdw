# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Sections 9-10: H_ZZ full-workspace cost
# audit (weighting/SYRK/correction phase split, sqrt(S) vs S convention, memory footprint) and
# row-chunked SYRK benchmark (chunk sizes 2048/4096/8192/16384/32768, BLAS threads 4/6/8/10) vs the
# existing full-workspace :blas_syrk, at W=100,000 (and W=500,000 where noted).
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
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "hez_drawmajor_candidate_2026-08-01.jl", "hzz_chunked_syrk_candidate_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
lp(xs...) = (println(xs...); flush(stdout))

const NT = Threads.nthreads()
lp("Threads.nthreads() = ", NT, " (BLAS threads swept; Julia threads not used by H_ZZ backends)")

function build_state(; W::Int)
    ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    K_mean, K_pair, L = 3, 3, 50
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
    cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    obj_cm = aug.obj_cm
    n = cctx.NCORE + cctx.ncm
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    θ_econ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
    θ_ext_calib = vcat(θ_econ_calib, ones(K_mean))
    Kbuf = Vector{Float64}(undef, W)
    Gbuf = Matrix{Float64}(undef, W, n + 64)
    obj_cm.moments!(Kbuf, Gbuf, θ_ext_calib, ctx.U, obj_cm)
    cctx.nu_ref[] = ones(K_mean)
    x_fake = 0.1 .* randn(MersenneTwister(20260801), n)
    tls = build_thread_local_scratch(cctx)
    h_scratch = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(h_scratch, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
    return ctx, cctx, obj_cm, n
end

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
function record!(W, backend, chunk_size, blas_threads, tmin, tmean, workspace_mb; maxdiff = NaN)
    push!(rows, (W = W, backend = backend, chunk_size = chunk_size, blas_threads = blas_threads,
        t_min_s = tmin, t_mean_s = tmean, workspace_mb = workspace_mb, maxdiff = maxdiff))
    @printf("  [W=%-7d|%-18s] chunk=%-8s blas_t=%-3d min=%.4fs mean=%.4fs workspace=%.1fMB maxdiff=%.3e\n",
        W, backend, chunk_size == 0 ? "full" : string(chunk_size), blas_threads, tmin, tmean, workspace_mb, maxdiff)
    flush(stdout)
end

for W in [100_000, 500_000]
    lp("\n", "="^100, "\n=== W=$W ===")
    ctx, cctx, obj_cm, n = build_state(W = W)
    nz = n_restriction(cctx.hzz_zc_op)
    S = copy(obj_cm.arg2); M = obj_cm.M
    raw_ws = build_zc_raw_weighted_workspace(cctx.hzz_zc_op, W)
    refresh_zc_raw_target_vector!(raw_ws, cctx.hzz_zc_ws, cctx.hzz_zc_op)
    HZZ_ref = Matrix{Float64}(undef, nz, nz)

    # Reference (materialize-centered-Zc) anchor, for maxdiff comparisons.
    cs = ensure_zc_centered_scratch!(nothing, cctx.hzz_zc_op, W)
    refresh_zc_centered!(cs, cctx.hzz_zc_op, cctx.hzz_zc_ws, S; fill_S = true)
    zc_restriction_gram!(HZZ_ref, cs, cctx.hzz_zc_op, M)

    full_workspace_mb = 2 * (W * nz * 8) / 1e6   # Phi + RW, both (W,nz) Float64
    lp("nZ=$nz, full Phi+RW workspace = $(round(full_workspace_mb, digits=1)) MB (2x $(round(W*nz*8/1e6, digits=1))MB each)")

    for bt in [1, 4, 6, 8, 10]
        BLAS.set_num_threads(bt)
        HZZ = Matrix{Float64}(undef, nz, nz)
        tmin, tmean = timeit(() -> zc_gram_blas_syrk!(HZZ, raw_ws, S, M))
        record!(W, "full_blas_syrk", 0, bt, tmin, tmean, full_workspace_mb; maxdiff = maximum(abs.(HZZ .- HZZ_ref)))
    end

    for chunk_size in [2048, 4096, 8192, 16384, 32768], bt in [4, 6, 8, 10]
        BLAS.set_num_threads(bt)
        cws = ZCChunkedSyrkWorkspace(W, nz, chunk_size)
        chunk_workspace_mb = (chunk_size * nz * 8) / 1e6
        HZZ = Matrix{Float64}(undef, nz, nz)
        tmin, tmean = timeit(() -> zc_gram_blas_syrk_chunked!(HZZ, raw_ws, cws, S, M; chunk_size = chunk_size))
        record!(W, "chunked_syrk", chunk_size, bt, tmin, tmean, chunk_workspace_mb; maxdiff = maximum(abs.(HZZ .- HZZ_ref)))
    end
    BLAS.set_num_threads(1)
end

outpath = joinpath(D4X, "..", "..", "docs", "HZZ_FULL_VS_CHUNKED_SYRK_BENCHMARK_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "W,backend,chunk_size,blas_threads,t_min_s,t_mean_s,workspace_mb,maxdiff")
    for r in rows
        println(io, "$(r.W),$(r.backend),$(r.chunk_size),$(r.blas_threads),$(r.t_min_s),$(r.t_mean_s),$(r.workspace_mb),$(r.maxdiff)")
    end
end
lp("Wrote ", outpath)

# ---- workspace cost audit CSV (Section 9) ----
audit_rows = NamedTuple[]
for W in [100_000, 500_000]
    nz = 630
    push!(audit_rows, (W = W, nx = nz,
        phi_mb = round(W*nz*8/1e6, digits=1), rw_mb = round(W*nz*8/1e6, digits=1),
        full_workspace_mb = round(2*W*nz*8/1e6, digits=1),
        convention = "blas_syrk forms sqrt(S).*Phi (RW), then BLAS.syrk!('U','T',...) so RW'*RW = Phi'*diag(S)*Phi; blas_gemm/threaded_packed form S.*Phi directly and use full mul!(Phi',RW)"))
end
audit_path = joinpath(D4X, "..", "..", "docs", "HZZ_FULL_WORKSPACE_COST_AUDIT_2026-08-01.csv")
open(audit_path, "w") do io
    println(io, "W,nx,phi_mb,rw_mb,full_workspace_mb,convention")
    for r in audit_rows
        println(io, "$(r.W),$(r.nx),$(r.phi_mb),$(r.rw_mb),$(r.full_workspace_mb),\"$(r.convention)\"")
    end
end
lp("Wrote ", audit_path)
lp("DONE")
