# D20 real-data profiling for the pairwise-quantile-independence restriction's Hessian callback,
# broken into sub-block timings per the plan's Section 11 request. Takes W as ARGS[1] (no silent
# default -- caller must choose).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl", "draw_design.jl",
          "pairwise_quantile_cutoff_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl"]
    isdefined(Main, Symbol(splitext(f)[1])) # no-op, just for readability
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : error("usage: julia profile_pairwise_quantile_d20.jl <W>")
println("=== D20 profiling, W=$W ===")
flush(stdout)

t_ctx = @elapsed begin
    global ctx = d20_real_setup_design(; W = W, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = 20260719,
        destination_sample = :exclude_row, σHat = 3.0, inner_lower_limit = -10.0)
end
println("context build: ", round(t_ctx, digits=2), "s  D=", ctx.D, "  size(U)=", size(ctx.U))
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileCutoffLayout(ctx.D)

t_aug = @elapsed begin
    global aug = build_pairwise_quantile_augmented_obj(ctx, layout)
end
println("augmented obj + PairwiseQuantileOperator build (incl. presort): ", round(t_aug, digits=2), "s")
flush(stdout)

bin_state = PairwiseQuantileBinState(size(ctx.U, 1), ctx.D)
hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, bin_state, aug.core_cf_ref)
ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_bin_state = bin_state,
                      pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx))

function quantile_naive(v::AbstractVector{Float64}, p::Float64)
    s = sort(v); n = length(s)
    return s[clamp(round(Int, p*n), 1, n)]
end
raw_cutoffs = zeros(n_raw(layout))
for o in 1:ctx.D
    base = raw_index(layout, o, 1)
    Uo = @view ctx.U[:, o]
    q = [quantile_naive(Uo, r/5) for r in 1:4]
    raw_cutoffs[base] = log(q[1])
    for k in 2:4
        gap = log(q[k]) - log(q[k-1])
        raw_cutoffs[base+k-1] = gap > 0 ? log(expm1(gap)) : -5.0
    end
end

println("\n=== running real KNITRO inner solve (this drives the Hessian callbacks we're profiling) ===")
flush(stdout)
t_solve = @elapsed begin
    global nStatus, x, obj, n_fg, n_hess = archPQ_base_state(x_free_calib, raw_cutoffs, ctx, ctx_cm, layout)
end
println("solve: ", round(t_solve, digits=2), "s  nStatus=", nStatus, "  n_fg=", n_fg, "  n_hess=", n_hess)
flush(stdout)

D = ctx.D; op = aug.op; npair = op.npair
n_rows = n_total_rows(D)
println("\nD=$D n_bins=5 n_cutoffs=4 outer_cutoff_params=$(4*D) unordered_pairs=$npair marginal_rows=$(n_marginal_rows(D)) pair_rows=$(n_pair_rows(D)) total_rows=$n_rows dense_G_production=false")
println("ncombo3 (T3) = ", length(op.triple_combos), "   ncombo4 (T4) = ", length(op.quad_combos))

# ---- one dedicated, timed Hessian callback at the SOLVED point, sub-block broken out ----
obj.arg0 .= (nStatus in (0,-100,-101,-103)) ? obj.arg0 : obj.arg0   # already synced by the solve
cf = aug.core_cf_ref[]
wctx = build_winner_pair_ctx(cf)
NCORE = aug.ncore_econ
obj.ddPsi!(obj.arg2, obj.arg0)
h = obj.arg2
tls = build_pairwise_quantile_thread_scratch(D, npair)
tabs = PairwiseQuantileHessianTables(op)

println("\n=== Hessian sub-block timing (one callback at the solved point) ===")
t_HEE = @elapsed begin
    hee_packed = Vector{Float64}(undef, NCORE*(NCORE+1)÷2)
    winner_pair_hessian!(hee_packed, obj, wctx)
end
@printf("H_EE (winner-pair, unchanged shared backend): %.4fs\n", t_HEE)

t_T1T2 = @elapsed build_pairwise_quantile_hessian_tables!(tabs, op, bin_state, h, tls)
@printf("T1/T2/T3/T4 raw table build (combined, includes H_MM/MP/PP raw material): %.4fs\n", t_T1T2)

HRR = zeros(n_rows, n_rows)
t_fill = @elapsed fill_pairwise_quantile_hessian_raw!(HRR, op, tabs)
@printf("H_MM/MP/PP raw block-fill (from tables): %.4fs\n", t_fill)

t_center = @elapsed center_and_scale_pairwise_quantile_hessian!(HRR, op, tabs)
@printf("centering correction (dense n_rows x n_rows pass): %.4fs\n", t_center)

cross_scratch = ensure_winner_zc_cross_scratch!(Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing), op.W, n_rows)
t_prep = @elapsed winner_pair_cross_hessian_zc_prep!(cross_scratch, wctx, h)
@printf("H_E,R prep (Snu/crs_buf): %.4fs\n", t_prep)

HEQ = zeros(NCORE, n_rows)
t_cross = @elapsed pairwise_quantile_cross_hessian_block!(HEQ, wctx, cross_scratch, op, bin_state, tls, h)
@printf("H_E,R cross-block (economic x restriction): %.4fs\n", t_cross)

t_pack = @elapsed begin
    local n = NCORE + n_rows
    local hvec = Vector{Float64}(undef, n*(n+1)÷2)
    local Hfull = zeros(n, n)
    local k = 0
    for i in 1:NCORE, j in i:NCORE
        k += 1
        Hfull[i,j] = hee_packed[k]; Hfull[j,i] = hee_packed[k]
    end
    Hfull[1:NCORE, NCORE+1:n] .= HEQ
    Hfull[NCORE+1:n, 1:NCORE] .= transpose(HEQ)
    Hfull[NCORE+1:n, NCORE+1:n] .= HRR
    local kk = 0
    for i in 1:n, j in i:n
        kk += 1
        hvec[kk] = 0.5*(Hfull[i,j]+Hfull[j,i])
    end
end
@printf("final assembly + packing (dense n x n, n=%d): %.4fs\n", NCORE+n_rows, t_pack)

t_total = t_HEE + t_T1T2 + t_fill + t_center + t_prep + t_cross + t_pack
@printf("\nSUM of measured sub-blocks: %.4fs\n", t_total)

# allocations (separate @allocated call per block, cheap re-run)
a_HEE = @allocated winner_pair_hessian!(hee_packed, obj, wctx)
a_T1T2 = @allocated build_pairwise_quantile_hessian_tables!(tabs, op, bin_state, h, tls)
a_fill = @allocated fill_pairwise_quantile_hessian_raw!(HRR, op, tabs)
a_center = @allocated center_and_scale_pairwise_quantile_hessian!(HRR, op, tabs)
a_cross = @allocated pairwise_quantile_cross_hessian_block!(HEQ, wctx, cross_scratch, op, bin_state, tls, h)
println("\n=== allocations (bytes, second call so JIT-warm) ===")
@printf("H_EE: %d   T1/T2/T3/T4 tables: %d   H_MM/MP/PP fill: %d   centering: %d   H_E,R cross: %d\n",
        a_HEE, a_T1T2, a_fill, a_center, a_cross)

# RSS
rss_kb = try
    parse(Int, split(read(`ps -o rss= -p $(getpid())`, String))[1])
catch
    -1
end
println("\nRSS after full profiling pass: ", rss_kb, " KB (", round(rss_kb/1024/1024, digits=2), " GB)")

println("\nDONE")
