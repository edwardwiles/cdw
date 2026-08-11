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
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl"]
    isdefined(Main, Symbol(splitext(f)[1])) # no-op, just for readability
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : error("usage: julia profile_pairwise_quantile_d20.jl <W> <L>")
PQ_L = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : error("usage: julia profile_pairwise_quantile_d20.jl <W> <L> [cutoff_source]")
# Version B needs an EXPLICIT cutoff source (no default in production); this profiler defaults to
# the campaign's own choice, the theoretical Frechet quantiles, and takes an override in ARGS[3].
CUTOFF_SOURCE = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :frechet_theoretical
println("=== D20 profiling, W=$W, L=$PQ_L ===")
flush(stdout)

t_ctx = @elapsed begin
    global ctx = d20_real_setup_design(; W = W, δ = 1.0, find_smallest = true,
        draw_design = :pseudorandom, draw_seed = 20260719,
        destination_sample = :exclude_row, σHat = 3.0, inner_lower_limit = -10.0)
end
println("context build: ", round(t_ctx, digits=2), "s  D=", ctx.D, "  size(U)=", size(ctx.U))
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)

t_aug = @elapsed begin
    global Zfeat = pairwise_quantile_frechet_features(ctx.U, ctx.μHat)   # restriction is on the Frechet z
    global aug = build_pairwise_quantile_augmented_obj(ctx, layout, Zfeat,
        pairwise_quantile_fixed_cutoffs(Zfeat, PQ_L; cutoff_source = CUTOFF_SOURCE, mu_frechet = ctx.μHat))
end
println("augmented obj + PairwiseQuantileOperator build (incl. presort): ", round(t_aug, digits=2), "s")
flush(stdout)

mass_state = PairwiseQuantileMassState(ctx.D, PQ_L)
hess_ctx = PairwiseQuantileCoreHessCtx(aug.ncore_econ, aug.op, mass_state, aug.core_cf_ref)
ctx_cm = merge(ctx, (obj = aug.obj_pq, pq_op = aug.op, pq_mass_state = mass_state,
                      pq_core_cf_ref = aug.core_cf_ref, pq_hess_ctx = hess_ctx))

# VERSION B: the outer restriction coordinates are BIN MASSES on the simplex; the canonical
# starting point is mu = 1/L (uniform_mass_raw) -- exactly version A's fixed target. The
# cutoffs are no longer outer coordinates at all: they are FIXED at context-build time from
# CUTOFF_SOURCE (see pairwise_quantile_fixed_cutoffs).
raw_masses = uniform_mass_raw(layout)

println("\n=== running real KNITRO inner solve (this drives the Hessian callbacks we're profiling) ===")
flush(stdout)
t_solve = @elapsed begin
    global nStatus, x, obj, n_fg, n_hess = archPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm, layout)
end
println("solve: ", round(t_solve, digits=2), "s  nStatus=", nStatus, "  n_fg=", n_fg, "  n_hess=", n_hess)
flush(stdout)

D = ctx.D; op = aug.op; npair = op.npair
n_rows = n_total_rows(D, PQ_L)
println("\nD=$D n_bins=$PQ_L n_cutoffs=$(PQ_L-1) outer_cutoff_params=$((PQ_L-1)*D) unordered_pairs=$npair marginal_rows=$(n_marginal_rows(D,PQ_L)) pair_rows=$(n_pair_rows(D,PQ_L)) total_rows=$n_rows dense_G_production=false")
println("ncombo3 (T3) = ", length(op.triple_combos), "   ncombo4 (T4) = ", length(op.quad_combos))

# ---- one dedicated, timed Hessian callback at the SOLVED point, sub-block broken out ----
obj.arg0 .= (nStatus in (0,-100,-101,-103)) ? obj.arg0 : obj.arg0   # already synced by the solve
cf = aug.core_cf_ref[]
wctx = build_winner_pair_ctx(cf)
NCORE = aug.ncore_econ
obj.ddPsi!(obj.arg2, obj.arg0)
h = obj.arg2
tls = build_pairwise_quantile_thread_scratch(D, npair, PQ_L)
tabs = PairwiseQuantileHessianTables(op)

println("\n=== Hessian sub-block timing (one callback at the solved point) ===")
t_HEE = @elapsed begin
    hee_packed = Vector{Float64}(undef, NCORE*(NCORE+1)÷2)
    winner_pair_hessian!(hee_packed, obj, wctx)
end
@printf("H_EE (winner-pair, unchanged shared backend): %.4fs\n", t_HEE)

t_T1T2 = @elapsed build_pairwise_quantile_hessian_tables!(tabs, op, h, tls)
@printf("T1/T2/T3/T4 raw table build (combined, includes H_MM/MP/PP raw material): %.4fs\n", t_T1T2)

HRR = zeros(n_rows, n_rows)
t_fill = @elapsed fill_pairwise_quantile_hessian_raw!(HRR, op, tabs)
@printf("H_MM/MP/PP raw block-fill (from tables): %.4fs\n", t_fill)

t_center = @elapsed center_and_scale_pairwise_quantile_hessian!(HRR, op, mass_state, tabs)
@printf("centering correction (dense n_rows x n_rows pass): %.4fs\n", t_center)

cross_scratch = ensure_winner_zc_cross_scratch!(Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing), op.W, n_rows)
t_prep = @elapsed winner_pair_cross_hessian_zc_prep!(cross_scratch, wctx, h)
@printf("H_E,R prep (Snu/crs_buf): %.4fs\n", t_prep)

HEQ = zeros(NCORE, n_rows)
cross_hess_scratch = PairwiseQuantileCrossHessScratch(D, npair, op.W, NCORE - 1, PQ_L)
t_cross = @elapsed pairwise_quantile_cross_hessian_block!(HEQ, wctx, cross_scratch, op, mass_state, tls, h, cross_hess_scratch)
@printf("H_E,R cross-block (economic x restriction): %.4fs\n", t_cross)

# ================================================================================================
# FINAL PACKED WRITE -- timed by calling the REAL production callback, not by replicating it here.
#
# ⚠️ THIS BLOCK USED TO BE A HAND-INLINED COPY OF THE PRODUCTION LOOP AT TOP-LEVEL SCOPE, AND ITS
# NUMBER WAS AN ARTIFACT OF THAT, NOT A PROPERTY OF PRODUCTION. At D=20/L=10/W=100k it reported
# ~86 s. The real production loop lives inside `pairwisequantile_hess_cb_builder`'s closure, where
# `HRR`/`HEQ`/`hee_packed`/`n` are captured with concrete types; the copy here read them as
# NON-CONST GLOBALS, making every one of ~127M element accesses a dynamic dispatch. Measured
# side-by-side at the identical L=10 dimensions (2026-08-11):
#
#     same loop inside a function, serial, row-walk (what production actually ran)   1.90 s
#     same loop inside a function, threaded, column-walk (production today)          0.40 s
#     same loop at TOP-LEVEL scope reading non-const globals (what this file did)   84.70 s
#
# So the "packed write is 46% of the L=10 solve" conclusion drawn from the old number was wrong by
# ~45x, and the sub-block sum it fed was wrong with it. Never re-inline a hot production loop into
# a profiling script's top level: call the real thing.
# ================================================================================================
local hess_ctx_prof = PairwiseQuantileCoreHessCtx(NCORE, op, mass_state, aug.core_cf_ref)
local hess_cb_prof = pairwisequantile_hess_cb_builder(hess_ctx_prof)
struct _ProfEvalRequest; x::Vector{Float64}; end
mutable struct _ProfEvalResult; hess::Vector{Float64}; end
local n_full = NCORE + n_rows
local prof_result = _ProfEvalResult(Vector{Float64}(undef, div(n_full * (n_full + 1), 2)))
local prof_request = _ProfEvalRequest(Float64[])
hess_cb_prof(nothing, nothing, prof_request, prof_result, obj)   # warm up / compile
t_callback = @elapsed hess_cb_prof(nothing, nothing, prof_request, prof_result, obj)
@printf("REAL production Hessian callback, end to end (n=%d): %.4fs\n", n_full, t_callback)
# The packed write is what the whole callback costs beyond the sub-blocks measured above.
t_pack = max(t_callback - (t_HEE + t_T1T2 + t_fill + t_center + t_prep + t_cross), 0.0)
@printf("  of which the final packed write (callback minus measured sub-blocks): %.4fs\n", t_pack)

t_total = t_HEE + t_T1T2 + t_fill + t_center + t_prep + t_cross + t_pack
@printf("\nSUM of measured sub-blocks: %.4fs\n", t_total)
@printf("REAL callback (independent measurement of the same thing): %.4fs\n", t_callback)
@printf("=> assembly is %.1f%% of the solve (%d callbacks x %.2fs of %.1fs)\n",
        100 * n_hess * t_callback / max(t_solve, eps()), n_hess, t_callback, t_solve)
@printf("=> the remaining %.1f%% is KNITRO's own work (dense %dx%d KKT factorization per IP iteration) + FG\n",
        100 * (1 - n_hess * t_callback / max(t_solve, eps())), n_full, n_full)

# allocations (separate @allocated call per block, cheap re-run)
a_HEE = @allocated winner_pair_hessian!(hee_packed, obj, wctx)
a_T1T2 = @allocated build_pairwise_quantile_hessian_tables!(tabs, op, h, tls)
a_fill = @allocated fill_pairwise_quantile_hessian_raw!(HRR, op, tabs)
a_center = @allocated center_and_scale_pairwise_quantile_hessian!(HRR, op, mass_state, tabs)
a_cross = @allocated pairwise_quantile_cross_hessian_block!(HEQ, wctx, cross_scratch, op, mass_state, tls, h, cross_hess_scratch)
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
