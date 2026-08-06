# 2026-08-05: diagnostic-only comparison of Architecture C's (hessian_cm_structured!, the real
# no-dense-H production Hessian) and Architecture A's (dense PsiObjectiveBundleImplicit callable)
# Hessian VALUES along the actual dual-iterate sequence a real, live, failing Architecture C
# KNITRO solve visits -- not just at 4 random points (test_cm_autodiff_groundtruth_2026-08-05.jl's
# own check, section 20 of the source MASTER.md). Motivation: the paired-basis-preconditioning
# pilot found the RAW two-family basis already converges cleanly (nStatus=0) via dense Architecture
# A at the exact point/W/L where Architecture C gets nStatus=-400 -- if the two Hessians were truly
# identical (not just close at random points), Newton-type convergence should be
# architecture-invariant up to ordinary roundoff. This script tests that directly: does Architecture
# C's Hessian actually match Architecture A's at the SPECIFIC points the real failing solve visits?
#
# Zero new Hessian/gradient math: both architectures' EXISTING Hessian implementations
# (archC_hess_cb_builder/hessian_cm_structured! for Architecture C, the dense obj(x,h=...) callable
# for Architecture A) are called unmodified. The only "new" code is a thin logging wrapper around
# the KNITRO Hessian callback that snapshots the current point and diffs the two outputs -- pure
# instrumentation, not a new Hessian formula. The real solve's behavior is unaffected (the wrapper
# still calls the real callback to fill evalResult.hess; KNITRO sees byte-identical output to an
# uninstrumented run).
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
const L = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const scale = length(ARGS) >= 3 ? ARGS[3] : "d20"   # "d4" | "d20"
const threaded_bins_flag = length(ARGS) >= 4 ? (ARGS[4] == "true") : true   # 2026-08-05 root-cause
# confirmation: build_bin_tables_threaded!/prefix_sum_tables_threaded! (cm_hessian_threaded.jl) have
# no Ttab12/Ttab22/CT12/CT22/Pow/fam2 handling at all (written before the two-family extension) --
# only the SERIAL build_bin_tables!/prefix_sum_tables! (cm_hessian_architectures.jl) fill them.
# threaded_bins=true is build_cm_bin_ctx's own default (what production actually runs). Passing
# false here forces the serial path as a decisive, code-change-free empirical test of that theory.

if scale == "d4"
    global ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
else
    global ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
end
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
lp("Context built. D=", ctx.D, " sigma=", ctx.σ, " muHat=", ctx.μHat, " W(draws)=", size(ctx.U, 1))

probs_ = collect(range(1 / L, (L - 1) / L, length = L))

# ---- Architecture C production context: the REAL no-dense-H path that fails ----
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs_,
    include_truncated_moment = true, moment_representation = :operator, inner_fg_backend = :cm_lookup,
    use_archB_moments = false, threaded_bins = threaded_bins_flag)
lp("threaded_bins=", threaded_bins_flag, " (cctx.use_threaded_bins=", pcx.cctx.use_threaded_bins, ")")
lp("Arch C: n_families=", pcx.cctx.n_families, " ncm=", pcx.cctx.ncm, " ncore=", pcx.cctx.NCORE)

# ---- Architecture A dense reference object, SAME context/theta -- reused unmodified from the
# main pilot's own construction pattern ----
aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored, probs = probs_)
obj_dense = aug.obj_cm
Kd = zeros(size(ctx.U, 1))
obj_dense.moments!(Kd, CS.select_G_from_H(obj_dense, obj_dense.H), θ_full_calib, ctx.U, obj_dense)
ncm_cdf = aug.ncm_cdf
ncore = aug.ncore
lp("Arch A (dense): ncore=", ncore, " ncm=", aug.ncm, " ncm_cdf=", ncm_cdf)

nvars = CS.inner_loop_number_variables(pcx.ctx_cm.obj)
lp("KNITRO inner variable count nvars=", nvars)

# ---- instrumented Hessian callback: calls the REAL Architecture C callback unmodified, then
# separately evaluates Architecture A's dense Hessian at the SAME point and diffs ----
struct HessDiffRec
    iter::Int
    x::Vector{Float64}
    maxdiff_all::Float64
    maxdiff_core::Float64
    maxdiff_cdf::Float64
    maxdiff_pow::Float64
    maxdiff_cross_cdfpow::Float64
    argmax_row::Int
    argmax_col::Int
    fg_diff::Float64   # |Architecture C's own FG functor f - dense obj_dense's own f| at the SAME
    # point -- rules out a coordinate-ordering confound: an ordering mismatch between
    # CMLookupState's dual vector and obj_dense's column order would corrupt FG identically to
    # Hessian, whereas a genuine Hessian-specific bug leaves FG intact.
end
diag_log = HessDiffRec[]

function make_wrapped_hess_builder(cctx)
    real_cb = archC_hess_cb_builder(cctx)
    return (_obj) -> (kc, cb, evalRequest, evalResult, userParams) -> begin
        r = real_cb(kc, cb, evalRequest, evalResult, userParams)
        xloc = copy(evalRequest.x)
        # KNITRO's KN_DENSE_ROWMAJOR Hessian buffer for a SYMMETRIC Hessian is the packed UPPER
        # TRIANGLE (length n*(n+1)/2, row i then columns j=i..n), not a full n^2 matrix -- confirmed
        # live (first attempt asserted nvars^2=6084 and got a KNITRO callback exception reporting
        # the real buffer length 3081 = 78*79/2). Both sides fill the SAME buffer type/convention
        # (obj_dense(x,h=...) is the same callable production's own dense Hessian path uses), so
        # unpacking both identically and diffing the resulting symmetric matrices is correct.
        npacked = (nvars * (nvars + 1)) ÷ 2
        @assert length(evalResult.hess) == npacked "Hessian buffer length $(length(evalResult.hess)) != n(n+1)/2=$npacked"
        H_c = copy(evalResult.hess)
        H_d = similar(H_c)
        obj_dense(xloc, h = H_d)
        function unpack_upper(v, n)
            M = zeros(n, n)
            k = 1
            for i in 1:n, j in i:n
                M[i, j] = v[k]; M[j, i] = v[k]
                k += 1
            end
            return M
        end
        Hc_mat = unpack_upper(H_c, nvars)
        Hd_mat = unpack_upper(H_d, nvars)
        Δ = abs.(Hc_mat .- Hd_mat)
        maxdiff_all, idx = findmax(Δ)
        # index layout: row/col 1 = zeta; 2:ncore = core (pre-gravity + gravity... matches
        # wrap_moments_with_cm's own layout: core cols 1:(ncore-1), CM ncore:(ncore+ncm-1), gravity
        # last); offset by +1 throughout because index 1 in the Hessian is zeta, not the first
        # moment column.
        cdf_rng = (1 + ncore) : (ncore + ncm_cdf)
        pow_rng = (1 + ncore + ncm_cdf) : (ncore + 2 * ncm_cdf)
        core_rng = 2:ncore
        maxdiff_core = maximum(Δ[core_rng, core_rng])
        maxdiff_cdf = maximum(Δ[cdf_rng, cdf_rng])
        maxdiff_pow = maximum(Δ[pow_rng, pow_rng])
        maxdiff_cross = maximum(Δ[cdf_rng, pow_rng])

        # FG cross-check at the SAME xloc: Architecture C's own CMLookupState functor (the exact
        # object whose FG callback KNITRO is calling every iteration -- already cached on cctx by
        # the time the Hessian callback fires, since KNITRO always evaluates FG before H) vs
        # obj_dense's own plain-objective call. Both existing, unmodified implementations.
        st = cctx.cmlookup_st
        g_c = zeros(nvars)
        f_c = st(xloc, g_c)
        f_d = obj_dense(xloc)
        fg_diff = abs(f_c - f_d)

        push!(diag_log, HessDiffRec(length(diag_log) + 1, xloc, maxdiff_all, maxdiff_core, maxdiff_cdf,
                                     maxdiff_pow, maxdiff_cross, idx[1], idx[2], fg_diff))
        lp("  [hess call ", length(diag_log), "] maxdiff_all=", maxdiff_all, " core=", maxdiff_core,
           " cdf=", maxdiff_cdf, " pow=", maxdiff_pow, " cross(cdf,pow)=", maxdiff_cross,
           " argmax=(", idx[1], ",", idx[2], ")", " fg_diff=", fg_diff)
        return r
    end
end

wrapped_builder = make_wrapped_hess_builder(pcx.cctx)

θ_full0 = θ_full_calib
obj = pcx.ctx_cm.obj
lp("="^100); lp("Running instrumented Architecture C solve (real production path, unmodified except Hessian logging)"); lp("="^100)
t0 = time()
K, xsol, nStatus, n_fg, n_hess = inner_loop_internal_cmlookup_production(obj, θ_full0, pcx.cctx;
    hess_cb_builder = wrapped_builder, skip_fill = false)
lp("RESULT: nStatus=", nStatus, "  n_fg=", n_fg, "  n_hess=", n_hess, "  t=", round(time() - t0, digits = 2), "s")

lp("="^100); lp("SUMMARY over ", length(diag_log), " captured Hessian calls"); lp("="^100)
if !isempty(diag_log)
    lp("iter 1 (first Hessian call): maxdiff_all=", diag_log[1].maxdiff_all, " cdf=", diag_log[1].maxdiff_cdf,
       " pow=", diag_log[1].maxdiff_pow, " cross=", diag_log[1].maxdiff_cross_cdfpow)
    lp("last iter: maxdiff_all=", diag_log[end].maxdiff_all, " cdf=", diag_log[end].maxdiff_cdf,
       " pow=", diag_log[end].maxdiff_pow, " cross=", diag_log[end].maxdiff_cross_cdfpow)
    lp("max over all iters: maxdiff_all=", maximum(r -> r.maxdiff_all, diag_log),
       " maxdiff_core=", maximum(r -> r.maxdiff_core, diag_log),
       " maxdiff_cdf=", maximum(r -> r.maxdiff_cdf, diag_log),
       " maxdiff_pow=", maximum(r -> r.maxdiff_pow, diag_log),
       " maxdiff_cross=", maximum(r -> r.maxdiff_cross_cdfpow, diag_log))
    lp("min over all iters: maxdiff_all=", minimum(r -> r.maxdiff_all, diag_log))
    lp("FG cross-check: max|fg_diff| over all iters=", maximum(r -> r.fg_diff, diag_log),
       " min|fg_diff|=", minimum(r -> r.fg_diff, diag_log))
    # localize the single worst (row,col) across every iterate -- convert back to (family, l, origin)
    worst = argmax(r -> r.maxdiff_all, diag_log)
    lp("worst single entry: iter=", worst.iter, " (row,col)=(", worst.argmax_row, ",", worst.argmax_col,
       ")  maxdiff_all=", worst.maxdiff_all)
end
lp("DONE")
