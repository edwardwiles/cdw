# ================================================================================================
# REAL-context D=4 EXACT-HESSIAN gate for the CM + pairwise-quantile family (family #7, 2026-08-12).
#
# Gates the packed assembler (`pack_cmpq_hessian!` + `cmpq_fill_hessian_blocks!` +
# `cmpq_hess_cb_builder`) against a reference that is EXACT, not a finite difference.
#
# WHY IT CAN BE EXACT. The per-draw dual index is LINEAR in the inner variables,
# `r = -zeta - E*lambda_E - G_R*lambda_R - G_CM*lambda_CM = -M*x`, so
#
#       H  =  (1/W) * M' diag(h) M ,   M = [1 | E | G_R | G_CM],   h_w = Psi''(r_w)
#
# with no second-order term. And because `r` is linear, `M` itself can be READ OUT of the production
# operator with no dense-G machinery and no derivative approximation at all: `M[:,j] = -dual_index!(st, e_j)`,
# column by column, EXACTLY (r(0) = 0). That gives a fully independent dense reference covering ALL
# SIX blocks -- including the economic H_EE/H_E,R/H_E,CM, which the standalone dense oracle cannot
# see, since it loads no economic context.
#
# This does not violate the "no dense G, ever" rule: `M` is built HERE, in a test, at D=4/W=8000
# (8000 x ~400 doubles), for the sole purpose of being a reference. Production never materializes it,
# and this file must never be included by production code.
#
# CHECKS
#   1  layout/bookkeeping: n_x, block widths, sig monotone, packed length.
#   2  `cmpq_pk_upper` == `pairwise_quantile_production.jl::_pk_upper` == a literal running counter.
#   3  THE MAIN GATE: the packed vector == packed upper triangle of (1/W) M' diag(h) M, reported
#      per block PAIR so a failure localizes to one of the six blocks immediately.
#   4  H_EE by TWO independent routes -- `winner_pair_hessian!` (this family's) vs
#      `fill_core_hessian_upper!`/`_fill_cm_HEE!` (CM's, default `:exact_winner_pair_parallel`
#      backend). Free strong gate: two stacks, one block, no extra derivation. Doubles as the live
#      confirmation that releasing `cctx.Ews` did not break CM's own H_EE route.
#   5  the direct `sig` read == `extract_cmpq_HRR!`/`extract_cmpq_HER!` (the gated extractors).
#   6  threaded vs serial CM bin tables give the identical packed vector.
#   7  H*d == FD of the analytic gradient. Checks the Hessian is the Hessian OF THIS f, not merely
#      of the Gram identity -- i.e. it would catch the linearity premise itself being wrong.
#   8  a REAL KNITRO inner solve under `ek_inner_cmpq.opt` (hessopt=exact): status, n_fg, n_hess,
#      Delta_dual, and the gradient norm at the returned point.
#
# Usage:  julia --project=. full_aod_diag/d4_exact/test_cm_pairwise_quantile_real_d4_hessian.jl
# ================================================================================================

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
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

const NFAIL = Ref(0); const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    ok ? (NPASS[] += 1) : (NFAIL[] += 1)
    println(ok ? "  PASS  " : "  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    flush(stdout)
    return ok
end

println("=== building ctx via d4_exact_setup ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
@printf("ctx.D=%d  size(ctx.U)=%s  muHat=%.6g  sigma=%.6g  refIndex1=%d  obj.d=%d\n",
        ctx.D, string(size(ctx.U)), ctx.μHat, ctx.σ, ctx.γ.refIndex1, ctx.obj.d)
flush(stdout)

const PROD_OPT = joinpath(dirname(D4X), "ek_inner_cmpq.opt")

"""
Column-by-column extraction of `M = [1 | E | G_R | G_CM]` straight out of the production operator:
`r(x) = -M*x` is linear with `r(0) = 0`, so `M[:,j] = -dual_index!(st, e_j)` EXACTLY. No dense-G
machinery, no finite difference. Test-only (see this file's header).
"""
function extract_dense_M(st, n_x::Int, W::Int)
    M = Matrix{Float64}(undef, W, n_x)
    e = zeros(n_x)
    for j in 1:n_x
        fill!(e, 0.0); e[j] = 1.0
        r = dual_index!(st, e)
        @inbounds for w in 1:W
            M[w, j] = -r[w]
        end
    end
    return M
end

function run_case(; L::Int, G::Int, n_families::Int, contrasts::Symbol, mass_start::Symbol,
                    seed::Int, do_solve::Bool)
    println("\n", "="^96)
    @printf("CASE  L=%d  G=%d (%d CM levels)  families=%d  contrasts=%s  mass_start=%s\n",
            L, G, G - 1, n_families, String(contrasts), String(mass_start))
    println("="^96); flush(stdout)

    cfg = CMPairwiseQuantileConfig(L = L, cm_grid_size = G, cm_moment_families = n_families,
                                   contrasts = contrasts, min_bin_count = 1, mass_start = mass_start)
    cmpq = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = PROD_OPT)
    ctx_cm = cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx = true)
    st = ctx_cm.cmpq_fg_state
    octx = ctx_cm.cmpq_hess_ctx
    obj = ctx_cm.obj
    D = ctx.D; nc = L - 1
    NCORE = cmpq.ncore_econ; n_restr = cmpq.n_restr; ncm = cmpq.ncm
    W = cmpq.op.W
    n_x = obj.outer_constr_index
    nR = NCORE + n_restr
    npacked = div(n_x * (n_x + 1), 2)

    # ---- CHECK 1: layout ------------------------------------------------------------------------
    check("layout: n_x == NCORE + n_restr + ncm", n_x == NCORE + n_restr + ncm,
          "n_x=$n_x = $NCORE+$n_restr+$ncm")
    check("layout: sig is strictly increasing and in range",
          all(octx.sig[i] < octx.sig[i+1] for i in 1:n_restr-1) &&
          octx.sig[1] >= 1 && octx.sig[end] <= n_total_rows(D, L),
          "sig[1]=$(octx.sig[1]) sig[end]=$(octx.sig[end]) of $(n_total_rows(D, L))")
    check("layout: cctx carries this family's economic width and CM level count",
          octx.cctx.NCORE == NCORE && octx.cctx.L == cmpq.Lcm && octx.cctx.ncm == ncm,
          "cctx.NCORE=$(octx.cctx.NCORE) cctx.L=$(octx.cctx.L) cctx.ncm=$(octx.cctx.ncm)")

    # ---- CHECK 2: the packed index formula, three ways -------------------------------------------
    ok_pk = true; k = 0
    for i in 1:NCORE, j in i:NCORE
        k += 1
        (cmpq_pk_upper(i, j, NCORE) == k && _pk_upper(i, j, NCORE) == k) || (ok_pk = false)
    end
    check("cmpq_pk_upper == _pk_upper == running counter (row-major upper triangle)", ok_pk,
          "$k entries over NCORE=$NCORE")

    # ---- prime the economic block and set a NON-UNIFORM shared mu ---------------------------------
    θ_econ0 = CS.reconstruct_full(x_free_calib, ctx_cm.m)
    prime_operator!(obj, θ_econ0, ctx, cmpq.core_cf_ref)
    rng = MersenneTwister(seed)
    mu_target = [0.30, 0.10, 0.25, 0.08, 0.12, 0.06, 0.04, 0.02, 0.015][1:nc]
    mu_target ./= (sum(mu_target) / 0.9)
    raw_masses = zeros(nc); raw_from_origin_masses!(raw_masses, mu_target)
    reset_for_solve!(st, raw_masses)
    if nc >= 2
        check("shared mu decoded non-uniform",
              maximum(st.mass_state.mu) - minimum(st.mass_state.mu) > 0.01,
              @sprintf("mu in [%.4f, %.4f]", minimum(st.mass_state.mu), maximum(st.mass_state.mu)))
    else
        println("  SKIP  non-uniform mu: L=$L has a single free mass coordinate")
    end

    # ---- the evaluation point --------------------------------------------------------------------
    x = 0.05 .* randn(rng, n_x)
    x[1] = 0.2
    g = zeros(n_x)
    f = st(x, g)
    check("FG returns finite f and g at the test point", isfinite(f) && all(isfinite, g),
          @sprintf("f=%.8g", f))

    # ---- the EXACT dense reference ----------------------------------------------------------------
    Mden = extract_dense_M(st, n_x, W)
    check("M's zeta column is the constant 1 (r is affine with r(0)=0, as the identity assumes)",
          all(Mden[w, 1] == 1.0 for w in 1:W))
    # r(x) recomputed from M must reproduce dual_index! exactly -- if it does not, `r` is not linear
    # and the whole Gram identity (and hence this reference) is void.
    r_x = copy(dual_index!(st, x))
    r_lin = -(Mden * x)
    e_lin = norm(r_x .- r_lin) / max(1e-300, norm(r_x))
    check("r is LINEAR in x: -M*x reproduces dual_index!(st,x)", e_lin < 1e-12,
          @sprintf("rel L2 %.3e", e_lin))

    operator_prep_for_hessian!(st, x)             # exactly what the callback does first
    obj.ddPsi!(obj.arg2, obj.arg0)
    h = copy(obj.arg2)
    Href = (Mden' * (h .* Mden)) ./ W

    # ---- the production blocks + packed write -----------------------------------------------------
    cmpq_fill_hessian_blocks!(octx, obj)
    HCC = @view octx.cctx.Hfull[NCORE+1:NCORE+ncm, NCORE+1:NCORE+ncm]
    hvec = Vector{Float64}(undef, npacked)
    pack_cmpq_hessian!(hvec, octx.hee_packed, octx.HEQ_pq, octx.HEC, octx.HRR_pq, octx.HRC, HCC,
                       octx.sig, NCORE, n_restr, ncm)

    # ---- CHECK 3: THE MAIN GATE, per block pair ---------------------------------------------------
    blk(i) = i <= NCORE ? 1 : (i <= nR ? 2 : 3)
    names = ("E", "R", "CM")
    worst = zeros(3, 3); scale = zeros(3, 3)
    kk = 0
    for i in 1:n_x, j in i:n_x
        kk += 1
        bi = blk(i); bj = blk(j)
        d = abs(hvec[kk] - Href[i, j])
        d > worst[bi, bj] && (worst[bi, bj] = d)
        a = abs(Href[i, j])
        a > scale[bi, bj] && (scale[bi, bj] = a)
    end
    check("packed vector has the right length", kk == npacked, "$kk of $npacked")
    for bi in 1:3, bj in bi:3
        scale[bi, bj] == 0.0 && continue
        rel = worst[bi, bj] / scale[bi, bj]
        check("packed H block $(names[bi]) x $(names[bj]) == (1/W) M' diag(h) M",
              rel < 1e-9, @sprintf("max abs %.3e / scale %.3e = %.3e", worst[bi, bj], scale[bi, bj], rel))
    end
    allworst = maximum(worst); allscale = maximum(scale)
    check("packed H OVERALL == (1/W) M' diag(h) M", allworst / allscale < 1e-9,
          @sprintf("max abs %.3e (scale %.3e)", allworst, allscale))

    # ---- CHECK 4: H_EE by two independent routes ---------------------------------------------------
    # `winner_pair_hessian!` (this family's, serial winner-pair) vs CM's `_fill_cm_HEE!`, whose
    # default backend is :exact_winner_pair_parallel -- a different workspace and a different fill.
    if octx.cctx.core_hessian_backend === :dense_reference
        println("  SKIP  H_EE two-route gate: CM's core backend is :dense_reference, which needs a " *
                "dense H this operator-native family does not have")
    else
        HEE_cm = zeros(NCORE, NCORE)
        _fill_cm_HEE!(HEE_cm, h, obj, octx.cctx, nothing, obj.M)
        w2 = 0.0; s2 = 0.0; kk2 = 0
        for i in 1:NCORE, j in i:NCORE
            kk2 += 1
            d = abs(octx.hee_packed[kk2] - HEE_cm[i, j])
            d > w2 && (w2 = d)
            abs(HEE_cm[i, j]) > s2 && (s2 = abs(HEE_cm[i, j]))
        end
        check("H_EE agrees between winner_pair_hessian! and CM's fill_core_hessian_upper! route",
              w2 / s2 < 1e-9, @sprintf("max abs %.3e / scale %.3e (backend :%s)", w2, s2,
                                       String(octx.cctx.core_hessian_backend)))
        check("releasing cctx.Ews did not break CM's own H_EE route (it is 0x0 and unread)",
              size(octx.cctx.Ews) == (0, 0), "size=$(size(octx.cctx.Ews))")
    end

    # ---- CHECK 5: the direct sig read == the gated extractors -------------------------------------
    HRR_x = zeros(n_restr, n_restr)
    extract_cmpq_HRR!(HRR_x, octx.HRR_pq, D, L, cmpq.refIndex1)
    okR = true
    for J in 1:n_restr, I in J:n_restr
        octx.HRR_pq[octx.sig[I], octx.sig[J]] == HRR_x[I, J] || (okR = false)
    end
    check("packed sig read of H_RR is BIT-IDENTICAL to extract_cmpq_HRR!", okR)
    HER_x = zeros(NCORE, n_restr)
    extract_cmpq_HER!(HER_x, octx.HEQ_pq, D, L, cmpq.refIndex1)
    okE = all(octx.HEQ_pq[i, octx.sig[J]] == HER_x[i, J] for i in 1:NCORE, J in 1:n_restr)
    check("packed sig read of H_E,R is BIT-IDENTICAL to extract_cmpq_HER!", okE)

    # ---- CHECK 6: threaded vs serial CM bin tables -------------------------------------------------
    was_threaded = octx.threaded_bins
    octx.threaded_bins = !was_threaded
    cmpq_fill_hessian_blocks!(octx, obj)
    hvec2 = Vector{Float64}(undef, npacked)
    HCC2 = @view octx.cctx.Hfull[NCORE+1:NCORE+ncm, NCORE+1:NCORE+ncm]
    pack_cmpq_hessian!(hvec2, octx.hee_packed, octx.HEQ_pq, octx.HEC, octx.HRR_pq, octx.HRC, HCC2,
                       octx.sig, NCORE, n_restr, ncm)
    octx.threaded_bins = was_threaded
    e6 = maximum(abs, hvec .- hvec2) / max(1e-300, maximum(abs, hvec))
    check("threaded and serial CM bin tables give the same packed Hessian", e6 < 1e-12,
          @sprintf("max rel %.3e (ran %s then %s)", e6, was_threaded ? "threaded" : "serial",
                   was_threaded ? "serial" : "threaded"))

    # ---- CHECK 7: H*d == FD of the analytic gradient -----------------------------------------------
    # The Gram identity says what H *should* be; this says the assembled H is the second derivative
    # of the f the FG callback actually returns. Uses the PRODUCTION packed vector, unpacked.
    Hprod = zeros(n_x, n_x)
    kk3 = 0
    for i in 1:n_x, j in i:n_x
        kk3 += 1
        Hprod[i, j] = hvec[kk3]; Hprod[j, i] = hvec[kk3]
    end
    d = randn(MersenneTwister(seed + 1), n_x); d ./= norm(d)
    ε = 1e-6
    gp = zeros(n_x); gm = zeros(n_x)
    st(x .+ ε .* d, gp); st(x .- ε .* d, gm)
    st(x, g)                                     # restore
    hd_fd = (gp .- gm) ./ (2ε)
    hd = Hprod * d
    e7 = norm(hd .- hd_fd) / max(1e-12, norm(hd_fd))
    check("H*d == FD of the analytic gradient (H is the Hessian OF THIS f)", e7 < 1e-5,
          @sprintf("rel L2 %.3e", e7))

    # ---- CHECK 8: a REAL KNITRO inner solve with hessopt=exact ------------------------------------
    if do_solve
        println("  --- REAL KNITRO inner solve, hessopt=exact (", basename(PROD_OPT), ") ---")
        flush(stdout)
        n0 = octx.n_hess_calls
        t0 = time()
        nStatus, xsol, objb, n_fg, n_hess = archCMPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm;
            hess_cb_builder = cmpq_hess_builder_for(ctx_cm))
        el = time() - t0
        gsol = zeros(n_x)
        f_final = st(xsol, gsol)
        @printf("  nStatus=%d  n_fg=%d  n_hess=%d  wall=%.2fs  f=%.12g  Delta_dual=%.12g  |g|=%.3e\n",
                nStatus, n_fg, n_hess, el, f_final, -f_final, norm(gsol)); flush(stdout)
        check("KNITRO reached an expected status under hessopt=exact",
              nStatus in (0, -100, -101, -102, -103), "nStatus=$nStatus")
        check("the exact-Hessian callback actually ran", n_hess > 0 && octx.n_hess_calls > n0,
              "n_hess=$n_hess, octx counted $(octx.n_hess_calls - n0)")
        check("Delta_dual is finite and small-and-positive",
              isfinite(f_final) && -f_final > 0.0 && -f_final < 50.0,
              @sprintf("Delta=%.6g", -f_final))
        check("the exact-Hessian solve converged in far fewer FG evals than the FG-only solve did",
              n_fg < 500, "n_fg=$n_fg (FG-only measured 16,734 at L=5/G=50/families=2)")
    end
    return nothing
end

run_case(L = 5, G = 10, n_families = 1, contrasts = :anchored, mass_start = :uniform,
         seed = 20260812, do_solve = true)
run_case(L = 5, G = 50, n_families = 2, contrasts = :anchored, mass_start = :empirical,
         seed = 20260813, do_solve = true)
run_case(L = 5, G = 10, n_families = 2, contrasts = :orthonormal, mass_start = :uniform,
         seed = 20260816, do_solve = true)
run_case(L = 2, G = 50, n_families = 1, contrasts = :anchored, mass_start = :uniform,
         seed = 20260814, do_solve = false)
run_case(L = 10, G = 50, n_families = 2, contrasts = :anchored, mass_start = :empirical,
         seed = 20260815, do_solve = false)

println("\n", "="^96)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^96)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_real_d4_hessian: $(NFAIL[]) check(s) failed")
