# ================================================================================================
# REAL-context D=4 FG gate for the CM + pairwise-quantile family (family #7, 2026-08-12).
#
# Uses the ACTUAL production economic context (`d4_exact_setup`), the real `OperatorPsiBundle`, the
# real `CompressedFactual` economic block, CM's real lookup kernels, and a REAL `KN_solve` -- not a
# synthetic standalone script. The standalone dense oracle
# (test_cm_pairwise_quantile_d4_dense_oracle.jl) already gates this family's own algebra in
# isolation; what THIS file gates is everything that only exists once the three blocks are composed:
#
#   * the inner variable LAYOUT and every offset into `x` and `g`
#     (`[zeta; lambda_E; lambda_L; lambda_P; lambda_CM]`). This is exactly where the standalone PQ
#     family had a real bug on 2026-08-09 -- `reshape(v,D,nc)` is column-major while `marginal_row`
#     is o-major -- found by a finite-difference check in a real KNITRO context, which is why the
#     central gate here is FD against `f` COORDINATE BY COORDINATE: a swapped or misplaced gradient
#     slot cannot survive it, whereas any check that contracts the gradient against a vector can;
#   * `outer_constr_index` bookkeeping (KNITRO's variable count);
#   * that the composed FG actually drives a real inner solve to convergence.
#
# The FD gate covers the ECONOMIC block too, at no extra effort: it never touches a dense `G`, it
# only re-evaluates `f`. That matters because this family is operator-native and has no dense
# economic path to compare against by construction.
#
# Usage:  julia --project=. full_aod_diag/d4_exact/test_cm_pairwise_quantile_real_d4_fg.jl
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
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

const NFAIL = Ref(0); const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    ok ? (NPASS[] += 1) : (NFAIL[] += 1)
    println(ok ? "  PASS  " : "  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    return ok
end

println("=== building ctx via d4_exact_setup ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
@printf("ctx.D=%d  size(ctx.U)=%s  muHat=%.6g  sigma=%.6g  refIndex1=%d  obj.d=%d\n",
        ctx.D, string(size(ctx.U)), ctx.μHat, ctx.σ, ctx.γ.refIndex1, ctx.obj.d)

const FGONLY_OPT = joinpath(dirname(D4X), "ek_inner_cmpq_fgonly.opt")
const PROD_OPT = joinpath(dirname(D4X), "ek_inner_cmpq.opt")

function run_case(; L::Int, G::Int, n_families::Int, mass_start::Symbol, seed::Int, do_solve::Bool)
    println("\n", "="^92)
    @printf("CASE  L=%d  G=%d (%d CM levels)  families=%d  mass_start=%s\n", L, G, G - 1, n_families,
            String(mass_start))
    println("="^92)
    cfg = CMPairwiseQuantileConfig(L = L, cm_grid_size = G, cm_moment_families = n_families,
                                  contrasts = :anchored, min_bin_count = 1, mass_start = mass_start)
    cmpq = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = FGONLY_OPT)
    ctx_cm = cm_pairwise_quantile_attach(ctx, cmpq)
    st = ctx_cm.cmpq_fg_state
    D = ctx.D; nc = L - 1
    n_restr = cmpq.n_restr
    n_x = cmpq.obj_cmpq.outer_constr_index

    check("gates: bins cross-checked, cutoffs bit-identical",
          cmpq.gates.cutoffs_bit_identical && cmpq.gates.bin_cells_checked == size(ctx.U, 1) * D,
          "cells=$(cmpq.gates.bin_cells_checked), min joint=$(cmpq.gates.min_joint_count)")
    check("outer_constr_index == ncore_econ + n_restr + ncm",
          n_x == cmpq.ncore_econ + n_restr + cmpq.ncm,
          "n_x=$n_x = $(cmpq.ncore_econ)+$(n_restr)+$(cmpq.ncm)")
    check("n_restr == (L-1) + (L-1)^2*C(D,2)", n_restr == nc + nc^2 * div(D * (D - 1), 2),
          "n_restr=$n_restr")
    check("outer mass coordinates collapse to L-1", length(cmpq.raw_start) == n_cmpq_raw(L) == nc,
          "$(length(cmpq.raw_start)) vs standalone PQ's $(nc*D)")

    # ---- prime the economic block at the calibration point (required before ANY FG eval) --------
    θ_econ0 = CS.reconstruct_full(x_free_calib, ctx_cm.m)
    prime_operator!(ctx_cm.obj, θ_econ0, ctx, cmpq.core_cf_ref)
    check("economic state primed (CompressedFactual published)",
          cmpq.core_cf_ref[] isa CompressedFactual)

    # ---- a deliberately NON-UNIFORM mu, and a non-trivial x -------------------------------------
    rng = MersenneTwister(seed)
    mu_target = [0.30, 0.10, 0.25, 0.08, 0.12, 0.06, 0.04, 0.02, 0.015][1:nc]
    mu_target ./= (sum(mu_target) / 0.9)          # keep strictly inside the simplex
    raw_masses = zeros(nc); raw_from_origin_masses!(raw_masses, mu_target)
    reset_for_solve!(st, raw_masses)
    if nc >= 2
        check("shared mu decoded non-uniform",
              maximum(st.mass_state.mu) - minimum(st.mass_state.mu) > 0.01,
              @sprintf("mu in [%.4f, %.4f]", minimum(st.mass_state.mu), maximum(st.mass_state.mu)))
    else
        # L=2 has ONE free mass, so "non-uniform" is not a property it can have. Reported as a skip
        # rather than passed vacuously (same discipline as the dense oracle's 9a/9c skip).
        println("  SKIP  non-uniform mu: L=$L has a single free mass coordinate")
    end

    x = 0.05 .* randn(rng, n_x)
    x[1] = 0.2
    g = zeros(n_x)
    f = st(x, g)
    check("FG returns finite f and g", isfinite(f) && all(isfinite, g), @sprintf("f=%.8g", f))

    # ---- THE central gate: FD of f, coordinate by coordinate ------------------------------------
    # Every block of the gradient is covered: zeta, the economic lambdas, this family's level and
    # pair rows, and CM's own grid duals -- with no dense G anywhere, only re-evaluations of f.
    g_fd = zeros(n_x)
    for i in 1:n_x
        h = 1e-6 * max(1.0, abs(x[i]))
        xp = copy(x); xp[i] += h
        xm = copy(x); xm[i] -= h
        g_fd[i] = (st(xp) - st(xm)) / (2h)
    end
    st(x, g)   # restore state at the base point
    blocks = [("zeta", 1:1), ("lambda_E", 2:1+cmpq.ncore1),
              ("lambda_L (level)", 2+cmpq.ncore1 : 1+cmpq.ncore1+nc),
              ("lambda_P (pair)", 2+cmpq.ncore1+nc : 1+cmpq.ncore1+n_restr),
              ("lambda_CM", 2+cmpq.ncore1+n_restr : n_x)]
    for (nm, rg) in blocks
        num = norm(g[rg] .- g_fd[rg]); den = max(1e-8, norm(g_fd[rg]))
        e = num / den
        check("FD gradient matches, block $nm", e < 2e-5,
              @sprintf("rel L2 %.3e over %d coords", e, length(rg)))
    end
    worst = argmax(abs.(g .- g_fd) ./ max.(1e-8, abs.(g_fd)))
    check("FD gradient matches, worst single coordinate",
          abs(g[worst] - g_fd[worst]) / max(1e-8, abs(g_fd[worst])) < 1e-3,
          @sprintf("coord %d: analytic %.6g vs FD %.6g", worst, g[worst], g_fd[worst]))

    # ---- the restriction block against a dense indicator reference, in the REAL context ---------
    # Independent of the FD check and of the standalone oracle: rebuilds the level/pair columns from
    # raw indicators here and contracts them, so a layout error that FD somehow tolerated would
    # still show up.
    psi1 = similar(st.arg1); ctx_cm.obj.dPsi!(psi1, st.arg0)
    W = op_W = cmpq.op.W
    nc2 = nc
    g_lvl_dense = zeros(nc); g_pair_dense = zeros(nc, nc, cmpq.op.npair)
    S = sum(psi1)
    for a in 1:nc
        acc = 0.0
        for w in 1:W
            Int(cmpq.op.bin[w, cmpq.refIndex1]) == a && (acc += psi1[w])
        end
        g_lvl_dense[a] = -(acc - S * st.mass_state.mu[a]) / W
    end
    for (pidx, (o, p)) in enumerate(cmpq.op.pairs), b in 1:nc, a in 1:nc
        acc = 0.0
        for w in 1:W
            (Int(cmpq.op.bin[w, o]) == a && Int(cmpq.op.bin[w, p]) == b) && (acc += psi1[w])
        end
        g_pair_dense[a, b, pidx] = -(acc - S * st.mass_state.mu[a] * st.mass_state.mu[b]) / W
    end
    off = 1 + cmpq.ncore1
    e_lvl = norm(g[off+1:off+nc] .- g_lvl_dense) / max(1e-12, norm(g_lvl_dense))
    gP_from_g = reshape(g[off+nc+1 : off+n_restr], nc, nc, cmpq.op.npair)
    e_pair = norm(gP_from_g .- g_pair_dense) / max(1e-12, norm(g_pair_dense))
    check("level-row gradient == dense indicator reference", e_lvl < 1e-12, @sprintf("rel %.3e", e_lvl))
    check("pair-row gradient == dense indicator reference", e_pair < 1e-12, @sprintf("rel %.3e", e_pair))

    # ---- REAL KNITRO inner solve, FG only (hessopt=lbfgs, so no Hessian callback is registered) --
    if do_solve
        println("  --- REAL KNITRO inner solve, FG only (", basename(FGONLY_OPT), ") ---")
        t0 = time()
        nStatus, xsol, objb, n_fg, n_hess = archCMPQ_base_state(x_free_calib, raw_masses, ctx, ctx_cm)
        el = time() - t0
        # Delta_dual = -f at the returned point, recomputed through the FG functor itself (the
        # verifier's own reporting convention; `archCMPQ_base_state` deliberately does not return
        # KNITRO's objSol, matching archPQ_base_state's signature).
        f_final = st(xsol)
        @printf("  nStatus=%d  n_fg=%d  n_hess=%d  wall=%.2fs  f=%.12g  Delta_dual=%.12g\n",
                nStatus, n_fg, n_hess, el, f_final, -f_final)
        # `-400` (iteration limit, feasible point) is ACCEPTED for this FG-only smoke, and that is
        # not a lowered bar -- it is what a quasi-Newton solve of a 400-variable inner problem does.
        # Measured here: L=5/G=50/families=2 gives n_x=412 and hits maxit=5000 after 16,734 FG
        # evaluations with Delta_dual finite at 1.92; L=5/G=10 (n_x=145) instead reaches -102. The
        # claim this smoke makes is "the composed FG drives a real KN_solve and yields a finite,
        # sane Delta", NOT "the FG alone converges" -- convergence at production n is the exact
        # Hessian's job (hessopt=exact, ek_inner_cmpq.opt), gated separately. The production option
        # file is never run without a Hessian builder: that is the hard error checked below.
        check("REAL KNITRO inner solve (FG only) drove KN_solve to an expected status",
              nStatus in (0, -100, -101, -102, -103, -400), "nStatus=$nStatus, n_fg=$n_fg")
        check("KNITRO actually iterated (the callback was exercised, not bypassed)", n_fg > 100,
              "n_fg=$n_fg")
        check("Delta_dual is finite and small-and-positive (not diverging to the lower_limit floor)",
              isfinite(f_final) && -f_final > 0.0 && -f_final < 50.0, @sprintf("Delta=%.6g", -f_final))
        check("no Hessian callback was registered under hessopt=lbfgs", n_hess == 0, "n_hess=$n_hess")
        # The production option file asks for an exact Hessian; with no builder supplied that must be
        # a HARD ERROR, not a silent quasi-Newton downgrade.
        cmpq_prod = build_cm_pairwise_quantile_context(ctx, cfg; inner_opt = PROD_OPT)
        ctx_prod = cm_pairwise_quantile_attach(ctx, cmpq_prod)
        θe = CS.reconstruct_full(x_free_calib, ctx_prod.m)
        prime_operator!(ctx_prod.obj, θe, ctx, cmpq_prod.core_cf_ref)
        reset_for_solve!(ctx_prod.cmpq_fg_state, raw_masses)
        threw = false
        try
            archCMPQ_base_state(x_free_calib, raw_masses, ctx, ctx_prod)
        catch e
            threw = occursin("hessopt=exact", sprint(showerror, e))
        end
        check("production option file + no Hessian builder == hard error (no silent downgrade)", threw)
    end
    return nothing
end

run_case(L = 5, G = 10, n_families = 1, mass_start = :uniform, seed = 20260812, do_solve = true)
run_case(L = 5, G = 50, n_families = 2, mass_start = :empirical, seed = 20260813, do_solve = true)
run_case(L = 2, G = 50, n_families = 2, mass_start = :uniform, seed = 20260814, do_solve = false)

println("\n", "="^92)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^92)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_real_d4_fg: $(NFAIL[]) check(s) failed")
