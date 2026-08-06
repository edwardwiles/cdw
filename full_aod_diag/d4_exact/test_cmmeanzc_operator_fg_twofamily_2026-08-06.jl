# D=4 gate (2026-08-06, paired-basis-preconditioning pilot continuation): independently verifies
# CMMeanZCOperatorState's two-family (Pow-gated) forward/backward FG math (cm_meanzc_lookup_kernels.jl)
# against the dense-reference bundle's own native FG evaluation, at real solved and random points.
# This is the gate build_cm_meanzc_production_context's own internal guard has been waiting on
# ("the :operator FG path... has not been extended/verified... within this task's time budget") --
# the CODE already existed (built the same 2026-08-05 pass as CMLookupState's own two-family
# extension) but had never been cross-checked against an independent ground truth. Mirrors the
# existing single-family test_operator_no_H_bundle_equivalence_cmzc.jl's own "compare_at" idiom,
# simplified: only ONE context is needed (moment_representation=:dense_reference, which
# include_truncated_moment=true already requires/defaults to) since CMMeanZCOperatorState is
# constructed DIRECTLY (not through build_cm_meanzc_production_context, which still refuses
# moment_representation=:operator for two-family) and wraps the SAME dense obj -- CMMeanZCOperatorState
# only ever reads obj.Psi!/obj.dPsi!/obj.U, identical for either bundle type, so this genuinely tests
# its OWN math against ground truth, not bundle-type consistency.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "threaded_cross_hessian.jl", "winner_pair_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_lookup_kernels.jl",
          "cm_meanzc_lookup_production.jl", "cm_meanzc_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(20260806)
L = 3       # matches test_cm_meanzc_archc_twofamily_2026-08-05.jl's own known-good D4 config --
K_mean = 1  # L=10 was tried first and hit a genuine KNITRO infeasibility (nStatus=-300) at the
            # calibration point for this widened+two-family combination at D4's tiny synthetic
            # scale; L=3 is the value the existing (Hessian-side) two-family CM+ZC gate already
            # uses successfully, not picked to dodge a real bug -- this test only needs a FEASIBLE
            # evaluation point (any x, not necessarily a converged optimum) to compare FG math, so
            # even a smaller-scale config is fully sufficient for what it checks.

println("="^90)
println("CM+ZC two-family (include_truncated_moment=true): CMMeanZCOperatorState FG vs dense reference")
println("="^90)

for contrasts in (:anchored, :orthonormal)
    println("== contrasts = $contrasts ==")
    # 2026-08-06: build aug/cctx DIRECTLY (test_cm_meanzc_archc_twofamily_2026-08-05.jl's own
    # pattern), NOT through build_cm_meanzc_production_context -- that wrapper's own internal guard
    # is exactly what this gate exists to eventually justify relaxing, and (separately, discovered
    # live) passing inner_fg_backend=:dense_reference into it leaves cctx.meanzc_zc_op/zc_layout
    # unbuilt (Nothing), since build_cm_meanzc_bin_ctx only constructs the ZC operator/layout when
    # inner_fg_backend=:operator (its own default, CM_MEANZC_INNER_FG_BACKEND_DEFAULT[]) -- calling
    # it directly with no override uses that default and builds them regardless of moment_representation.
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = 0, contrasts = contrasts,
                                         include_truncated_moment = true)
    check("aug.n_families == 2", aug.n_families == 2)
    obj_d = aug.obj_cm
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    check("cctx.n_families == 2", cctx.n_families == 2)
    check("cctx.Pow !== nothing", cctx.Pow !== nothing)
    check("cctx.meanzc_zc_op !== nothing", cctx.meanzc_zc_op !== nothing)

    # 2026-08-06: proper nu1_guess (mean of the K=1 Z target's own raw draws), matching
    # test_cm_meanzc_archc_twofamily_2026-08-05.jl's own already-validated construction -- a naive
    # fixed guess (e.g. 1.0) is a genuinely different, less-realistic evaluation point, not needed
    # for a pure FG-math cross-check (which works at ANY feasible-to-EVALUATE x, not just a solved
    # optimum) but kept for realism/consistency with the existing test.
    nu1_guess = sum(aug.Zraw_all[1]) / length(aug.Zraw_all[1])
    νvec0 = [nu1_guess]
    x_free_ext = vcat(x_free_calib, fill(nu1_guess, K_mean))
    m_ext = CS.FreeParamMap(ctx.l_full + K_mean, vcat(ctx.free_idx, [ctx.l_full + k for k in 1:K_mean]),
                             ctx.fixed_idx, ctx.fixed_vals)
    θ_ext_calib = CS.reconstruct_full(x_free_ext, m_ext)

    cctx.nu_ref[] = collect(νvec0)   # required before any CM+ZC Hessian/FG call -- refresh_zc_targets!
    # (zc_restriction_operator.jl) reads this to build the current outer point's mean/pair targets.

    n = cctx.NCORE + cctx.ncm
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)

    function compare_at(x::AbstractVector; label = "")
        st = CMMeanZCOperatorState(obj_d, cctx.ncore_core - 1, cctx.meanzc_zc_op, cctx.meanzc_zc_layout,
                                    cctx.core_cf_ref, cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1,
                                    bins_u, cctx.R; Pow = cctx.Pow)
        reset_for_solve!(st, νvec0)

        obj_d.moments!(@view(obj_d.H[:, 1]), CS.select_G_from_H(obj_d, obj_d.H), θ_ext_calib, obj_d.U, obj_d)
        obj_d.H[:, 2] .= 1.0

        g_dense = zeros(n)
        f_dense = obj_d(x, g_dense)

        g_op = zeros(n)
        f_op = st(x, g_op)

        e_f = abs(f_dense - f_op)
        e_g = maximum(abs.(g_dense .- g_op))
        @printf("  %-20s |f_dense-f_op|=%.3e  max|Δg|=%.3e\n", label, e_f, e_g)
        check("$label: objective agrees", e_f < 1e-10)
        check("$label: gradient agrees", e_g < 1e-8)
    end

    compare_at(zeros(n); label = "x=0")
    for i in 1:4
        compare_at(0.05 .* randn(n); label = "random[$i]")
    end
    println()
end

println("="^90)
if isempty(FAILURES)
    println("ALL CM+ZC TWO-FAMILY OPERATOR-FG-VS-DENSE GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
end
println("TOTAL: this-run check() calls above")
exit(isempty(FAILURES) ? 0 : 1)
