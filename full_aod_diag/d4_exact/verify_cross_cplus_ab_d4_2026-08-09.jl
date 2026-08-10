# A/B gate for the new OZC-CROSS Backend C+ gradient (2026-08-09): does
# cm_originzc_cross_production_gradient_cplus (cm_originzc_cross_cplus.jl, NEW, and the one the
# production driver's DEFAULT cm_gradient_backend=:cplus now dispatches to) agree with the ALREADY-
# VERIFIED cm_originzc_cross_production_gradient (:shared_inplace_pooled path)?
#
# This is the necessary gate before running any production outer loop on the default backend: the
# two must agree to tight tolerance at the SAME point, with and without Variant D (aml). The C+ path
# differs ONLY in how the (g,A_od) economic block is computed (workspace-reusing factorized cache +
# composite_gradient_at_Cplus_from_cache instead of economic_A_gradient!); the eta block is computed
# by the identical cross functions in both, so any disagreement isolates to the economic block.
#
# D4 scale: fast, and sufficient -- this is a backend-equivalence check (same math, two
# implementations), not a scale-dependent scientific question.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "compressed_live.jl", "autarky_cf.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl",
          # Backend C+ dependency chain (defines GradWorkspacePool / LFixFactorizedWorkspace /
          # build_lfix_base_cache_C! / composite_gradient_at_Cplus_from_cache) -- must precede the
          # two cplus files below, which reference those types at their own include time.
          "gradient_workspace.jl", "lfix_factorized_workspace.jl", "lfix_base_workspace_pooled.jl",
          "lfix_cm_cplus.jl",
          "cm_originzc_cplus.jl", "cm_originzc_cross_cplus.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS
    ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

W = size(ctx.obj.U, 1)
pool = build_grad_workspace_pool(W)
ws = build_lfix_factorized_workspace(ctx.D, hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D, W)

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    for use_aml in (false, true)
        (use_aml && K_mean < 2) && continue     # kstar=2 needs K_mean>=2
        tag = use_aml ? "Variant D (kstar=2)" : "no aml"
        println("\n==== D4 OZC-CROSS C+ A/B  K=$K_mean/$K_pair  $tag ====")
        layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
        aml = use_aml ? ActiveMeanLayout(layout, ctx.bi, 2, D) : nothing
        pcx = build_originzc_cross_production_context(ctx, CS, layout; aml = aml)

        νdense = nu0_origin(K_mean, D)
        νfull = νdense   # aml only omits a MEAN ROW; the dense nu vector consumed here is unchanged
        base, verify = archOZ_verified_state(x_free_calib, νfull, pcx.ctx_cm)
        check("K=$K_mean/$K_pair $tag: inner solve converged", verify.inner_status in (0, -100, -101, -103))

        # BOTH backends get the SAME h_mode and the SAME shared bandwidth_cache Dict -- the FD
        # bandwidth must match exactly or the comparison is meaningless (this repo has been bitten
        # before: adaptive-h vs fixed-h produces a false ~1e-3 gap that looks like a real bug, see
        # memory feedback-fd-bandwidth-mismatch-looks-like-a-bug). The first call populates the Dict
        # and the second reuses it, which is precisely what makes the two directly comparable, and
        # also mirrors how the production driver threads one bandwidth_cache through every eval.
        bwc = Dict{Int,Float64}()
        g_ref, _ = cm_originzc_cross_production_gradient(x_free_calib, νfull, pcx, ctx, pe;
            base = base, verify = verify, threaded = true, h_mode = :cached, bandwidth_cache = bwc)
        g_cpl, _ = cm_originzc_cross_production_gradient_cplus(x_free_calib, νfull, pcx, ctx, pe, pool, ws;
            base = base, verify = verify, threaded = true, h_mode = :cached, bandwidth_cache = bwc)

        check("K=$K_mean/$K_pair $tag: same length", length(g_ref) == length(g_cpl))
        n_eta_expected = aml === nothing ? n_eta(layout) : aml.n_eta_active
        check("K=$K_mean/$K_pair $tag: eta block length = $(n_eta_expected)",
              length(g_ref) == (D*(hasproperty(ctx,:D_dest) ? ctx.D_dest : D)) + n_eta_expected)
        d = maximum(abs.(g_ref .- g_cpl))
        rel = d / max(1e-12, maximum(abs.(g_ref)))
        @printf("    max|g_ref - g_cplus| = %.4e   rel = %.4e\n", d, rel)
        check("K=$K_mean/$K_pair $tag: C+ matches reference gradient", rel < 1e-8)
    end
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
