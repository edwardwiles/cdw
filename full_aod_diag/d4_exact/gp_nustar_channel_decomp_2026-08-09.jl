# Decompose d(Delta_dual)/d(gp) into its two channels, to test specifically whether gp's effect
# THROUGH nu_star (= nu_{bi,k*=2}, the Variant D derived value) is handled correctly (2026-08-09,
# user question: "the thing that strikes me is the ENORMOUS difference in the analytic gradient wrt
# gp [-1.53 base vs -29.92 cross]. Are you certain you're dealing with the way that gp shows up in
# nu_2 in the right way?").
#
# Under Variant D, gp affects Delta_dual through TWO channels:
#   (1) DIRECT: the economic block (gp enters the moments/payoff directly). This is what
#       economic_A_gradient! returns in g_econ[1] BEFORE apply_focal_kstar_chain_rule! is applied.
#   (2) VIA nu_star: nu_{bi,k*} is DERIVED as cf_denom/cf_num, which depends on gp -- so moving gp
#       moves the k*-level target for the focal origin, which moves Delta_dual. This is exactly the
#       term apply_focal_kstar_chain_rule! adds: coeff * d(nu_star)/d(gp), with
#       coeff = d_delta_d_nu_star (from d_delta_dual_d_eta_active_and_nustar[_cross]) and
#       d(nu_star)/d(gp) = nu_star * sigma / gp (nu_star_value_and_dgrad, autarky_cf.jl).
#
# Three independent checks, cheapest first:
#
#  (i)  d(nu_star)/d(gp) itself, analytic (nu_star*sigma/gp) vs central FD of the pure algebraic
#       function originzc_profiled_nu_value. NO inner solve needed -- pure algebra, instant, and it
#       isolates one factor of the chain-rule product completely.
#  (ii) The channel DECOMPOSITION: report g_direct (pre-chain-rule), the chain term
#       (coeff*d_nu_d_gp), and their sum, separately, for both families. If the cross family's ~20x
#       larger total gradient comes from `coeff` (i.e. genuinely many more restrictions touching
#       nu_{bi,2} in the K_pair^2 grid) that is REAL; if it comes from a mishandled d_nu_d_gp it is a BUG.
#  (iii) FD ISOLATION of channel (2): re-solve at perturbed gp with nu_star deliberately HELD FIXED
#       at its baseline value (breaking channel 2 only). That FD estimates channel (1) alone and must
#       match g_direct. Combined with the already-run full FD (nu_star recomputed, = both channels),
#       the DIFFERENCE of the two FDs is an independent empirical estimate of channel (2) alone,
#       directly comparable to the analytic chain term. This is the decisive test.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl",
          "compressed_live.jl", "autarky_cf.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

# local copy of originzc_profiled_nu_value (avoids including cm_originzc_checkpoint.jl, which a
# concurrent background agent is editing) -- identical formula, cm_originzc_checkpoint.jl:32-46.
function local_profiled_nu_value(xf::AbstractVector{Float64}, ctx)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    AodPow = aod_pow_matrix(θ_full, ctx)
    γ_prime_bi = θ_full[3+ctx.D]
    cf_num, cf_denom, _ = autarky_cf_scalars(ctx.obj, AodPow, ctx.σ, γ_prime_bi)
    return cf_denom / cf_num
end

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
println("ctx built. D=$D sigma=$(ctx.σ) bi=$(ctx.bi)")
flush(stdout)

const K_mean = 3; const K_pair = 3; const KSTAR = 2
νfull0_dense_theory = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

# ===================== (i) d(nu_star)/d(gp): analytic vs FD, no solve =====================
println("\n================ (i) d(nu_star)/d(gp): analytic vs pure-algebra FD ================")
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
nu_star0, d_nu_d_gp_analytic, d_lognu_d_adid = nu_star_value_and_dgrad(θ_full_calib, ctx)
gp0 = x_free_calib[1]
for h in (1e-5, 1e-6, 1e-7)
    fdp = local_profiled_nu_value(vcat(gp0 + h, x_free_calib[2:end]), ctx)
    fdm = local_profiled_nu_value(vcat(gp0 - h, x_free_calib[2:end]), ctx)
    fd = (fdp - fdm) / (2h)
    @printf("  h=%.0e: analytic d(nu*)/d(gp)=%.10f   FD=%.10f   rel diff=%.3e\n",
            h, d_nu_d_gp_analytic, fd, abs(d_nu_d_gp_analytic - fd)/abs(fd))
end
@printf("  (nu_star=%.10f, gp=%.10f, sigma=%.1f -> nu*sigma/gp=%.10f)\n",
        nu_star0, gp0, ctx.σ, nu_star0 * ctx.σ / gp0)
flush(stdout)

# ===================== (ii)+(iii) channel decomposition + FD isolation =====================
function decompose(label, layout, aml)
    println("\n================ $label ================")
    flush(stdout)
    pcx = layout isa OriginByPowerCrossLayout ?
        build_originzc_cross_production_context(ctx, CS, layout; aml = aml) :
        build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, aml = aml)
    νfull_active0, _ = gather_active_grad(aml, νfull0_dense_theory)
    νfull_base = scatter_nu_eff(aml, νfull_active0, nu_star0)

    t0 = @elapsed (base0, verify0) = archOZ_verified_state(x_free_calib, νfull_base, pcx.ctx_cm)
    @printf("  baseline: %.1fs inner_status=%d Delta_dual=%.10f\n", t0, verify0.inner_status, verify0.Delta_dual)
    flush(stdout)

    # --- analytic decomposition ---
    eta_grad_active, d_delta_d_nu_star = layout isa OriginByPowerCrossLayout ?
        d_delta_dual_d_eta_active_and_nustar_cross(base0.λstar, pcx.aug, aml, νfull_base; mean_m = verify0.m_mean) :
        d_delta_dual_d_eta_active_and_nustar(base0.λstar, pcx.aug, aml, νfull_base; mean_m = verify0.m_mean)
    # g_direct: economic gradient BEFORE the chain-rule term is applied.
    cache = layout isa OriginByPowerCrossLayout ?
        build_lfix_base_cache_originzc_cross(x_free_calib, pcx.ctx_cm, base0, pcx.aug, νfull_base) :
        build_lfix_base_cache_originzc(x_free_calib, pcx.ctx_cm, base0, pcx.aug, νfull_base)
    ws = get_or_build_econ_a_grad_ws(cache.W)
    Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
    g_econ = zeros(D * Ddest)
    economic_A_gradient!(g_econ, base0, pcx.ctx_cm, pe, ws; cache = cache, threaded = true)
    g_direct_gp = g_econ[1]
    chain_term_gp = d_delta_d_nu_star * d_nu_d_gp_analytic
    @printf("  ANALYTIC: g_direct(gp)=%+.6f  |  d_delta_d_nu_star=%+.6f * d(nu*)/d(gp)=%+.6f = chain=%+.6f  |  TOTAL=%+.6f\n",
            g_direct_gp, d_delta_d_nu_star, d_nu_d_gp_analytic, chain_term_gp, g_direct_gp + chain_term_gp)
    flush(stdout)

    # --- FD with nu_star HELD FIXED (channel 1 only) ---
    h = 1e-4
    function delta_fixed_nu(gp_pert)
        xfp = vcat(gp_pert, x_free_calib[2:end])
        _, vp = archOZ_verified_state(xfp, νfull_base, pcx.ctx_cm)   # SAME nu vector, not recomputed
        return vp.Delta_dual
    end
    t1 = @elapsed dfp = delta_fixed_nu(gp0 + h)
    t2 = @elapsed dfm = delta_fixed_nu(gp0 - h)
    fd_fixed = (dfp - dfm) / (2h)
    @printf("  FD (nu* HELD FIXED, channel 1 only): %+.6f   [%.0fs+%.0fs]   vs analytic g_direct=%+.6f  rel=%.3e\n",
            fd_fixed, t1, t2, g_direct_gp, abs(fd_fixed - g_direct_gp)/max(1e-12, abs(fd_fixed)))
    flush(stdout)

    # --- FD with nu_star RECOMPUTED (both channels) ---
    function delta_varying_nu(gp_pert)
        xfp = vcat(gp_pert, x_free_calib[2:end])
        nsp = local_profiled_nu_value(xfp, ctx)
        νp = scatter_nu_eff(aml, νfull_active0, nsp)
        _, vp = archOZ_verified_state(xfp, νp, pcx.ctx_cm)
        return vp.Delta_dual
    end
    t3 = @elapsed dvp = delta_varying_nu(gp0 + h)
    t4 = @elapsed dvm = delta_varying_nu(gp0 - h)
    fd_vary = (dvp - dvm) / (2h)
    fd_channel2 = fd_vary - fd_fixed
    @printf("  FD (nu* RECOMPUTED, both channels): %+.6f   [%.0fs+%.0fs]\n", fd_vary, t3, t4)
    @printf("  => FD-implied channel 2 (= vary - fixed) = %+.6f   vs analytic chain term = %+.6f   rel=%.3e\n",
            fd_channel2, chain_term_gp, abs(fd_channel2 - chain_term_gp)/max(1e-12, abs(fd_channel2)))
    flush(stdout)
    return (g_direct = g_direct_gp, chain = chain_term_gp, coeff = d_delta_d_nu_star,
            fd_fixed = fd_fixed, fd_vary = fd_vary, fd_ch2 = fd_channel2)
end

base_layout = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
rb = decompose("base family, Variant D (kstar=2)", base_layout, ActiveMeanLayout(base_layout, ctx.bi, KSTAR, D))
rc = decompose("OZC-CROSS, Variant D (kstar=2)", cross_layout, ActiveMeanLayout(cross_layout, ctx.bi, KSTAR, D))

println("\n================ SUMMARY: where does the ~20x come from? ================")
@printf("  base : direct=%+.4f  chain=%+.4f  (coeff=%+.4f)  total=%+.4f\n", rb.g_direct, rb.chain, rb.coeff, rb.g_direct+rb.chain)
@printf("  cross: direct=%+.4f  chain=%+.4f  (coeff=%+.4f)  total=%+.4f\n", rc.g_direct, rc.chain, rc.coeff, rc.g_direct+rc.chain)
@printf("  ratio: direct %.2fx | chain %.2fx | coeff %.2fx | total %.2fx\n",
        rc.g_direct/rb.g_direct, rc.chain/rb.chain, rc.coeff/rb.coeff, (rc.g_direct+rc.chain)/(rb.g_direct+rb.chain))
