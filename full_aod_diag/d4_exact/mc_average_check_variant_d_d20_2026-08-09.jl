# Explicit Monte Carlo average check under F* (2026-08-09, user request): rather than just reporting
# "residual ~0", print the ACTUAL empirical average of each moment under the solved F* (m_star, the
# tilted/optimal draw weights from the dual solve) side-by-side with the target formula's value, for
# every mean-block level and every pair-block (k1,k2) combo, at D20 real data, K_mean=K_pair=3,
# Variant D (kstar=2) active, for BOTH families. This is the direct, most legible form of "are the
# formulas right": E_{F*}[z_o(w)^k] should equal mean_targets(...)[o], and E_{F*}[z_o^k1 * z_p^k2]
# should equal pair_targets(...)[pair] (= nu_{o,k1}*nu_{p,k2}), to numerical precision, if (a) the
# feature construction (build_raw_cross_pair_matrix_levels/build_raw_mean_pair_matrix_levels) and
# (b) the target formula (mean_targets/pair_targets(OriginByPowerCrossLayout,...)) are both correct.
# Also splits the pair block into "touches bi" vs "doesn't touch bi" subsets, per the open mechanism
# question about whether the level-2 Delta* effect concentrates near the autarky-collinear origin.
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
          "compressed_live.jl", "autarky_cf.jl", "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D sigma=$(ctx.σ) muHat=$(ctx.μHat) bi=$(ctx.bi) W=$W")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]

const K_mean = 3
const K_pair = 3
const KSTAR = 2
νfull0_dense_theory = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

function mc_check(label, layout, aml)
    println("\n================ $label ================")
    flush(stdout)
    pcx = layout isa OriginByPowerCrossLayout ?
        build_originzc_cross_production_context(ctx, CS, layout; aml = aml) :
        build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, aml = aml)
    nu_star = originzc_profiled_nu_value(x_free_calib, ctx)
    νfull_active, _ = gather_active_grad(aml, νfull0_dense_theory)
    νfull = scatter_nu_eff(aml, νfull_active, nu_star)
    base, verify = archOZ_verified_state(x_free_calib, νfull, pcx.ctx_cm)
    println("inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)")
    m_star = base.m_star
    Wn = length(m_star)
    @printf("sum(m_star)=%.10f  sum(m_star)/W=%.10f  (should be 1.0 -- m_star sums to W, not 1; every MC average below divides by W, matching recovered_mean_residuals_origin's own convention)\n", sum(m_star), sum(m_star) / Wn)

    println("\n-- mean block: E_F*[z_o^k] vs mean_targets(...)[o] --")
    for k in 1:K_mean
        active = aml.mean_active_origins[k]
        Zk = pcx.aug.Zraw_all[k]
        tgt = mean_targets(layout, νfull, k, D)
        max_abs = 0.0
        for o in active
            mc_avg = dot(m_star, @view(Zk[:, o])) / Wn
            resid = mc_avg - tgt[o]
            max_abs = max(max_abs, abs(resid))
        end
        tag = k == KSTAR ? " (kstar, focal origin $(ctx.bi) OMITTED from this level)" : ""
        @printf("  level k=%d%s: max|E_F*[z^k] - target| over %d active origins = %.3e\n", k, tag, length(active), max_abs)
    end

    println("\n-- pair block: E_F*[z_o^k1 * z_p^k2] vs pair_targets(...)[pair] --")
    levels = layout isa OriginByPowerCrossLayout ? cross_pair_level_index(K_pair) : [(k, k) for k in 1:K_pair]
    pairs = packed_pair_index(D)
    bi = ctx.bi
    for (klin, (k1, k2)) in enumerate(levels)
        Zpk = pcx.aug.Zpairraw_all[klin]
        tgt = pair_targets(layout, νfull, klin, D)
        max_abs_all = 0.0; max_abs_bi = 0.0; max_abs_nonbi = 0.0
        n_bi = 0; n_nonbi = 0
        for (j, (o, p)) in enumerate(pairs)
            mc_avg = dot(m_star, @view(Zpk[:, j])) / Wn
            resid = mc_avg - tgt[j]
            max_abs_all = max(max_abs_all, abs(resid))
            if o == bi || p == bi
                max_abs_bi = max(max_abs_bi, abs(resid)); n_bi += 1
            else
                max_abs_nonbi = max(max_abs_nonbi, abs(resid)); n_nonbi += 1
            end
        end
        tag = (k1 == KSTAR || k2 == KSTAR) ? " [touches level $KSTAR]" : ""
        @printf("  (k1=%d,k2=%d)%s: max|resid| all %d pairs=%.3e | involves bi(=%d): %d pairs, max=%.3e | no bi: %d pairs, max=%.3e\n",
                k1, k2, tag, length(pairs), max_abs_all, bi, n_bi, max_abs_bi, n_nonbi, max_abs_nonbi)
    end
    flush(stdout)
    return base, verify, pcx
end

base_layout = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
base_aml = ActiveMeanLayout(base_layout, ctx.bi, KSTAR, D)
cross_aml = ActiveMeanLayout(cross_layout, ctx.bi, KSTAR, D)

mc_check("base family, Variant D (kstar=2)", base_layout, base_aml)
mc_check("OZC-CROSS, Variant D (kstar=2)", cross_layout, cross_aml)
