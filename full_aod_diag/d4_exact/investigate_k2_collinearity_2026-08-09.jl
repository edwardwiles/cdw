# Targeted test (2026-08-09, user pushback on the 7x Delta* jump): is level k=2 (=sigma-1, the
# KNOWN focal-country autarky-moment collinearity level for sigma=3 -- see
# fullA-zc-profiled-focal-sigmaminus1-mean-production-ready / cmzc-k2-singularity-root-cause
# memory) the specific driver, given OZC-CROSS's full K=3 cross grid touches level 2 in 5/9 pair
# combos ((1,2),(2,1),(2,2),(2,3),(3,2)) vs the base family's 1/3 ((2,2) only)? Neither this run
# nor the earlier base-family control applies the Variant D fix (aml=nothing in both), so this
# fragility -- if real -- is present in both, just hit far more often by the cross grid.
#
# Direct test: build a K=3 restriction set that KEEPS the cross grid but EXCLUDES every combo
# touching level 2, leaving only {(1,1),(1,3),(3,1),(3,3)} (4 combos, same as base's 3 diagonal
# plus the two k=1/k=3 cross terms, deliberately avoiding level 2 entirely). If Delta* for this
# set is close to a modest multiple of the base family's (not another 7x-style blowup), that
# implicates level-2 involvement specifically, not the cross-grid mechanism in general.
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
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra
using SpecialFunctions: gamma

# ---- Ad hoc layout supporting an ARBITRARY explicit (k1,k2) level list (not the consecutive
# 1:K_pair grid OriginByPowerCrossLayout assumes) -- built ONLY for this diagnostic, not reused
# elsewhere. Mean block (n_eta/target_index) is byte-identical to OriginByPowerCrossLayout/
# OriginByPowerLayout; only pair_targets differs (dispatches on the explicit level list).
struct AdHocLevelListLayout <: MeanZCTargetLayout
    D::Int
    K_mean::Int
    levels::Vector{Tuple{Int,Int}}
end
n_eta(layout::AdHocLevelListLayout) = layout.K_mean * layout.D
target_index(layout::AdHocLevelListLayout, o::Int, k::Int) = (k - 1) * layout.D + o
function pair_targets(layout::AdHocLevelListLayout, νfull::AbstractVector{Float64}, klin::Int, D::Int)
    pairs = packed_pair_index(D)
    k1, k2 = layout.levels[klin]
    return [νfull[target_index(layout, o, k1)] * νfull[target_index(layout, p, k2)] for (o, p) in pairs]
end

function build_adhoc_augmented_obj(ctx, CS, layout::AdHocLevelListLayout)
    obj0 = ctx.obj
    ncore_econ = obj0.d
    D = ctx.D
    K_mean = layout.K_mean
    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = ctx.μHat)
    npair = div(D * (D - 1), 2)
    Zpairraw_all = Vector{Matrix{Float64}}(undef, length(layout.levels))
    pairs = packed_pair_index(D)
    for (klin, (k1, k2)) in enumerate(layout.levels)
        Zk1 = Zraw_all[k1]; Zk2 = Zraw_all[k2]
        Zpk = Matrix{Float64}(undef, size(ctx.U, 1), npair)
        @inbounds for (j, (o, p)) in enumerate(pairs)
            @views Zpk[:, j] .= Zk1[:, o] .* Zk2[:, p]
        end
        Zpairraw_all[klin] = Zpk
    end
    n_mean = K_mean * D
    n_pair = length(layout.levels) * npair
    d_new = ncore_econ + n_mean + n_pair
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair
    obj_oz = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, l = obj0.l, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        inner_loop_opt = obj0.inner_loop_opt)
    return (obj_cm = obj_oz, ncore = ncore_econ, Zraw_all = Zraw_all, Zpairraw_all = Zpairraw_all,
            layout = layout, K_mean = K_mean, K_pair = length(layout.levels), n_mean = n_mean, n_pair = n_pair,
            ncore_econ = ncore_econ, core_cf_ref = Ref{Any}(nothing), moments_skip! = nothing, aml = nothing)
end
function build_adhoc_production_context(ctx, CS, layout::AdHocLevelListLayout)
    aug = build_adhoc_augmented_obj(ctx, CS, layout)
    octx = build_originzc_core_hess_ctx(aug, ctx; fg_backend = :operator)
    ctx_cm = merge(ctx, (obj = aug.obj_cm, octx = octx))
    return (ctx_cm = ctx_cm, aug = aug, octx = octx)
end

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D μHat=$(ctx.μHat)")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]
νfull0 = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:3]...)

println("\n==== K=3, pair grid EXCLUDING level 2: {(1,1),(1,3),(3,1),(3,3)} ====")
flush(stdout)
layout_no2 = AdHocLevelListLayout(D, 3, [(1,1),(1,3),(3,1),(3,3)])
pcx_no2 = build_adhoc_production_context(ctx, CS, layout_no2)
t = @elapsed (base_no2, verify_no2) = archOZ_verified_state(x_free_calib, νfull0, pcx_no2.ctx_cm)
println("solve: $(t)s inner_status=$(verify_no2.inner_status) Delta_dual=$(verify_no2.Delta_dual) max_abs_moment_kkt_resid=$(verify_no2.max_abs_moment_kkt_resid)")
flush(stdout)

println("\n(for comparison, already known: base diagonal-only K=3/3 Delta_dual=0.00948988814906665 (57.7s);")
println(" full OZC-CROSS K=3/3 (9 combos, includes 5 touching level 2) Delta_dual=0.06673068646924812 (~5min))")
