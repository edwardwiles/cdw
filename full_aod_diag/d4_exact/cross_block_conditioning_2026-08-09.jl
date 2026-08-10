# Are the CROSS blocks specifically ill-conditioned? (2026-08-09, user's sharpened question: "the
# concern is not about the mean. That's also there in the diagonal version, and Delta* is very low.
# The concern is specifically about the cross blocks.")
#
# Correct framing: individual moment discrepancies being ~1 SE from target (which I already showed
# for every block, mean AND cross) does NOT bound Delta*. Delta* is a JOINT quadratic form,
# approximately
#     Delta* ~= (1/(2W)) * chi2,     chi2 = W * g' V^{-1} g
# where g is the vector of moment discrepancies (obs mean - target) and V is the COVARIANCE of the
# moment functions. If V is near-singular, V^{-1} amplifies perfectly ordinary-looking g enormously.
#
# And there is a concrete structural reason to suspect exactly that here: for ONE origin pair (o,p),
# the 9 cross features {z_o^k1 * z_p^k2}_{k1,k2=1..3} are built from only SIX underlying random
# quantities (z_o^1,z_o^2,z_o^3,z_p^1,z_p^2,z_p^3) -- and z_o^2, z_o^3 are deterministic functions of
# z_o^1 (powers of the same draw!), so really from TWO (z_o, z_p). Across the whole design, 1710
# cross-pair restrictions (9 * 190) are functions of just 20 underlying draw columns. The base family
# imposes only the 3 diagonal blocks (570) from the same 20 columns. So the cross grid is FAR more
# redundant, and its V is expected to be much closer to singular.
#
# This script computes, for BOTH families, at the SAME calibration point/draws:
#   - V (empirical covariance of the restriction moment functions: mean block + pair block)
#   - its eigenvalue spectrum, condition number, and effective rank
#   - the chi2 statistic W*g'V^{-1}g (via eigendecomposition, with a spectrum truncation sweep so the
#     answer is not an artifact of one arbitrary regularization choice)
#   - the IMPLIED Delta* ~= chi2/(2W), compared against the ACTUAL solved Delta*
#     (base 0.009489888, cross 0.066730686 at W=100k, K=3/3 -- from this session's runs)
# DECISIVE TEST: if chi2_cross/chi2_base ~= 7 (the observed Delta* ratio), the 7x is fully explained
# by the cross blocks' redundancy/conditioning acting on ordinary MC noise -- real, not a bug.
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
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

const W = 100_000
const K_mean = 3
const K_pair = 3
const ACTUAL_DELTA = Dict("base" => 0.009489888149044069, "cross" => 0.06673068647073198)

println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D; μ = ctx.μHat
println("ctx built. D=$D muHat=$μ W=$W  (BLAS threads=$(BLAS.get_num_threads()))")
flush(stdout)

νfull0 = vcat([fill(gamma(1 - μ * k), D) for k in 1:K_mean]...)
base_layout  = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
Zraw_all, Zpairraw_diag = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, K_pair; μ = μ)
Zpairraw_cross = build_raw_cross_pair_matrix_levels(Zraw_all, K_pair)

"Assemble the (W x p) restriction-moment matrix and its target vector for a family."
function assemble(layout, Zpair_blocks, nblocks)
    cols = Matrix{Float64}[]
    tgts = Float64[]
    for k in 1:K_mean
        push!(cols, Zraw_all[k])
        append!(tgts, mean_targets(layout, νfull0, k, D))
    end
    for klin in 1:nblocks
        push!(cols, Zpair_blocks[klin])
        append!(tgts, pair_targets(layout, νfull0, klin, D))
    end
    return hcat(cols...), tgts
end

function analyze(name, layout, Zpair_blocks, nblocks)
    println("\n================ $name ================")
    flush(stdout)
    M, tgt = assemble(layout, Zpair_blocks, nblocks)
    p = size(M, 2)
    @printf("  moment count p = %d  (mean %d + pair %d)\n", p, K_mean*D, p - K_mean*D)
    flush(stdout)

    mhat = vec(mean(M, dims = 1))
    g = mhat .- tgt
    # V = E[mm'] - mu mu'  (population covariance of the moment FUNCTIONS, not of their means)
    t = @elapsed begin
        V = (M' * M) ./ W
        V .-= mhat * mhat'
        V .= (V .+ V') ./ 2      # enforce exact symmetry before eigen
    end
    @printf("  covariance built in %.1fs\n", t); flush(stdout)

    ev = eigvals(Symmetric(V))
    ev_pos = filter(>(0), ev)
    @printf("  eigenvalues: max=%.4e  min=%.4e  min_positive=%.4e\n", maximum(ev), minimum(ev), minimum(ev_pos))
    @printf("  condition number (max/min_pos) = %.4e\n", maximum(ev)/minimum(ev_pos))
    for thr in (1e-8, 1e-10, 1e-12)
        @printf("    eigenvalues < %.0e * max : %d of %d (%.1f%%)\n",
                thr, count(ev .< thr*maximum(ev)), p, 100*count(ev .< thr*maximum(ev))/p)
    end
    flush(stdout)

    # chi2 = W * g' V^{-1} g, via eigendecomposition with a truncation sweep.
    F = eigen(Symmetric(V))
    gt = F.vectors' * g
    println("  chi2 = W*g'V^-1 g under spectrum truncation (drop eigenvalues < tol*max):")
    results = Tuple{Float64,Int,Float64,Float64}[]
    for tol in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
        keep = F.values .>= tol * maximum(F.values)
        chi2 = W * sum((gt[keep] .^ 2) ./ F.values[keep])
        dfk = count(keep)
        @printf("    tol=%.0e: kept %4d/%4d dims | chi2=%.4e | chi2/df=%.3f | implied Delta*=chi2/(2W)=%.6f\n",
                tol, dfk, p, chi2, chi2/dfk, chi2/(2W))
        push!(results, (tol, dfk, chi2, chi2/(2W)))
    end
    @printf("  ACTUAL solved Delta* = %.6f\n", ACTUAL_DELTA[name])
    flush(stdout)
    return results
end

rb = analyze("base",  base_layout,  Zpairraw_diag,  K_pair)
rc = analyze("cross", cross_layout, Zpairraw_cross, K_pair^2)

println("\n================ DECISIVE COMPARISON ================")
@printf("  ACTUAL Delta* ratio (cross/base) = %.3f\n", ACTUAL_DELTA["cross"]/ACTUAL_DELTA["base"])
println("  Predicted ratio from the chi2 quadratic form, by truncation tolerance:")
for i in eachindex(rb)
    tol = rb[i][1]
    @printf("    tol=%.0e: chi2 ratio = %.3f   (implied Delta* %.6f vs %.6f)\n",
            tol, rc[i][3]/rb[i][3], rc[i][4], rb[i][4])
end
println("\nIf the chi2 ratio ~= the actual ~7.03, the cross blocks' redundancy/ill-conditioning fully")
println("explains the jump acting on ordinary MC noise. If chi2 ratio is ~1-2 while actual is 7, the")
println("quadratic-form story does NOT explain it and something else is going on.")
