# WHAT are the near-null directions, and are they even REAL? (2026-08-09, user's challenge: "I'm
# confused by your very exact decomposition of 20 and 190 and 570... If this is 'structural, not
# numerical accident' then shouldn't we be able to identify it theoretically?")
#
# METHODOLOGICAL CORRECTION this script exists to make: the previous script
# (cross_block_conditioning_2026-08-09.jl) formed V = M'M/W - mu*mu' and eigendecomposed it. Forming
# a Gram matrix SQUARES the condition number, so with float64 (eps~2.2e-16) any eigenvalue below
# ~eps*cond(V) is numerical garbage. If cond of the CENTERED DATA is ~1e6 then cond(V)~1e12 and
# eigenvalues at 1e-10/1e-12 relative -- exactly the ones the "190"/"570" counts came from, and
# exactly the ones that drove the chi2 ratio from 1.97 (safe tol) up to 7.16 (unsafe tol) -- are not
# trustworthy. This script recomputes everything from the SVD of the CENTERED matrix directly, where
# eigenvalue lambda_i = s_i^2/W and a relative eigenvalue of 1e-12 corresponds to a relative singular
# value of 1e-6, comfortably resolvable in float64. Same quantities, numerically stable route.
#
# Then it ANSWERS the structural question by inspecting the actual right singular vectors:
#   - how much of each near-null direction's mass sits on the mean block vs each pair block
#   - whether each direction is LOCALIZED on a single origin / single origin-pair (which is what
#     "20 = one per origin" or "190 = one per pair" would require if those counts were real)
#
# It also tests ONE concrete THEORETICAL prediction derived by hand. Write z_o = a, z_p = b,
# independent, common mean m_k = E[a^k] = Gamma(1-mu*k). To FIRST order in the fluctuations
# (a^k = m_k + ea_k, b^k = m_k + eb_k, with ea/eb small -- here sd(z)/E(z) ~ 0.19, so this is a
# genuinely small expansion parameter):
#       a^k b^k - m_k^2  =  (m_k+ea_k)(m_k+eb_k) - m_k^2  =  m_k(ea_k + eb_k) + ea_k*eb_k
#                        ~=  m_k * [(a^k - m_k) + (b^k - m_k)]        (dropping the 2nd-order term)
# i.e. the DIAGONAL pair moment is, to first order, a fixed linear combination of the two mean
# moments at the same level. So the combination
#       c_{op,k} := (a^k b^k) - m_k*(a^k) - m_k*(b^k)
# should have variance of order (2nd-order)^2 << the individual variances -- a genuine, theoretically
# derived near-null direction, one per (pair, level). This script measures Var(c)/Var(a^k b^k)
# directly and compares it to the naive first-order prediction, for both the diagonal and the
# off-diagonal (k1!=k2) cross combos.
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
println("ctx built. D=$D muHat=$μ W=$W")
flush(stdout)

νfull0 = vcat([fill(gamma(1 - μ * k), D) for k in 1:K_mean]...)
base_layout  = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
Zraw_all, Zpairraw_diag = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, K_pair; μ = μ)
Zpairraw_cross = build_raw_cross_pair_matrix_levels(Zraw_all, K_pair)
pairs = packed_pair_index(D)
npair = length(pairs)

# =================== (0) THEORETICAL first-order near-null prediction ===================
println("\n================ (0) Hand-derived first-order near-null combination ================")
println("  c_{op,k1k2} := z_o^k1 z_p^k2  -  m_k2 * z_o^k1  -  m_k1 * z_p^k2   (should be ~2nd order)")
println("  sd(z^k)/E[z^k] (the expansion parameter):")
for k in 1:K_mean
    zk = Zraw_all[k]
    @printf("    k=%d: mean=%.4f sd=%.4f  ratio=%.4f\n", k, mean(zk), std(zk), std(zk)/mean(zk))
end
mk = [gamma(1 - μ*k) for k in 1:K_mean]
println("  Var(c)/Var(pair feature), averaged over the 190 pairs:")
levels = cross_pair_level_index(K_pair)
for (klin, (k1, k2)) in enumerate(levels)
    ratios = Float64[]
    for (j, (o, p)) in enumerate(pairs)
        f  = @view Zpairraw_cross[klin][:, j]
        za = @view Zraw_all[k1][:, o]
        zb = @view Zraw_all[k2][:, p]
        c  = f .- mk[k2] .* za .- mk[k1] .* zb
        push!(ratios, var(c) / var(f))
    end
    tag = k1 == k2 ? "DIAG" : "off "
    @printf("    (k1=%d,k2=%d) [%s]: mean Var(c)/Var(f) = %.5f   (first-order theory predicts << 1)\n",
            k1, k2, tag, mean(ratios))
end
flush(stdout)

# =================== stable SVD-based spectrum + null-direction structure ===================
function assemble(layout, Zpair_blocks, nblocks)
    cols = Matrix{Float64}[]; tgts = Float64[]
    labels = Tuple{Symbol,Int,Int,Int}[]   # (:mean or :pair, level/klin, origin_or_o, p)
    for k in 1:K_mean
        push!(cols, Zraw_all[k]); append!(tgts, mean_targets(layout, νfull0, k, D))
        for o in 1:D; push!(labels, (:mean, k, o, 0)); end
    end
    for klin in 1:nblocks
        push!(cols, Zpair_blocks[klin]); append!(tgts, pair_targets(layout, νfull0, klin, D))
        for (o, p) in pairs; push!(labels, (:pair, klin, o, p)); end
    end
    return hcat(cols...), tgts, labels
end

function analyze(name, layout, Zpair_blocks, nblocks)
    println("\n================ $name : STABLE SVD spectrum ================")
    flush(stdout)
    M, tgt, labels = assemble(layout, Zpair_blocks, nblocks)
    p = size(M, 2)
    mhat = vec(mean(M, dims = 1))
    g = mhat .- tgt
    Mc = M .- mhat'                      # centered
    t = @elapsed F = svd(Mc)             # economy SVD, numerically stable
    s = F.S
    lam = (s .^ 2) ./ W                  # eigenvalues of V
    @printf("  p=%d  SVD in %.1fs\n", p, t)
    @printf("  singular values: max=%.4e  min=%.4e  cond(centered M)=%.4e\n", s[1], s[end], s[1]/s[end])
    @printf("  implied cond(V) = cond(M)^2 = %.4e\n", (s[1]/s[end])^2)
    @printf("  float64 RELIABILITY FLOOR on relative eigenvalues from the OLD Gram route: ~eps*cond(V) = %.2e\n",
            2.22e-16 * (s[1]/s[end])^2)
    println("  relative eigenvalue lam_i/lam_1 quantiles:")
    for q in (1.0, 0.99, 0.95, 0.90, 0.75, 0.5)
        i = max(1, Int(round(q * p)))
        @printf("    rank %4d (%.0f%%): lam/lam_1 = %.4e\n", i, 100q, lam[i]/lam[1])
    end
    for thr in (1e-6, 1e-8, 1e-10, 1e-12)
        @printf("    count lam/lam_1 < %.0e : %d of %d (%.1f%%)\n", thr, count(lam ./ lam[1] .< thr), p,
                100*count(lam ./ lam[1] .< thr)/p)
    end
    flush(stdout)

    # ---- structure of the smallest directions ----
    println("  structure of the 6 SMALLEST directions (mass by block, and localization):")
    nmean = K_mean * D
    for r in 0:5
        i = p - r
        v = F.V[:, i]
        mass_mean = sum(abs2, v[1:nmean])
        blockmass = [sum(abs2, v[nmean+(b-1)*npair+1 : nmean+b*npair]) for b in 1:nblocks]
        # localization: how many origins/pairs carry 90% of the mass?
        srt = sort(abs2.(v), rev = true); cum = cumsum(srt)
        n90 = findfirst(>=(0.9), cum)
        top = sortperm(abs2.(v), rev = true)[1:3]
        @printf("    rank %4d lam/lam_1=%.3e | mean-block mass=%.3f | max pair-block mass=%.3f (block %d) | #coords for 90%% mass=%d\n",
                i, lam[i]/lam[1], mass_mean, maximum(blockmass), argmax(blockmass), n90)
        @printf("        top-3 coords: %s\n", join([string(labels[c]) for c in top], ", "))
    end
    flush(stdout)

    # ---- chi2 with a truncation sweep, but now flagged against the reliability floor ----
    gt = F.V' * g
    floorrel = 2.22e-16 * (s[1]/s[end])^2
    println("  chi2 = W*g'V^-1 g (stable SVD route), truncation sweep:")
    out = Tuple{Float64,Int,Float64,Float64}[]
    for tol in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
        keep = lam ./ lam[1] .>= tol
        chi2 = W * sum((gt[keep] .^ 2) ./ lam[keep])
        flag = tol < floorrel ? "  <-- BELOW old-Gram reliability floor" : ""
        @printf("    tol=%.0e: kept %4d/%4d | chi2=%.4e | chi2/df=%.3f | implied Delta*=%.6f%s\n",
                tol, count(keep), p, chi2, chi2/count(keep), chi2/(2W), flag)
        push!(out, (tol, count(keep), chi2, chi2/(2W)))
    end
    @printf("  ACTUAL solved Delta* = %.6f\n", ACTUAL_DELTA[name])
    flush(stdout)
    return out
end

rb = analyze("base",  base_layout,  Zpairraw_diag,  K_pair)
rc = analyze("cross", cross_layout, Zpairraw_cross, K_pair^2)

println("\n================ COMPARISON (stable route) ================")
@printf("  ACTUAL Delta* ratio = %.3f\n", ACTUAL_DELTA["cross"]/ACTUAL_DELTA["base"])
for i in eachindex(rb)
    @printf("    tol=%.0e: chi2 ratio = %.3f  (implied Delta* cross=%.6f base=%.6f)\n",
            rb[i][1], rc[i][3]/rb[i][3], rc[i][4], rb[i][4])
end
