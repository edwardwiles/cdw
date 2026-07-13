# §6: sparsity-pattern determination for the moment map's Jacobian
# ∂[K,1,G]/∂θ (N×(d+2) output × l inputs), both from economic structure and
# from numerical union across benchmark points + random perturbations.
include("setup_context.jl")
include("derivative_core.jl")

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]
D = data["D"]; bi = data["bi"]

# ---- (A) expected structural pattern, from economics ----
# θ layout: [μ(1), σ(2,fixed), γ_θ[1..D](3:2+D, INERT/unused), γ'_focal(3+D),
#            A_od (Aod_offset+1 : Aod_offset+D^2), column-major (o,d)]
# G layout (nTotalMoments=18 for D=4): cols 1:D^2 = trade shares (o,d);
#   col D^2+1 = counterfactual price index (focal only); col D^2+2 = gravity.
# H layout: col1=K, col2=const, col 2+j = G col j.
Aod_offset = 3 + D
l = 3 + D + D^2
d = D^2 + 2
expected = falses(d + 2, l)   # rows = H columns (d+2), cols = θ params (l)
expected[1, :] .= false; expected[1, 3+D] = true   # K = θ[3+D] only
expected[2, :] .= false                             # const column, no θ dependence ever
# CORRECTED reasoning (first draft was wrong — caught by exactly the numerical
# check §6 asks for; see report): a trade-share moment ROW (o,d) is
# G[ω,d1] = pricesInd[o]*pricesTempσ[o] - P[d1]*denom[d]. The WINNER indicator
# pricesInd is a HARD argmax (MinInd!) — ForwardDiff treats it as LOCALLY
# CONSTANT away from ties (a well-documented property of this codebase, see
# memory "derivative-algo-experiments": the argmin-indicator's derivative is a
# measure-zero boundary term ForwardDiff correctly omits). So row (o,d)'s
# value depends on A ONLY through pricesTempσ[o], which is built from
# constConsσ[o,d] — i.e. ONLY A[o,d] itself, never A[o',d] for o'≠o, for ANY θ
# (this is a structural fact of the formula, not a fragile per-θ accident).
# The naive "whole column A[:,d] affects every row in that destination"
# intuition describes what determines the WINNER, which is exactly the part
# hard-max makes invisible to AD — the opposite of the naive guess.
# σ (θ[2]) is NOT a free outer parameter (bounds-pinned) but IS a genuine
# argument of the trade-share/price-index formulas (constConsσ ∝ ^(1-σ), plus
# the gamma(μ(1-σ)+1) normalization) — included here since sparsity describes
# the FUNCTION, not what KNITRO chooses to search.
for jd in 1:D
    for o in 1:D
        gcol = o + (jd - 1) * D
        hrow = 2 + gcol
        expected[hrow, 1] = true          # μ
        expected[hrow, 2] = true          # σ
        expected[hrow, Aod_offset + o + (jd - 1) * D] = true   # ONLY this origin's own A entry
    end
end
# counterfactual price-index moment (G col D^2+1): depends on μ, σ, γ'_focal, A[focal,focal] only
# (constConsσ[baseIndex,baseIndex] — same "own entry only" logic; no hard-max here since
# there's no winner selection for the single-country autarky counterfactual moment)
let hrow = 2 + D^2 + 1
    expected[hrow, 1] = true
    expected[hrow, 2] = true
    expected[hrow, 3+D] = true
    expected[hrow, Aod_offset + bi + (bi - 1) * D] = true   # A[baseIndex,baseIndex] only
end
# gravity moment (G col D^2+2): SMOOTH (no hard-max/argmin at all — a two-way-FE
# contrast over the full grid), genuinely DENSE in ALL A_od + μ. Does NOT
# depend on σ (AodPow has no σ exponent) — verified against newGravityMoment!.jl.
let hrow = 2 + D^2 + 2
    expected[hrow, 1] = true
    for oo in 1:D^2
        expected[hrow, Aod_offset + oo] = true
    end
end
println("Expected structural nnz = ", count(expected), " / ", length(expected), " (density=", round(count(expected)/length(expected); digits=4), ")")

# ---- (B)+(C) numerical union: dense ForwardDiff Jacobian at all 4 points + random perturbations ----
Random.seed!(20260712)
θ_lower = ones(l) .* 1e-8   # not used for bound checks here, just for random perturbation scale
test_thetas = [points[k].θ for k in (:A,:B,:C,:D)]
for _ in 1:4
    base = points[:A].θ
    θr = copy(base)
    θr[1] = base[1] * (0.7 + 0.6*rand())                     # perturb μ
    θr[3+D] = clamp(base[3+D] * (0.9 + 0.2*rand()), 0.5, 1.0) # perturb γ'_focal
    for i in Aod_offset+1:Aod_offset+D^2
        θr[i] = base[i] * (0.7 + 0.6*rand())                  # perturb all A_od
    end
    push!(test_thetas, θr)
end

N_test = 200   # small N for sparsity detection speed; STRUCTURE doesn't depend on N (verified: pattern is per-draw-row-independent in θ)
Usub = pp.U[1:N_test, :]
numeric_union = falses(d + 2, l)
tol = 1e-8
for θt in test_thetas
    f! = (Hvec, θ) -> begin
        H = reshape(Hvec, N_test, d + 2)
        moment_map!(H, θ, Usub, pp.γ)
    end
    Hvec0 = zeros(N_test * (d + 2))
    J = ForwardDiff.jacobian(f!, Hvec0, θt)   # (N_test*(d+2)) x l
    Jr = reshape(J, N_test, d + 2, l)
    for hrow in 1:d+2, j in 1:l
        if maximum(abs.(Jr[:, hrow, j])) > tol
            numeric_union[hrow, j] = true
        end
    end
end
println("Numeric union nnz (row=H col, col=θ) = ", count(numeric_union), " / ", length(numeric_union))

mismatch_expected_not_numeric = expected .& .!numeric_union
mismatch_numeric_not_expected = numeric_union .& .!expected
println("Expected-but-not-numeric (economically expected, but derivative ~0 numerically): ", count(mismatch_expected_not_numeric))
println("Numeric-but-not-expected (surprise nonzero, structural pattern missed something): ", count(mismatch_numeric_not_expected))
if count(mismatch_numeric_not_expected) > 0
    idxs = findall(mismatch_numeric_not_expected)
    println("  first few surprises (hrow,thetacol): ", idxs[1:min(10,end)])
end

open(joinpath(@__DIR__, "sparsity_summary.txt"), "w") do f
    println(f, "Moment map H (N x $(d+2)) Jacobian sparsity, θ length l=$l")
    println(f, "Expected (economic) nnz = $(count(expected)) / $(length(expected))  density=$(round(count(expected)/length(expected);digits=4))")
    println(f, "Numeric union nnz       = $(count(numeric_union)) / $(length(numeric_union))")
    println(f, "Expected-not-numeric    = $(count(mismatch_expected_not_numeric))")
    println(f, "Numeric-not-expected    = $(count(mismatch_numeric_not_expected))")
    println(f, "")
    println(f, "Per-row (H column) nnz-in-theta count (numeric union):")
    rownames = vcat("K", "const", ["G_share[o=$( (j-1)%D+1 ),d=$( (j-1)÷D+1 )]" for j in 1:D^2], "G_counterfactual_price", "G_gravity")
    for hrow in 1:d+2
        println(f, "  $(rownames[hrow]): $(count(numeric_union[hrow,:])) / $l nonzero θ-derivatives")
    end
end
@save joinpath(@__DIR__, "sparsity_pattern.jld2") expected numeric_union D bi l d
println("SPARSITY DONE")
