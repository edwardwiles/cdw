# ============================================================================
# Claude Code task 2026-08-01, §13: explicit rank/redundancy diagnostics,
# D=4. Compares the FULL homogeneous basis (D moments per destination,
# including the anchor) against the REDUCED basis (D-1 per destination).
# Produces REDUNDANT_VS_REDUCED_RANK_GATE_D4_2026-08-01.csv.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_contraction_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_hessian_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
using LinearAlgebra, Random, Printf

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
D = ctx.D; Ddest = D
θ0 = copy(ctx.θ0_up)
cf = build_compressed_factual(θ0, ctx; check_ties = false)
W = cf.W
has_france = cf.cf_col > 0
ncolI_full = cf.oci - 1
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
n_red = layout.total_reduced_economic_moments

println("D=$D Ddest=$Ddest W=$W ncolI_full=$ncolI_full n_reduced=$n_red")

# --- Build full and reduced dense G ---
Gfull = zeros(W, ncolI_full)
Hmoments = homogeneous_factual_moment(θ0, ctx)
for d in 1:D
    s = dest_slot(ctx, d)
    for o in 1:D
        j = s + (o - 1) * Ddest
        Gfull[:, j] .= Hmoments[d][:, o] .* cf.nrm[j] .* cf.gdiv[j]
    end
end
Gred = zeros(W, n_red)
for k in eachindex(layout.retained_full_factual_j)
    o = layout.retained_origin[k]; s = layout.retained_slot[k]
    j_full = layout.retained_full_factual_j[k]
    Gred[:, k] .= Hmoments[s][:, o] .* cf.nrm[j_full] .* cf.gdiv[j_full]
end
if has_france
    Hfrance = homogeneous_france_moment(θ0, ctx)
    Gfull[:, cf.cf_col] .= Hfrance .* cf.nrm[cf.cf_col] .* cf.gdiv[cf.cf_col]
    Gred[:, layout.france_ratio_reduced_j] .= Hfrance .* cf.nrm[cf.cf_col] .* cf.gdiv[cf.cf_col]
end

println("\n" * "="^78); println("PART 1: exact per-destination null identity in the FULL basis"); println("="^78)
null_confirmed = Bool[]
for d in 1:D
    s = dest_slot(ctx, d)
    cols = [s + (o - 1) * Ddest for o in 1:D]
    colsum = sum(Gfull[:, cols], dims = 2)
    resid = maximum(abs.(colsum))
    println("destination $d (slot $s): max|sum_o Gfull[:,o,d]| = $resid")
    push!(null_confirmed, resid < 1e-10)
end
@assert all(null_confirmed) "expected EXACT null identity (sum_o == 0) for every destination in the full basis"

println("\n" * "="^78); println("PART 2: rank / singular values -- full vs reduced"); println("="^78)
svF = svd(Gfull)
svR = svd(Gred)
rankF = count(>(1e-8 * maximum(svF.S)), svF.S)
rankR = count(>(1e-8 * maximum(svR.S)), svR.S)
println("FULL:    size=$(size(Gfull))  rank=$rankF (of $ncolI_full)  smallest σ=$(minimum(svF.S))  largest σ=$(maximum(svF.S))  cond=$(maximum(svF.S)/max(minimum(svF.S),1e-300))")
println("REDUCED: size=$(size(Gred))   rank=$rankR (of $n_red)  smallest σ=$(minimum(svR.S))  largest σ=$(maximum(svR.S))  cond=$(maximum(svR.S)/max(minimum(svR.S),1e-300))")
@assert ncolI_full - rankF == Ddest "expected EXACTLY Ddest=$Ddest rank-deficient directions in the full basis, got $(ncolI_full-rankF)"
@assert rankR == n_red "expected the reduced basis to be FULL RANK ($n_red), got $rankR"
println("CONFIRMED: full basis has exactly Ddest=$Ddest zero singular values (one per destination's deleted identity); reduced basis is full column rank.")

println("\n" * "="^78); println("PART 3: Hessian eigenvalues at a valid inner point -- full vs reduced"); println("="^78)
rng = MersenneTwister(20260801 + 3)
x0_full = vcat(0.0, 0.05 .* randn(rng, ncolI_full))
obj = ctx.obj
t0_full = homogeneous_dual_contraction(x0_full[2:end], cf, ctx, θ0)
obj.arg0 .= -x0_full[1] .- t0_full
wctx_full = build_homogeneous_winner_pair_ctx(cf, ctx, θ0)
n_full_packed = 1 + ncolI_full
h_full = zeros(n_full_packed * (n_full_packed + 1) ÷ 2)
homogeneous_winner_pair_hessian!(h_full, obj, wctx_full)
Hfull_dense = zeros(n_full_packed, n_full_packed)
k = 1
for i in 1:n_full_packed, j in i:n_full_packed
    global k
    Hfull_dense[i, j] = h_full[k]; Hfull_dense[j, i] = h_full[k]; k += 1
end
evF = eigvals(Symmetric(Hfull_dense[2:end, 2:end]))

rng2 = MersenneTwister(20260801 + 3)   # same seed shape as x0_full's tail, restricted to retained coords for comparability
x0_red = vcat(0.0, [x0_full[1 + j] for j in layout.retained_full_factual_j])
if has_france
    push!(x0_red, x0_full[1 + cf.cf_col])
end
t0_red = reduced_homogeneous_dual_contraction(x0_red[2:end], cf, ctx, θ0, layout)
obj.arg0 .= -x0_red[1] .- t0_red
wctx_red = build_reduced_homogeneous_winner_pair_ctx(cf, ctx, θ0, layout)
n_red_packed = 1 + n_red
h_red = zeros(n_red_packed * (n_red_packed + 1) ÷ 2)
reduced_homogeneous_winner_pair_hessian!(h_red, obj, wctx_red)
Hred_dense = zeros(n_red_packed, n_red_packed)
k = 1
for i in 1:n_red_packed, j in i:n_red_packed
    global k
    Hred_dense[i, j] = h_red[k]; Hred_dense[j, i] = h_red[k]; k += 1
end
evR = eigvals(Symmetric(Hred_dense[2:end, 2:end]))

println("FULL Hessian (lambda-block) eigenvalues, sorted by |.|: ", sort(abs.(evF))[1:min(6, length(evF))], " ...")
println("REDUCED Hessian (lambda-block) eigenvalues, sorted by |.|: ", sort(abs.(evR))[1:min(6, length(evR))], " ...")
n_near_zero_full = count(<(1e-6), sort(abs.(evF)))
println("FULL: number of near-zero (<1e-6) eigenvalues = $n_near_zero_full (expect >= Ddest=$Ddest)")
println("REDUCED: smallest |eigenvalue| = $(minimum(abs.(evR))) (expect >> 0)")
@assert n_near_zero_full >= Ddest "expected at least Ddest=$Ddest near-zero eigenvalues in the full Hessian's lambda-block"
@assert minimum(abs.(evR)) > 1e-6 "reduced Hessian's lambda-block has an unexpectedly near-zero eigenvalue"
println("CONFIRMED: the full basis's Hessian has >= Ddest structurally-singular directions; the reduced basis's Hessian does not.")

# --- Write CSV deliverable ---
outpath = joinpath(@__DIR__, "..", "..", "REDUNDANT_VS_REDUCED_RANK_GATE_D4_2026-08-01.csv")
open(outpath, "w") do io
    println(io, "metric,full_basis,reduced_basis")
    @printf(io, "n_columns,%d,%d\n", ncolI_full, n_red)
    @printf(io, "rank,%d,%d\n", rankF, rankR)
    @printf(io, "rank_deficiency,%d,%d\n", ncolI_full - rankF, n_red - rankR)
    @printf(io, "smallest_singular_value,%.6e,%.6e\n", minimum(svF.S), minimum(svR.S))
    @printf(io, "largest_singular_value,%.6e,%.6e\n", maximum(svF.S), maximum(svR.S))
    @printf(io, "condition_number,%.6e,%.6e\n", maximum(svF.S) / max(minimum(svF.S), 1e-300), maximum(svR.S) / max(minimum(svR.S), 1e-300))
    @printf(io, "hessian_lambda_block_dim,%d,%d\n", length(evF), length(evR))
    @printf(io, "hessian_smallest_abs_eigenvalue,%.6e,%.6e\n", minimum(abs.(evF)), minimum(abs.(evR)))
    @printf(io, "hessian_n_near_zero_eigenvalues_lt_1e-6,%d,%d\n", n_near_zero_full, count(<(1e-6), abs.(evR)))
    @printf(io, "hessian_largest_abs_eigenvalue,%.6e,%.6e\n", maximum(abs.(evF)), maximum(abs.(evR)))
end
println("\nWrote $outpath")

println("\n" * "="^78); println("ALL RANK/REDUNDANCY DIAGNOSTICS CONFIRMED"); println("="^78)
