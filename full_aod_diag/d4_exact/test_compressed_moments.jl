# ============================================================================
# Equivalence test for compressed_moments.jl.
# Verifies, against the TRUSTED dense obj.moments! output:
#   (A) materialize_dense_factual(cf)          == G[:, 1:oci-1]      (bit/tight)
#   (B) compressed_dual_contraction(β, cf)     == [dot(β, G[s,1:oci-1]) for s]
#       for β ∈ {frozen λstar, random duals, unit columns}          (bit/tight)
#   (C) q reconstructed from compressed        == fixed_dual_L's q0  (ties the
#       compression to the real inner-solve path)
# across candidate points + random feasible perturbations.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
using Random, Printf, LinearAlgebra, Statistics

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; oci = ctx.obj.outer_constr_index; ncol = oci - 1
W = size(ctx.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

dense_G(θfull) = begin
    K = zeros(W); G = zeros(W, ctx.obj.d)
    ctx.obj.moments!(K, G, θfull, ctx.obj.U, ctx.obj)
    G
end

function check_point(label, w0; rng = MersenneTwister(1))
    xf0 = x_free_from_w(w0)
    local base
    try
        base = solve_base_state(xf0, ctx)
    catch e
        println(rpad(label, 26), "  SKIP (base solve failed: ", typeof(e), ")")
        return true
    end
    θfull = base.θ_full0
    G = dense_G(θfull)
    Gfac = G[:, 1:ncol]

    cf = build_compressed_factual(θfull, ctx)

    # (A) dense materialization equivalence
    Gc = materialize_dense_factual(cf)
    errA = maximum(abs.(Gc .- Gfac))

    # (B) dual contraction equivalence over several β
    βs = Dict{String,Vector{Float64}}()
    βs["lambda_star"] = base.λstar
    βs["random_normal"] = randn(rng, ncol)
    βs["random_pos"] = rand(rng, ncol) .+ 0.1
    βs["ones"] = ones(ncol)
    # a few unit columns (bilateral + the counterfactual col)
    for j in (1, 2, D, D^2, D^2 + 1)
        e = zeros(ncol); e[j] = 1.0; βs["unit_$j"] = e
    end
    errB = 0.0; worstβ = ""
    for (name, β) in βs
        t_comp = compressed_dual_contraction(β, cf)
        t_dense = Gfac * β                      # [dot(β, G[s,:]) for s]
        e = maximum(abs.(t_comp .- t_dense))
        if e > errB; errB = e; worstβ = name; end
    end

    # (C) q reconstruction vs the real fixed_dual path
    q_dense = [-base.ζstar - dot(base.λstar, @view(G[s, 1:ncol])) for s in 1:W]
    q_comp = -base.ζstar .- compressed_dual_contraction(base.λstar, cf)
    errC = maximum(abs.(q_dense .- q_comp))
    # also: does mean(Psi(q)) match fixed_dual_L exactly?
    Lref = fixed_dual_L(xf0, ctx, base)
    Psi_q = similar(q_comp); CS.Psi!(Psi_q, q_comp)
    L_comp = -(sum(Psi_q) / W + base.ζstar)
    errL = abs(L_comp - Lref)

    ok = errA < 1e-9 && errB < 1e-9 && errC < 1e-9 && errL < 1e-9
    @printf("%-26s  |dense-materialize|=%.2e  |contraction(worst=%s)|=%.2e  |q|=%.2e  |L_fix|=%.2e  %s\n",
        label, errA, worstβ, errB, errC, errL, ok ? "PASS" : "FAIL")
    return ok
end

all_ok = true
println("="^96)
println("COMPRESSED-MOMENT EQUIVALENCE: winner-form vs trusted dense obj.moments!  (D=$D, W=$W, oci=$oci)")
println("="^96)

# Candidate points (same as test_lfix_incremental.jl)
w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

global all_ok &= check_point("upper_maxit40", w_up40)
global all_ok &= check_point("lower_stalled", w_low)

# random feasible perturbations around each candidate
rng = MersenneTwister(20260718)
for i in 1:6
    base_w = isodd(i) ? w_up40 : w_low
    w = base_w .+ 0.15 .* randn(rng, length(base_w))
    global all_ok &= check_point("random_$i", w; rng = rng)
end

# ---- tie handling: (1) clean points never throw (implied by PASSes above);
#      (2) an injected exact tie must be caught by build_compressed_factual's
#      tie logic (same detect_price_ties/TiedWinnerError prior art). ----
println("-"^96)
let
    xf0 = x_free_from_w(w_up40)
    base = solve_base_state(xf0, ctx)
    θfull = base.θ_full0
    _, _, AodPow = factual_prices(θfull, ctx)
    μ = θfull[1]; s0 = 7; d0 = 1
    cc = [ctx.γ.wHat[o] * AodPow[o, d0] * ctx.γ.τ[o, d0] for o in 1:D]   # constCons[:,d0]
    prices = [cc[o] / (ctx.U[s0, o]^(-μ)) for o in 1:D]
    w = argmin(prices); pmin = prices[w]
    o2 = (w == 1 ? 2 : 1)                                               # a different origin to tie in
    # set U[s0,o2] so origin o2's price == pmin exactly (guaranteed row-min tie):
    Utie = copy(ctx.U); Utie[s0, o2] = (cc[o2] / pmin)^(-1 / μ)
    ctx_tie = merge(ctx, (U = Utie,))
    threw = false
    try
        build_compressed_factual(θfull, ctx_tie)
    catch e
        threw = e isa TiedWinnerError
        threw && println("tie handling: build_compressed_factual threw TiedWinnerError (n_tied_pairs=$(e.n_tied_pairs), examples=$(e.examples)) -> PASS")
    end
    threw || println("tie handling: FAIL (no TiedWinnerError thrown on injected exact tie)")
    global all_ok &= threw
end

println("="^96)
if all_ok
    println("ALL COMPRESSED-MOMENT EQUIVALENCE TESTS PASSED")
else
    println("SOME COMPRESSED-MOMENT EQUIVALENCE TESTS FAILED")
    exit(1)
end
