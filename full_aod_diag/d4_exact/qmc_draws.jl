# ============================================================================
# Continuation 10, Phase 7: draw generators for the QMC precision comparison.
# All three paths produce a W x D Exp(1) matrix (D=20 under UoModel=1, the
# fixed AD_PARAMS default -- see qmc_context_real_d20.jl's header comment for
# why D not D^2) via the SAME inverse-CDF transform genExpRands! uses
# (exp_from_uniform01, defined in qmc_context_real_d20.jl) -- only the
# underlying [0,1) source differs:
#   - pseudorandom_U:  Julia's default RNG (MersenneTwister via Random.rand!),
#                      i.e. exactly what prepare_cc/drawU.jl's genExpRands!
#                      does, called directly here so we can control/vary the
#                      seed per replicate for a fair scramble-vs-baseline
#                      comparison (production itself always uses seedU=888).
#   - halton_U:        cc_algo/rhalton.jl's scrambled Halton sequence
#                      (validated in c10_phase7_rhalton_validate.jl),
#                      singleseed varies the digit-scrambling permutations
#                      per dimension -> independent replicates.
#   - sobol_U:         Sobol.jl's SobolSeq (deterministic low-discrepancy
#                      base sequence) + an independent Cranley-Patterson
#                      random shift (mod 1) per replicate -- Sobol.jl (the
#                      JuliaMath package available in this environment) does
#                      NOT implement digital/Owen scrambling itself, so we
#                      randomize via the standard, simple RQMC shift
#                      technique (Cranley & Patterson 1976) instead, which is
#                      enough to (a) decorrelate replicates and (b) give an
#                      unbiased randomized-QMC estimator; it is a strictly
#                      weaker randomization than rhalton.jl's per-digit
#                      permutation scrambling and this is flagged explicitly
#                      in the report, not glossed over.
# ============================================================================
using Random
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "rhalton.jl"))

"Exactly prepare_cc/drawU.jl's genExpRands! path (UoModel=1 => sizeU=D), with an explicit, variable seed for replicate control."
function pseudorandom_U(W::Int, D::Int; seed::Int)
    Random.seed!(seed)
    U01 = rand(W, D)
    return exp_from_uniform01(U01)
end

"Scrambled Halton via cc_algo/rhalton.jl, transformed through the model's real Exp(1) inverse-CDF."
function halton_U(W::Int, D::Int; seed::Int)
    U01 = rhalton(W, D; singleseed = seed)
    return exp_from_uniform01(U01)
end

"Sobol' (Sobol.jl SobolSeq) + Cranley-Patterson random shift (mod 1) for replicate randomization, transformed through the model's real Exp(1) inverse-CDF."
function sobol_U(W::Int, D::Int; seed::Int)
    s = SobolSeq(D)
    base = Matrix{Float64}(undef, W, D)
    for i in 1:W
        base[i, :] = next!(s)
    end
    Random.seed!(seed)
    shift = rand(D)
    U01 = mod.(base .+ reshape(shift, 1, D), 1.0)
    return exp_from_uniform01(U01)
end
