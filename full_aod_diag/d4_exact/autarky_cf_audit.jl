# Diagnostic: audit the focal autarky counterfactual price-index moment.
# Confirms data conventions and the exact factual-vs-counterfactual domestic scalar.
include(joinpath(@__DIR__, "context.jl"))
using Printf

ctx = d4_exact_setup()
γo = ctx.γ; D = ctx.D; bi = ctx.bi; σ = ctx.σ
θ = copy(ctx.θ0_up)
μ = θ[1]

ind = ctx.obj.γ.indicators
println("counterType = ", ind.counterType, "   UoModel = ", ind.UoModel,
        "   OuterScaling = ", ind.OuterScaling, "   θConstant = ", ind.θConstant)
println("baseIndex bi = ", bi, "   σ = ", σ, "   μ = ", μ, "   oci = ", ctx.obj.outer_constr_index)

# --- data conventions used by the counterfactual domestic scalar ---
wHat = γo.wHat
wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
@printf("wHat[bi]      = %.15g\n", wHat[bi])
@printf("wPrime[bi]    = %.15g   (autarky focal wage, inserted as 1)\n", wPrime[bi])
@printf("τ[bi,bi]      = %.15g\n", γo.τ[bi,bi])
@printf("τPrime[bi,bi] = %.15g\n", γo.τPrime[bi,bi])
@printf("L[bi]         = %.15g\n", γo.L[bi])
@printf("LPrime[bi]    = %.15g\n", γo.LPrime[bi])
γ_prime_bi = θ[3+D]
@printf("γ_prime[bi]   = θ[3+D] = %.15g   (FREE variable)\n", γ_prime_bi)
@printf("γ[bi]         = 1 (normalization)\n")

# --- AodPow (factual, reused unchanged by the CF) ---
Aod_θ = reshape(θ[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D,D))
lambda = reshape(γo.P,(D,D))'
Aod = Aod_θ .* γo.cHat .* (((γo.wHat.*γo.τ)./(γo.wHat[1,1].*γo.τ[1,:]')).^(1/μ)) .* (lambda./lambda[1,:]')
AodPow = (Aod ./ γo.cHat).^(-μ)
@printf("AodPow[bi,bi] = %.15g\n", AodPow[bi,bi])

# --- factual domestic scalar (o=bi,d=bi) as hFunction! computes it ---
constConsσ_fac_bibi = wHat[bi]^(1-σ) * (AodPow[bi,bi]*γo.τ[bi,bi])^(1-σ)
denom_fac_bi        = 1.0^σ * (wHat[bi]*γo.L[bi])           # γ[bi]≡1
# --- counterfactual domestic scalar (autarky) as hFunctionCounter! computes it ---
constConsσ_cf_bibi  = wPrime[bi]^(1-σ) * (AodPow[bi,bi]*γo.τPrime[bi,bi])^(1-σ)
denom_cf_bi         = γ_prime_bi^σ * (wPrime[bi]*γo.LPrime[bi])

println("\n--- factual vs counterfactual DOMESTIC scalars ---")
@printf("constConsσ_factual[bi,bi] = %.15g\n", constConsσ_fac_bibi)
@printf("constConsσ_counter[bi,bi] = %.15g\n", constConsσ_cf_bibi)
@printf("ratio counter/factual (numerator scale) = %.15g\n", constConsσ_cf_bibi/constConsσ_fac_bibi)
@printf("denom_factual[bi]  = %.15g\n", denom_fac_bi)
@printf("denom_counter[bi]  = %.15g\n", denom_cf_bi)

# closed-form ratio: since both share the same AodPow[bi,bi] and draw column bi,
# counter/factual numerator = (wPrime/wHat)^(1-σ) * (τPrime/τ)^(1-σ)
ratio_pred = (wPrime[bi]/wHat[bi])^(1-σ) * (γo.τPrime[bi,bi]/γo.τ[bi,bi])^(1-σ)
@printf("predicted num ratio (wP/w)^(1-σ)(τP/τ)^(1-σ) = %.15g  (match=%g)\n",
        ratio_pred, abs(ratio_pred - constConsσ_cf_bibi/constConsσ_fac_bibi))

# --- confirm the CURRENT production CF column matches this scalar form ---
# Build moments via the production path, extract column D^2+1.
K = zeros(size(ctx.U,1), ctx.nTotalMoments)  # placeholder sizing
nT = ctx.nTotalMoments
Gp = zeros(size(ctx.U,1), nT)
Kp = zeros(size(ctx.U,1))
ctx.obj.moments!(Kp, Gp, θ, ctx.U, ctx.obj)
cf_col_prod = Gp[:, D^2+1]

# reconstruct from scalar form: constConsσ_cf/UσPow[:,bi] - denom_cf, then /gammafac, *SW
UσPow_bi = γo.Uσ[:,bi].^(-μ)
gammafac = gamma(μ*(1-σ)+1)
SW = γo.SamplingWeights[1:size(ctx.U,1)]
cf_col_recon = SW .* ((constConsσ_cf_bibi ./ UσPow_bi .- denom_cf_bi) ./ gammafac)
@printf("\nmax|CF_prod - CF_scalarform| = %.3e\n", maximum(abs.(cf_col_prod .- cf_col_recon)))

# also verify: does the FACTUAL domestic value share the same draw term 1/UσPow[:,bi]?
# factual domestic CES value (pre-winner) = constConsσ_fac_bibi / UσPow[:,bi]
# so CF = (constConsσ_cf/constConsσ_fac) * factual_domestic_value - denom_cf, all /gammafac*SW
fac_dom_val = constConsσ_fac_bibi ./ UσPow_bi
cf_from_fac = SW .* (((constConsσ_cf_bibi/constConsσ_fac_bibi) .* fac_dom_val .- denom_cf_bi) ./ gammafac)
@printf("max|CF_prod - CF_via_factual_reuse| = %.3e\n", maximum(abs.(cf_col_prod .- cf_from_fac)))

println("\nDONE")
