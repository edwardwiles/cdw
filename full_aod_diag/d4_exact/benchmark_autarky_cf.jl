# ============================================================================
# Benchmark: specialized focal-autarky CF path vs generic hFunctionCounter!.
# Measures (median of N reps, GC between reps, JULIA_NUM_THREADS from env):
#   (A) isolated CF-construction component: generic hFunctionCounter! (autarky)
#       vs specialized autarky_cf_scalars + fill_autarky_cf_column!
#   (B) full moment build: production vs specialized, each with/without pow_cache
#   (C) exact value-call: evaluate_fullA with generic vs specialized obj.moments!
#   (D) L_fix base-state construction time (build_lfix_base_cache)
# Allocations via @allocated on a single warmed call.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
include(joinpath(@__DIR__, "autarky_cf.jl"))
using Statistics, Printf

med(f, N) = median([(GC.gc(false); @elapsed f()) for _ in 1:N])
ms(x) = @sprintf("%.4f", x*1e3)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
γo = ctx.γ; bi = ctx.bi; σ = ctx.σ
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

wpt = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf = x_free_from_w(wpt)
θ = CS.reconstruct_full(xf, ctx.m)
μ = θ[1]

# precompute AodPow + UσPow for the isolated component test
Aod_θ = reshape(θ[ctx.Aod_offset+1:ctx.Aod_offset+D2], (D,D))
lambda = reshape(γo.P,(D,D))'
Aod = Aod_θ .* γo.cHat .* (((γo.wHat.*γo.τ)./(γo.wHat[1,1].*γo.τ[1,:]')).^(1/μ)) .* (lambda./lambda[1,:]')
AodPow = (Aod ./ γo.cHat).^(-μ)
UPowm = ctx.U .^ (-μ); UσPowm = γo.Uσ .^ (-μ)
γ = ones(D); γ_prime = ones(D); γ_prime[bi] = θ[3+D]
wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
Gbuf = zeros(W, ctx.obj.d); Kbuf = zeros(W)

N = 200
println("D=$D  W=$W  threads=$(Threads.nthreads())  N=$N reps (median)\n")

# ---------- (A) isolated CF component ----------
# generic: exactly the call moments_gammanorm makes (autarky branch writes G[:,D2+1])
gen_cf() = hFunctionCounter!(Kbuf, Gbuf, UPowm, UσPowm, wPrime, γo.τPrime, σ, γ_prime, AodPow, γo.LPrime, 1, bi, 1)
# specialized
function spec_cf()
    cf_num, cf_denom, o1 = autarky_cf_scalars(ctx.obj, AodPow, σ, θ[3+D])
    fill_autarky_cf_column!(Gbuf, UσPowm, cf_num, cf_denom, o1)
end
gen_cf(); spec_cf()  # warm
tA_gen = med(gen_cf, N); tA_spec = med(spec_cf, N)
aA_gen = @allocated gen_cf(); aA_spec = @allocated spec_cf()
println("(A) ISOLATED CF COMPONENT (writes G[:,D^2+1] only)")
@printf("    generic hFunctionCounter!(autarky) : %s ms   %d bytes\n", ms(tA_gen), aA_gen)
@printf("    specialized scalars+broadcast      : %s ms   %d bytes\n", ms(tA_spec), aA_spec)
@printf("    speedup = %.2fx   alloc reduction = %d bytes\n\n", tA_gen/tA_spec, aA_gen - aA_spec)

# ---------- (B) full moment build, 4 configs ----------
pc = MuSigmaPowCache(ctx.U, γo.Uσ)
prod_build()          = EK_moments_gammanorm_directgp!(Kbuf, Gbuf, θ, ctx.U, ctx.obj)
prodpc_build()        = EK_moments_gammanorm_directgp_fast!(Kbuf, Gbuf, θ, ctx.U, ctx.obj, pc)
spec_build()          = EK_moments_gammanorm_directgp_autarkyCF!(Kbuf, Gbuf, θ, ctx.U, ctx.obj)
specpc_build()        = EK_moments_gammanorm_directgp_autarkyCF!(Kbuf, Gbuf, θ, ctx.U, ctx.obj; pow_cache = pc)
for f in (prod_build, prodpc_build, spec_build, specpc_build); f(); f(); end
tB = Dict(); aB = Dict()
for (nm,f) in (("prod (generic CF, recompute pow)",prod_build),
               ("prod + pow_cache",prodpc_build),
               ("autarkyCF (recompute pow)",spec_build),
               ("autarkyCF + pow_cache",specpc_build))
    tB[nm] = med(f, N); aB[nm] = @allocated f()
end
println("(B) FULL MOMENT BUILD (obj.moments!)")
base = tB["prod (generic CF, recompute pow)"]
for nm in ("prod (generic CF, recompute pow)","prod + pow_cache","autarkyCF (recompute pow)","autarkyCF + pow_cache")
    @printf("    %-34s %s ms  %8d bytes   (%.2fx vs prod)\n", nm, ms(tB[nm]), aB[nm], base/tB[nm])
end
println()

# ---------- (C) exact value-call (evaluate_fullA) ----------
# generic obj.moments! is the default; make a copy of ctx for the specialized binding
ctx_spec = d4_exact_setup(find_smallest = true)
pc2 = MuSigmaPowCache(ctx_spec.obj.U, ctx_spec.obj.γ.Uσ)
enable_autarky_cf!(ctx_spec; pow_cache = pc2)
val_gen()  = evaluate_fullA(xf, ctx; use_cache=false, warm=false)
val_spec() = evaluate_fullA(xf, ctx_spec; use_cache=false, warm=false)
r0 = val_gen(); r1 = val_spec()
@printf("(C) EXACT VALUE-CALL equivalence: Delta_dual gen=%.15g spec=%.15g  |diff|=%.2e\n",
        r0.Delta_dual, r1.Delta_dual, abs(r0.Delta_dual - r1.Delta_dual))
Nc = 40
tC_gen = med(val_gen, Nc); tC_spec = med(val_spec, Nc)
@printf("    evaluate_fullA generic  : %s ms\n", ms(tC_gen))
@printf("    evaluate_fullA autarkyCF: %s ms   (%.2fx)\n\n", ms(tC_spec), tC_gen/tC_spec)

println("DONE")
