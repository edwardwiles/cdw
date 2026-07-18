# ============================================================================
# Benchmark: V2 cached-base CF path vs autarky_cf.jl (v1) vs generic
# hFunctionCounter!. Measures (median of N reps, GC between reps):
#   (A)  isolated CF component with UσPow ALREADY materialized (as in a full
#        build, where the factual block materializes it anyway): gen vs v1 vs v2
#        — expected near-flat (v2's only edge is divide→multiply).
#   (A') CF column built FROM RAW Uσ (no pre-materialized UσPow): the CF-only
#        sweep scenario. v1/generic pay a W-length σ-power per call; v2 hits its
#        once-built reciprocal cache. This is v2's genuine niche.
#   (B)  full moment build: prod / prod+pc / v1 / v1+pc / v2 / v2+pc.
#   (C)  exact value-call: evaluate_fullA generic vs v2-wired (equivalence+time).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
include(joinpath(@__DIR__, "autarky_cf.jl"))
include(joinpath(@__DIR__, "autarky_cf_v2.jl"))
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

Aod_θ = reshape(θ[ctx.Aod_offset+1:ctx.Aod_offset+D2], (D,D))
lambda = reshape(γo.P,(D,D))'
Aod = Aod_θ .* γo.cHat .* (((γo.wHat.*γo.τ)./(γo.wHat[1,1].*γo.τ[1,:]')).^(1/μ)) .* (lambda./lambda[1,:]')
AodPow = (Aod ./ γo.cHat).^(-μ)
UσPowm = γo.Uσ .^ (-μ)
Gbuf = zeros(W, ctx.obj.d); Kbuf = zeros(W)
γ_prime = ones(D); γ_prime[bi] = θ[3+D]
wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
UPowm = ctx.U .^ (-μ)

# pre-build the v2 base once (amortized: this is the once-per-ctx build)
base = AutarkyCFBase(W)
o1_v2 = (γo.indicators.UoModel == 1) ? bi : bi + (bi-1)*D
get_autarky_cf_base!(base, γo.Uσ, μ, o1_v2)

N = 200
println("D=$D  W=$W  threads=$(Threads.nthreads())  N=$N reps (median)\n")

# ---------- (A) isolated CF component, UσPow pre-materialized ----------
# All hot kernels behind function barriers taking CONCRETE-TYPED args, so timings
# reflect the kernel, not untyped-global dispatch (which dominates at ~0.01 ms).
function gen_cf(Kbuf, Gbuf, UPowm, UσPowm, wPrime, τPrime, σ, γ_prime, AodPow, LPrime, bi)
    hFunctionCounter!(Kbuf, Gbuf, UPowm, UσPowm, wPrime, τPrime, σ, γ_prime, AodPow, LPrime, 1, bi, 1)
end
function v1_cf(obj, Gbuf, UσPowm, AodPow, σ, γp_bi, D2)
    cf_num, cf_denom, o1 = autarky_cf_scalars(obj, AodPow, σ, γp_bi)
    fill_autarky_cf_column!(Gbuf, UσPowm, cf_num, cf_denom, o1)
end
function v2_cf(obj, Gbuf, inv_uσ::Vector{Float64}, AodPow, σ, γp_bi, D2)
    cf_num, cf_denom, _ = autarky_cf_scalars(obj, AodPow, σ, γp_bi)
    fill_autarky_cf_column_v2!(Gbuf, inv_uσ, cf_num, cf_denom, D2+1)
end
let obj = ctx.obj, γp_bi = θ[3+D], τP = γo.τPrime, LP = γo.LPrime, invv = base.inv_uσ
    gen_cf(Kbuf, Gbuf, UPowm, UσPowm, wPrime, τP, σ, γ_prime, AodPow, LP, bi)
    v1_cf(obj, Gbuf, UσPowm, AodPow, σ, γp_bi, D2); v2_cf(obj, Gbuf, invv, AodPow, σ, γp_bi, D2)
    global tA_gen = med(() -> gen_cf(Kbuf, Gbuf, UPowm, UσPowm, wPrime, τP, σ, γ_prime, AodPow, LP, bi), N)
    global tA_v1  = med(() -> v1_cf(obj, Gbuf, UσPowm, AodPow, σ, γp_bi, D2), N)
    global tA_v2  = med(() -> v2_cf(obj, Gbuf, invv, AodPow, σ, γp_bi, D2), N)
    global aA_gen = @allocated gen_cf(Kbuf, Gbuf, UPowm, UσPowm, wPrime, τP, σ, γ_prime, AodPow, LP, bi)
    global aA_v1  = @allocated v1_cf(obj, Gbuf, UσPowm, AodPow, σ, γp_bi, D2)
    global aA_v2  = @allocated v2_cf(obj, Gbuf, invv, AodPow, σ, γp_bi, D2)
end
println("(A) ISOLATED CF COMPONENT, UσPow pre-materialized (full-build context)")
@printf("    generic hFunctionCounter!(autarky) : %s ms   %d bytes\n", ms(tA_gen), aA_gen)
@printf("    v1 autarky_cf.jl (scalars+divide)  : %s ms   %d bytes   (%.2fx vs gen)\n", ms(tA_v1), aA_v1, tA_gen/tA_v1)
@printf("    v2 cached-base   (scalars+multiply): %s ms   %d bytes   (%.2fx vs gen, %.2fx vs v1)\n\n",
        ms(tA_v2), aA_v2, tA_gen/tA_v2, tA_v1/tA_v2)

# ---------- (A') CF column FROM RAW Uσ (CF-only sweep; no pre-materialized UσPow) ----------
Uσ = γo.Uσ; ucol = zeros(W)
function v1_scratch(obj, Gbuf, Uσ, ucol, AodPow, σ, μ, γp_bi, D2, W)
    cf_num, cf_denom, o1 = autarky_cf_scalars(obj, AodPow, σ, γp_bi)
    @inbounds @simd for s in 1:W; ucol[s] = Uσ[s, o1]^(-μ); end   # per-call σ-power
    @inbounds @simd for s in 1:W; Gbuf[s, D2+1] = cf_num / ucol[s] - cf_denom; end
end
function v2_scratch(obj, Gbuf, base::AutarkyCFBase, Uσ, AodPow, σ, μ, γp_bi, o1_v2, D2)
    cf_num, cf_denom, _ = autarky_cf_scalars(obj, AodPow, σ, γp_bi)
    inv_uσ = get_autarky_cf_base!(base, Uσ, μ, o1_v2)              # cache hit (no power)
    fill_autarky_cf_column_v2!(Gbuf, inv_uσ, cf_num, cf_denom, D2+1)
end
let obj = ctx.obj, γp_bi = θ[3+D]
    v1_scratch(obj, Gbuf, Uσ, ucol, AodPow, σ, μ, γp_bi, D2, W)
    v2_scratch(obj, Gbuf, base, Uσ, AodPow, σ, μ, γp_bi, o1_v2, D2)
    global tAp_v1 = med(() -> v1_scratch(obj, Gbuf, Uσ, ucol, AodPow, σ, μ, γp_bi, D2, W), N)
    global tAp_v2 = med(() -> v2_scratch(obj, Gbuf, base, Uσ, AodPow, σ, μ, γp_bi, o1_v2, D2), N)
end
println("(A') CF COLUMN FROM RAW Uσ (CF-only sweep: v1 pays per-call σ-power, v2 hits cache)")
@printf("    v1 from-scratch (power + divide)   : %s ms\n", ms(tAp_v1))
@printf("    v2 cached-base  (multiply only)    : %s ms   (%.2fx vs v1)\n\n", ms(tAp_v2), tAp_v1/tAp_v2)

# ---------- (B) full moment build ----------
pc = MuSigmaPowCache(ctx.U, γo.Uσ)
base_b = AutarkyCFBase(W)
prod_build()   = EK_moments_gammanorm_directgp!(Kbuf, Gbuf, θ, ctx.U, ctx.obj)
prodpc_build() = EK_moments_gammanorm_directgp_fast!(Kbuf, Gbuf, θ, ctx.U, ctx.obj, pc)
v1_build()     = EK_moments_gammanorm_directgp_autarkyCF!(Kbuf, Gbuf, θ, ctx.U, ctx.obj)
v1pc_build()   = EK_moments_gammanorm_directgp_autarkyCF!(Kbuf, Gbuf, θ, ctx.U, ctx.obj; pow_cache = pc)
v2_build()     = EK_moments_gammanorm_directgp_autarkyCF_v2!(Kbuf, Gbuf, θ, ctx.U, ctx.obj; base = base_b)
v2pc_build()   = EK_moments_gammanorm_directgp_autarkyCF_v2!(Kbuf, Gbuf, θ, ctx.U, ctx.obj; base = base_b, pow_cache = pc)
for f in (prod_build, prodpc_build, v1_build, v1pc_build, v2_build, v2pc_build); f(); f(); end
println("(B) FULL MOMENT BUILD (obj.moments!)")
baseT = med(prod_build, N)
for (nm,f) in (("prod (generic CF, recompute pow)",prod_build),
               ("prod + pow_cache",prodpc_build),
               ("v1 autarkyCF (recompute pow)",v1_build),
               ("v1 autarkyCF + pow_cache",v1pc_build),
               ("v2 cached-base (recompute pow)",v2_build),
               ("v2 cached-base + pow_cache",v2pc_build))
    t = med(f, N); a = @allocated f()
    @printf("    %-34s %s ms  %8d bytes   (%.2fx vs prod)\n", nm, ms(t), a, baseT/t)
end
println()

# ---------- (C) exact value-call ----------
ctx_spec = d4_exact_setup(find_smallest = true)
pc2 = MuSigmaPowCache(ctx_spec.obj.U, ctx_spec.obj.γ.Uσ)
base_c = enable_autarky_cf_v2!(ctx_spec; pow_cache = pc2)
val_gen()  = evaluate_fullA(xf, ctx; use_cache=false, warm=false)
val_v2()   = evaluate_fullA(xf, ctx_spec; use_cache=false, warm=false)
r0 = val_gen(); r1 = val_v2()
@printf("(C) EXACT VALUE-CALL equivalence: Delta_dual gen=%.15g v2=%.15g  |diff|=%.2e\n",
        r0.Delta_dual, r1.Delta_dual, abs(r0.Delta_dual - r1.Delta_dual))
Nc = 40
tC_gen = med(val_gen, Nc); tC_v2 = med(val_v2, Nc)
@printf("    evaluate_fullA generic : %s ms\n", ms(tC_gen))
@printf("    evaluate_fullA v2      : %s ms   (%.2fx)\n", ms(tC_v2), tC_gen/tC_v2)
@printf("    v2 base: n_build=%d n_reuse=%d\n\n", base_c.n_build, base_c.n_reuse)

println("DONE")
