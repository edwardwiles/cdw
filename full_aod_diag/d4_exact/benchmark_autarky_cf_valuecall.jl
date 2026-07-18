# Focused, INTERLEAVED value-call comparison on a SINGLE ctx (swap obj.moments!
# binding between reps) so the only difference is the moment build, not two
# separate KNITRO-state ctx objects. Confirms the value call is inner-solve
# dominated and CF specialization is immaterial there (and bit-identical).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "moments_fast.jl"))
include(joinpath(@__DIR__, "autarky_cf.jl"))
using Statistics, Printf

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
wpt = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf = x_free_from_w(wpt)

orig_moments = ctx.obj.moments!                       # generic production binding
pc = MuSigmaPowCache(ctx.obj.U, ctx.obj.γ.Uσ)
spec_moments = (K,G,θ,U,o) -> EK_moments_gammanorm_directgp_autarkyCF!(K,G,θ,U,o; pow_cache=pc)

# warm both
ctx.obj.moments! = orig_moments; evaluate_fullA(xf, ctx; use_cache=false, warm=false)
ctx.obj.moments! = spec_moments; evaluate_fullA(xf, ctx; use_cache=false, warm=false)

Nc = 60
tg = Float64[]; ts = Float64[]
dg = 0.0
for _ in 1:Nc
    ctx.obj.moments! = orig_moments
    GC.gc(false); push!(tg, @elapsed (rg = evaluate_fullA(xf, ctx; use_cache=false, warm=false)))
    ctx.obj.moments! = spec_moments
    GC.gc(false); t0 = time(); rs = evaluate_fullA(xf, ctx; use_cache=false, warm=false); push!(ts, time()-t0)
end
ctx.obj.moments! = orig_moments
@printf("INTERLEAVED value-call (single ctx, N=%d): generic median=%.3f ms  autarkyCF median=%.3f ms  (%.3fx)\n",
        Nc, median(tg)*1e3, median(ts)*1e3, median(tg)/median(ts))
@printf("  generic  min=%.3f ms   autarkyCF min=%.3f ms\n", minimum(tg)*1e3, minimum(ts)*1e3)
println("DONE")
