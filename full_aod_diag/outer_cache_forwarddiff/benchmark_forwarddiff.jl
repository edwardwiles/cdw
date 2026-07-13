# ============================================================================
# ForwardDiff optimization for Method B (direct scalar gradient of the
# envelope, mega-prompt §10-11). Builds on the ALREADY-VALIDATED envelope
# formula in full_aod_diag/ad_benchmark/derivative_core.jl (do not re-derive
# or re-validate correctness here — ad_benchmark/correctness_results.csv
# already proved Method B matches production to relerr ~1e-15 at 4 points).
# This script's job is purely: what is the FASTEST correct configuration?
#
# Sweeps, at all 4 frozen D=4 benchmark points (A-D, ad_benchmark/benchmark_points.jld2):
#   1. baseline: ForwardDiff.gradient(closure, θ) -- fresh closure, auto chunk,
#      allocating output (this is what derivative_methods.jl's method_B and,
#      by extension, any naive Method-B production port would do).
#   2. cached GradientConfig + ForwardDiff.gradient! into a preallocated buffer,
#      auto chunk size (ForwardDiff.Chunk(θ)).
#   3. same, with explicit chunk sizes 1,2,3,4,6,9,18,23 (l=23 at D=4).
#   4. a NON-closure formulation (ctx passed as a second arg via a stored
#      GradientConfig -- avoids re-capturing ctx in a new closure every call).
#
# Also runs @code_warntype on envelope_scalar_div_ctx to check type stability,
# and reports allocations via BenchmarkTools.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/benchmark_forwarddiff.jl
# ============================================================================
using BenchmarkTools, Printf, InteractiveUtils
const ADB = joinpath(dirname(@__DIR__), "ad_benchmark")
include(joinpath(ADB, "setup_context.jl"))   # brings in Parameters/ForwardDiff/JLD2/CS/etc. and moments_gammanorm.jl
include(joinpath(ADB, "derivative_core.jl"))

const OUT = @__DIR__

# ---- rebuild the D=4 γ_d≡1+direct-γ' context (fast: no KNITRO calls needed for U/γ) ----
so, pp = build_ad_context()
@unpack U, γ, outer_constr_index, nTotalMoments = pp

pts = JLD2.load(joinpath(ADB, "benchmark_points.jld2"))["points"]
l = length(pts[:A].θ)
println(">>> l (theta length) = ", l)

function make_ctx(p)
    (U = U, γobj = γ, λ = p.λ, arg1 = p.arg1, d = nTotalMoments, outer_constr_index = outer_constr_index)
end

# ---- type stability check ----
println("\n>>> @code_warntype envelope_scalar_div_ctx (point A) — looking for red/Union/Any flags")
ctxA = make_ctx(pts[:A])
io = IOBuffer()
code_warntype(io, envelope_scalar_div_ctx, (Vector{Float64}, typeof(ctxA)))
warntype_str = String(take!(io))
n_union = count("Union", warntype_str); n_any = count("::Any", warntype_str)
println("Union{} occurrences: ", n_union, "   ::Any occurrences: ", n_any,
        n_union == 0 && n_any == 0 ? "  -> fully concrete/type-stable" : "  -> INSPECT (see warntype_envelope.txt)")
open(joinpath(OUT, "warntype_envelope.txt"), "w") do io2; write(io2, warntype_str); end

# non-closure: ctx baked into a tag-free functor struct so ForwardDiff sees a stable concrete type
struct EnvelopeFunctor{C}
    ctx::C
end
(f::EnvelopeFunctor)(θ) = envelope_scalar_div_ctx(θ, f.ctx)

results = NamedTuple[]

for (label, p) in pts
    ctx = make_ctx(p)
    θ = p.θ
    g = zeros(l)
    f = EnvelopeFunctor(ctx)

    g_ref = ForwardDiff.gradient(θ2 -> envelope_scalar_div_ctx(θ2, ctx), θ)   # correctness reference (already validated vs production elsewhere)

    # 1. baseline: FRESH closure, FRESH (implicit) config, allocating output every call
    #    -- this is exactly what derivative_methods.jl's method_B does.
    t_base = @belapsed ForwardDiff.gradient(θ2 -> envelope_scalar_div_ctx(θ2, $ctx), $θ)
    alloc_base = @allocated ForwardDiff.gradient(θ2 -> envelope_scalar_div_ctx(θ2, ctx), θ)

    # 2. cached GradientConfig built from a FIXED closure (same object every call), auto chunk,
    #    preallocated output buffer -- the config's Tag must match the function it's later
    #    called with, so we must reuse the SAME closure object, not rebuild one per call.
    fclosure = θ2 -> envelope_scalar_div_ctx(θ2, ctx)
    cfg_auto = ForwardDiff.GradientConfig(fclosure, θ)
    t_cached = @belapsed ForwardDiff.gradient!($g, $fclosure, $θ, $cfg_auto)
    alloc_cached = @allocated ForwardDiff.gradient!(g, fclosure, θ, cfg_auto)
    @assert isapprox(g, g_ref; rtol=1e-12) "cached-config gradient mismatch at $label"

    # 3. functor (no closure at all -- ctx is a type parameter, not a captured variable) + cached
    #    config, auto chunk, preallocated output buffer.
    cfg_f = ForwardDiff.GradientConfig(f, θ)
    t_functor = @belapsed ForwardDiff.gradient!($g, $f, $θ, $cfg_f)
    alloc_functor = @allocated ForwardDiff.gradient!(g, f, θ, cfg_f)
    @assert isapprox(g, g_ref; rtol=1e-12) "functor gradient mismatch at $label"

    push!(results, (point=label, config="1_baseline_closure_autochunk", time_s=t_base, alloc_bytes=alloc_base))
    push!(results, (point=label, config="2_cached_config_closure_autochunk", time_s=t_cached, alloc_bytes=alloc_cached))
    push!(results, (point=label, config="3_functor_cached_config_autochunk", time_s=t_functor, alloc_bytes=alloc_functor))

    # 4. explicit chunk sweep (functor form, cached config, preallocated g)
    for c in (1, 2, 3, 4, 6, 9, l)
        cfg_c = ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(c))
        t_c = @belapsed ForwardDiff.gradient!($g, $f, $θ, $cfg_c)
        alloc_c = @allocated ForwardDiff.gradient!(g, f, θ, cfg_c)
        @assert isapprox(g, g_ref; rtol=1e-12) "chunk=$c gradient mismatch at $label"
        push!(results, (point=label, config="4_functor_chunk$(c)", time_s=t_c, alloc_bytes=alloc_c))
    end

    println(">>> point ", label, " done")
    flush(stdout)
end

open(joinpath(OUT, "forwarddiff_benchmark.csv"), "w") do io3
    println(io3, "point,config,time_s,alloc_bytes")
    for r in results
        println(io3, join((r.point, r.config, r.time_s, r.alloc_bytes), ","))
    end
end

println("\n================ SUMMARY (mean over points) ================")
by_cfg = Dict{String, Vector{Float64}}()
by_cfg_alloc = Dict{String, Vector{Float64}}()
for r in results
    push!(get!(by_cfg, r.config, Float64[]), r.time_s)
    push!(get!(by_cfg_alloc, r.config, Float64[]), r.alloc_bytes)
end
for cfg in sort(collect(keys(by_cfg)))
    t = by_cfg[cfg]; a = by_cfg_alloc[cfg]
    @printf("%-32s  mean_time=%.6fs  mean_alloc=%.0f bytes\n", cfg, sum(t)/length(t), sum(a)/length(a))
end
fastest = sort(collect(keys(by_cfg)), by = k -> sum(by_cfg[k])/length(by_cfg[k]))[1]
println("\nFASTEST CONFIG: ", fastest)
println("Wrote: ", joinpath(OUT, "forwarddiff_benchmark.csv"), " and warntype_envelope.txt")
