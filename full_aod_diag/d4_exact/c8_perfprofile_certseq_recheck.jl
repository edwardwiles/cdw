# ============================================================================
# Targeted recheck of c8_perfprofile_harness.jl's Part C3(b) embedded
# certificate-sequence result: the first run showed a 442ms outlier on
# point 1 of the winner_cache_mode=:certificate K=10 sequence (vs ~46ms for
# the :none baseline at the SAME point, and vs the ISOLATED
# winner_value_update! call's clean 42ms cold-build cost) -- suspected GC
# pause from the immediately-preceding Part B (W=80000, N=25x2 reps)
# leftover garbage, not a real property of the certificate. Forces GC.gc()
# before this test to check. Standalone, minimal (skips Parts A/B/D).
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Statistics, Printf, Random, LinearAlgebra

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

ctx4 = d4_exact_setup(find_smallest = true)
pe4 = build_pivot_elimination(ctx4)
D4 = ctx4.D

const W_UPPER = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf_upper = x_free_from_w(W_UPPER, pe4)

println("Warming up...")
base0_4 = solve_base_state(xf_upper, ctx4)
composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = true, winner_cache_mode = :none)
wc0 = PersistentWinnerCache()
composite_gradient_at_fast(xf_upper, ctx4, pe4; base = base0_4, threaded = true, winner_cache_mode = :certificate, winner_cache = wc0)
winner_value_update!(PersistentWinnerCache(), ctx4, xf_upper)
println("Warm-up done.")

Random.seed!(777)
const K_SEQ = 10
const STEP_MAGS = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
seq_w = Vector{Vector{Float64}}()
for i in 1:K_SEQ
    mag = STEP_MAGS[rand(1:length(STEP_MAGS))]
    dirn = randn(D4^2); dirn ./= norm(dirn)
    wp = copy(W_UPPER); wp[2:end] .+= mag .* dirn[2:end]
    push!(seq_w, wp)
end
seq_xf = [x_free_from_w(w, pe4) for w in seq_w]

# Force a full GC + a pause before EACH timed run, isolating GC-pause noise as a variable.
GC.gc(true)
t_none = Float64[]
for xf in seq_xf
    push!(t_none, @elapsed composite_gradient_at_fast(xf, ctx4, pe4; threaded = true, winner_cache_mode = :none))
end
println(":none  (post-GC): ", round.(t_none .* 1000, digits = 2), "  total=", round(sum(t_none)*1000, digits=1), "ms")

GC.gc(true)
wc_seq = PersistentWinnerCache()
t_cert = Float64[]
for xf in seq_xf
    push!(t_cert, @elapsed composite_gradient_at_fast(xf, ctx4, pe4; threaded = true, winner_cache_mode = :certificate, winner_cache = wc_seq))
end
println(":certificate (post-GC): ", round.(t_cert .* 1000, digits = 2), "  total=", round(sum(t_cert)*1000, digits=1), "ms")

# repeat WITHOUT forced GC (mirrors the original harness ordering more closely, for comparison)
wc_seq2 = PersistentWinnerCache()
t_cert2 = Float64[]
for xf in seq_xf
    push!(t_cert2, @elapsed composite_gradient_at_fast(xf, ctx4, pe4; threaded = true, winner_cache_mode = :certificate, winner_cache = wc_seq2))
end
println(":certificate (2nd pass, no forced GC): ", round.(t_cert2 .* 1000, digits = 2), "  total=", round(sum(t_cert2)*1000, digits=1), "ms")

println("\nDONE.")
