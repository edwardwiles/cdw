# ============================================================================
# Generates 50 log-normal A_od perturbations of A* at graduated noise scales
# (matching multistart_screening_d20.jl's own convention: sigma in
# [0.1, 0.3, 0.6, 1.0, 2.0], 10 reps per scale) for the LU multistart follow-up
# (see conversation 2026-07-15: LU is cheap enough that a much larger multistart
# is affordable; T1/T3 solves cost ~20-30s, T2 solves cost ~430s -- 50 points x
# 3 targets = 150 solves, ~7.2h estimated, dominated by T2's cost).
#
# The SAME 50 points are reused across all 3 targets (T1/T2/T3), matching this
# whole task's shared-start philosophy -- generated ONCE here and saved, so the
# multistart driver just loads them (no per-run RNG risk).
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/generate_lu_multistart_points.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Random, JLD2, Printf, LinearAlgebra

const SEED_MULTI = 20260716   # distinct from the original shared_starts.jld2 seed (20260715)
const NOISE_SCALES = [0.1, 0.3, 0.6, 1.0, 2.0]
const N_PER_SCALE = 10        # 5 scales x 10 reps = 50 points
const OUT_PATH = joinpath(@__DIR__, "lu_multistart_points.jld2")

Acol_star = θr0[4:3+D]

Random.seed!(SEED_MULTI)
points = Vector{Vector{Float64}}()
sigmas = Float64[]
for σ in NOISE_SCALES, _ in 1:N_PER_SCALE
    noise = σ .* randn(D)
    push!(points, Acol_star .* exp.(noise))
    push!(sigmas, σ)
end
@assert length(points) == 50

relΔAs = [norm(p .- Acol_star) / norm(Acol_star) for p in points]
@printf("Generated %d points, sigma schedule=%s (10 reps each)\n", length(points), NOISE_SCALES)
@printf("relΔA range: [%.3f, %.3f]\n", minimum(relΔAs), maximum(relΔAs))

JLD2.save(OUT_PATH, Dict(
    "seed" => SEED_MULTI, "noise_scales" => NOISE_SCALES, "n_per_scale" => N_PER_SCALE,
    "Acol_star" => Acol_star, "points" => points, "sigmas" => sigmas, "relDeltaAs" => relΔAs,
    "D" => D, "W" => W, "done" => true, "timestamp" => string(Dates.now())))
println("Saved to: $OUT_PATH")
println("GENERATE_LU_MULTISTART_POINTS DONE")
