# ============================================================================
# Generates rand1/rand2 -- the two log-normal random perturbations of A_od* used as cold
# starts by ALL FOUR methods (LC/LU/GC/GU) in the head-to-head comparison -- ONCE, and saves
# them to a single shared JLD2 file. All 4 method drivers LOAD this file rather than drawing
# their own random points, so there is no possibility of an RNG-consumption-order mismatch
# silently making "the same seed" produce different vectors across methods.
#
# Seed and per-vector sigma match the convention already used elsewhere in this repo
# (multistart_screening_d20.jl: Random.seed!(20260715); noise = sigma .* randn(D);
# Acol_pert = Acol_star .* exp.(noise)) -- rand1 at sigma=0.5 (a moderate perturbation),
# rand2 at sigma=1.0 (a larger one), matching multistart_screening_d20.jl's own tested
# NOISE_SCALES and the scale already spot-tested successfully elsewhere in this repo
# (scaled_constrained_D20_randomstart_seed1_sigma0.5.jld2 / ..._seed2_sigma1.0.jld2).
#
# Cheap and non-KNITRO: just RNG + a save, no solves. Run once before launching any of the
# 4 overnight jobs.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     julia -t 19 --project=. sequential_gravity/head_to_head/generate_shared_starts.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Random, JLD2, Printf, LinearAlgebra

const SEED = 20260715
const SIGMA1 = 0.5
const SIGMA2 = 1.0
const OUT_PATH = joinpath(@__DIR__, "shared_starts.jld2")

Acol_star = θr0[4:3+D]

Random.seed!(SEED)
noise1 = SIGMA1 .* randn(D)
noise2 = SIGMA2 .* randn(D)
rand1 = Acol_star .* exp.(noise1)
rand2 = Acol_star .* exp.(noise2)

relΔA1 = norm(rand1 .- Acol_star) / norm(Acol_star)
relΔA2 = norm(rand2 .- Acol_star) / norm(Acol_star)

@printf("Acol_star = %s\n", Acol_star)
@printf("rand1 (sigma=%.2f): relΔA = %.4f\n", SIGMA1, relΔA1)
@printf("rand2 (sigma=%.2f): relΔA = %.4f\n", SIGMA2, relΔA2)

# Cheap feasibility sanity check (informational only, matches multistart_screening_d20.jl's
# own check) -- confirms both shared starts are at least GRAVITY-feasible before 4 separate
# overnight jobs all try to use them; does NOT check divergence-budget feasibility at any
# particular target (that's each method's own job to discover).
θtest1 = copy(θr0); θtest1[4:3+D] .= rand1
θtest2 = copy(θr0); θtest2[4:3+D] .= rand2
_, R1, _, _, _, ok1 = seq_gravcol(θtest1; δ = Inf, maxit = 100, tol = 5e-4)
_, R2, _, _, _, ok2 = seq_gravcol(θtest2; δ = Inf, maxit = 100, tol = 5e-4)
@printf("rand1 gravity-feasible (delta=Inf check): %s (R_mean=%.3e)\n", ok1, R1)
@printf("rand2 gravity-feasible (delta=Inf check): %s (R_mean=%.3e)\n", ok2, R2)
if !ok1 || !ok2
    @warn "at least one shared random start is NOT gravity-feasible even ignoring the divergence budget -- every method's cold start from it will report 'no feasible point found', which is a legitimate but uninformative result. Consider a smaller sigma if this is unexpected."
end

JLD2.save(OUT_PATH, Dict(
    "seed" => SEED, "sigma1" => SIGMA1, "sigma2" => SIGMA2,
    "Acol_star" => Acol_star, "rand1" => rand1, "rand2" => rand2,
    "relDeltaA1" => relΔA1, "relDeltaA2" => relΔA2,
    "rand1_gravity_ok" => ok1, "rand2_gravity_ok" => ok2,
    "D" => D, "W" => W, "done" => true, "timestamp" => string(Dates.now())))
println("\nSaved shared starts to: $OUT_PATH")
println("GENERATE_SHARED_STARTS DONE")
