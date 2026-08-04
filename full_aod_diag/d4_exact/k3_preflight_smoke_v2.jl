# k3_preflight_smoke_v2.jl -- corrected K=3 preflight smoke, per the mission's own Section 8
# methodology: do NOT hand-reconstruct the economic [gp;A] block (both a from-scratch attempt in
# k3_preflight_smoke.jl AND the pre-existing d20_originzc_shakedown.jl's own construction crashed
# identically at K=1 too -- "BoundsError: 403-element Vector at index [24:423]" -- BEFORE any
# K-dependent code runs, proving that failure is a K-INDEPENDENT staleness bug in economic-block
# reconstruction code, not evidence about K=3 at all). Instead: take the REAL, already-verified K=1
# origin_zc economic block straight from the immutable frozen registry (Section 2), and extend ONLY
# the K=3-specific eta_nu tail. This is "exactly what the previous run did, extended to K=3."
haskey(ENV, "CAMPAIGN_W") || error("set CAMPAIGN_W first")
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
using Statistics, Serialization
lp(xs...) = (println(xs...); flush(stdout))

# Real, verified K=1 origin_zc economic block (gp;A_nonpivot, 380-length), frozen registry entry
# family=origin_zc direction=upper delta=0.01 GT=0.032882 (IMMUTABLE_SEED_REGISTRY_K1_2026-08-04.csv).
const K1_SEED_PATH = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/frozen_seeds_K1/890424b84acebe64/W100k/origin_zc/upper/delta_0.01/cdb537b72ace59c1.jls"
isfile(K1_SEED_PATH) || error("K=1 frozen seed not found: $K1_SEED_PATH")
state = deserialize(K1_SEED_PATH)
w_k1 = state.report.final_w
lp("K=1 real economic block: length(full_w)=", length(w_k1), " GT=", state.report.final_GT, " Delta*=", state.report.final_Delta_star)
w_a = w_k1[1:380]   # [gp; A_nonpivot], the family-shared economic block -- drop the K=1 eta tail.
lp("economic block [gp;A] length: ", length(w_a), " (expect 380)")

# Fresh context ONLY to get U draws for the K=3 nu0 tail construction -- NOT used for gp0/z0 (those
# come from the real verified w_a above), so this is immune to whatever staleness bug affects
# hand-reconstructing gp0/z0 from ctx.θ0_up.
ctx0 = d20_real_setup(W = 5_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx0.D

layout = OriginByPowerLayout(D, 3, 3)
nu0 = Vector{Float64}(undef, n_eta(layout))
for o in 1:D, k in 1:3
    nu0[target_index(layout, o, k)] = mean(ctx0.U[:, o] .^ k)
end
w0 = vcat(w_a, log.(nu0))
lp("K=3 full outer vector length: ", length(w0), " (expect 440)")

OUTDIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/k3_preflight/v2"
isdir(OUTDIR) || mkpath(OUTDIR)

lp("="^70); lp("K=3 origin_zc smoke from REAL K=1 economic block + fresh K=3 eta tail"); lp("="^70)
result = try
    run_originzc_upper_checkpointed(w0; W = 5_000, delta = 1.0, draw_design = :sobol_randomized,
        draw_seed = 20260719, maxtime_real = 30.0, ckpt_dir = OUTDIR, label = "k3_v2_smoke",
        checkpoint_interval_s = 9999.0, distribution_restriction = :origin_specific_moments_zero_covariance,
        K_mean = 3, K_pair = 3, destination_sample = :exclude_row, verbose = true)
catch e
    lp("THREW: ", sprint(showerror, e))
    nothing
end
if result !== nothing
    lp("RAN. knitro_status=", result.knitro_status, " kappa=", result.kappa,
       " best=", result.best === nothing ? "NOTHING (no feasible point found)" : "Delta=$(result.best.Delta) n_eval=$(result.best.n_eval)")
end
