# k3_preflight_smoke_v3_ksweep.jl -- with the REAL verified K=1 economic block held fixed (isolates
# the economic-block-construction confound entirely), sweep K_mean=K_pair in {1,2,3} to find exactly
# where the failure starts, and print the actual nu0/eta0 values (not just their length) to check
# for out-of-bounds/degenerate numbers.
haskey(ENV, "CAMPAIGN_W") || error("set CAMPAIGN_W first")
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
using Statistics, Serialization
lp(xs...) = (println(xs...); flush(stdout))

const K1_SEED_PATH = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/frozen_seeds_K1/890424b84acebe64/W100k/origin_zc/upper/delta_0.01/cdb537b72ace59c1.jls"
state = deserialize(K1_SEED_PATH)
w_a = state.report.final_w[1:380]

ctx0 = d20_real_setup(W = 5_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx0.D
OUTDIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/k3_preflight/v3"
isdir(OUTDIR) || mkpath(OUTDIR)

for K in (1, 2, 3)
    lp("="^60); lp("K_mean=K_pair=", K, " (real K=1 econ block)"); lp("="^60)
    layout = OriginByPowerLayout(D, K, K)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for o in 1:D, k in 1:K
        nu0[target_index(layout, o, k)] = mean(ctx0.U[:, o] .^ k)
    end
    eta0 = log.(nu0)
    lp("nu0 range: [", minimum(nu0), ", ", maximum(nu0), "]  all_finite=", all(isfinite, nu0), " all_positive=", all(>(0), nu0))
    lp("eta0(=log nu0) range: [", minimum(eta0), ", ", maximum(eta0), "]  all_finite=", all(isfinite, eta0))
    w0 = vcat(w_a, eta0)
    result = try
        run_originzc_upper_checkpointed(w0; W = 5_000, delta = 1.0, draw_design = :sobol_randomized,
            draw_seed = 20260719, maxtime_real = 15.0, ckpt_dir = joinpath(OUTDIR, "K$K"),
            label = "v3_K$K", checkpoint_interval_s = 9999.0,
            distribution_restriction = :origin_specific_moments_zero_covariance,
            K_mean = K, K_pair = K, destination_sample = :exclude_row, verbose = true)
    catch e
        lp("K=", K, " THREW: ", sprint(showerror, e))
        nothing
    end
    if result !== nothing
        lp("K=", K, " knitro_status=", result.knitro_status, " kappa=", result.kappa,
           " best=", result.best === nothing ? "NOTHING" : "Delta=$(result.best.Delta)")
    end
end
