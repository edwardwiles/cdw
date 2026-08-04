# One-shot diagnostic: is the "Could not evaluate objective at initial point" failure caused by
# W=5000 (too few draws for the zero-covariance pairwise moments to be well-defined) or by the
# :origin_specific_moments_zero_covariance restriction itself, independent of W? Tests
# K_mean=K_pair=1 at W=5000 vs W=80000 (the W the known-working test file uses).
haskey(ENV, "CAMPAIGN_W") || error("set CAMPAIGN_W first")
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
using Statistics
lp(xs...) = (println(xs...); flush(stdout))
OUTDIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/k3_preflight/w_sweep"
isdir(OUTDIR) || mkpath(OUTDIR)

for W in (5_000, 80_000)
    lp("="^60); lp("W=", W, " K_mean=K_pair=1 zero_covariance"); lp("="^60)
    ctx0 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
    D = ctx0.D
    g0 = ctx0.θ0_up[ctx0.free_idx][1]
    pe0 = build_pivot_elimination(ctx0)
    zfree0 = pivot_reduce(reshape(log.(ctx0.θ0_up[ctx0.free_idx][2:end]), ctx0.D, ctx0.D_dest), pe0)
    w_a = vcat(g0, zfree0)
    layout = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for o in 1:D
        nu0[target_index(layout, o, 1)] = mean(ctx0.U[:, o])
    end
    w0 = vcat(w_a, log.(nu0))
    result = try
        run_originzc_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :sobol_randomized,
            draw_seed = 20260719, maxtime_real = 15.0, ckpt_dir = joinpath(OUTDIR, "W$W"),
            label = "wsweep_W$W", checkpoint_interval_s = 9999.0,
            distribution_restriction = :origin_specific_moments_zero_covariance,
            K_mean = 1, K_pair = 1, destination_sample = :exclude_row, verbose = true)
    catch e
        lp("W=", W, " THREW: ", sprint(showerror, e))
        nothing
    end
    if result !== nothing
        lp("W=", W, " ran. best=", result.best === nothing ? "NOTHING" : "kappa=$(result.kappa) Delta=$(result.best.Delta)")
    end
end
