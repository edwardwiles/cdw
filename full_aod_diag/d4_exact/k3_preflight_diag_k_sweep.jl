# k3_preflight_diag_k_sweep.jl -- ad hoc diagnostic (not part of the permanent preflight gate):
# k3_preflight_smoke.jl found BOTH origin_zc and cm_meanzc failing "Could not evaluate objective or
# constraints at the initial point" at K_mean=K_pair=3. This sweeps K=1,2,3 with the IDENTICAL
# nu0-construction method for origin_zc only, to isolate whether the failure is K=3-specific (a
# real gap needing a fix before Wave 1) or present already at K=1/2 (meaning the raw-moment nu0
# construction itself is unsound, a bug in the preflight script, not production code).
haskey(ENV, "CAMPAIGN_W") || error("set CAMPAIGN_W first")
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
using Statistics
lp(xs...) = (println(xs...); flush(stdout))

ctx0 = d20_real_setup(W = 5_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
D = ctx0.D
g0 = ctx0.θ0_up[ctx0.free_idx][1]
pe0 = build_pivot_elimination(ctx0)
zfree0 = pivot_reduce(reshape(log.(ctx0.θ0_up[ctx0.free_idx][2:end]), ctx0.D, ctx0.D_dest), pe0)
w_a = vcat(g0, zfree0)
OUTDIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/k3_preflight/k_sweep"
isdir(OUTDIR) || mkpath(OUTDIR)

for K in (1, 2, 3)
    lp("="^60); lp("K_mean=K_pair=", K); lp("="^60)
    layout = OriginByPowerLayout(D, K, K)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for o in 1:D, k in 1:K
        nu0[target_index(layout, o, k)] = mean(ctx0.U[:, o] .^ k)
    end
    w0 = vcat(w_a, log.(nu0))
    result = try
        run_originzc_upper_checkpointed(w0; W = 5_000, delta = 1.0, draw_design = :sobol_randomized,
            draw_seed = 20260719, maxtime_real = 15.0, ckpt_dir = joinpath(OUTDIR, "K$K"),
            label = "ksweep_K$K", checkpoint_interval_s = 9999.0,
            distribution_restriction = :origin_specific_moments_zero_covariance,
            K_mean = K, K_pair = K, destination_sample = :exclude_row, verbose = true)
    catch e
        lp("K=", K, " THREW: ", sprint(showerror, e))
        nothing
    end
    if result !== nothing
        lp("K=", K, " ran. best=", result.best === nothing ? "NOTHING (infeasible/no point found)" : "kappa=$(result.kappa) Delta=$(result.best.Delta)")
    end
end
