# reverify_stuck_boundary_points.jl -- loads a REAL checkpoint from the completed K=1 campaign
# (not a constructed/synthetic point) and re-evaluates its LAST outer iterate (checkpoint.zfree/g,
# NOT best_feasible -- CHECKPOINT_CONTENTS_VERIFICATION_2026-08-03.md's own distinction) under the
# CURRENT (fixed) verification gate, to test directly whether a point the real campaign logged as
# `feasible=true verified=false` near a delta boundary was correctly rejected or was a genuine
# false-negative from the m_min=0 underflow bug (TARGETED_K3_EXTENSIONS_2026-08-04, Section 7).
haskey(ENV, "CAMPAIGN_W") || error("set CAMPAIGN_W first")
const D4E = @__DIR__
include(joinpath(D4E, "full_chain_include.jl"))
using Statistics, Serialization
lp(xs...) = (println(xs...); flush(stdout))

ckpt_path = ARGS[1]   # e.g. .../origin_zc/lower/delta_1.0/origin_zc_lower_EXPLORE_DIRECT_SR1_latest.jls
family = ARGS[2]
delta = parse(Float64, ARGS[3])
find_smallest = ARGS[4] == "true"

ckpt = family == "origin_zc" ? load_cm_checkpoint_v10(ckpt_path) : load_cm_checkpoint(ckpt_path)
lp("Checkpoint loaded: g=", ckpt.g, " (last outer iterate's gp, NOT best_feasible)")
lp("checkpoint.best_feasible: ", ckpt.best_feasible === nothing ? "nothing" : "gp=$(ckpt.best_feasible.gp) Delta=$(ckpt.best_feasible.Delta)")

# Reconstruct the LAST-ITERATE w (zfree is canonical z-space; eta_nu is the family's own extra tail)
# Need theta_cm/xy_cm/pe to convert zfree->A_native (powered_aspace) -- built exactly as
# cm_checkpoint.jl's own resume logic does it (cm_fixed_theta/precompute_cm_aspace_xy), from a
# context built with the SAME W/delta/draw_seed/draw_design the checkpoint's own run used.
ctx = d20_real_setup_design(W = parse(Int, ENV["CAMPAIGN_W"]), δ = delta, find_smallest = find_smallest,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row)
pe = build_pivot_elimination(ctx)
theta_cm = cm_fixed_theta(ctx)
xy_cm = precompute_cm_aspace_xy(ctx)
A_native = cm_a_from_z(ckpt.zfree, theta_cm, xy_cm, pe)
w_last = vcat(ckpt.g, A_native, ckpt.eta_nu)
lp("Reconstructed last-iterate w: length=", length(w_last), " w[1](gp)=", w_last[1])

OUTDIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/targeted_k3_extensions/reverify_stuck"
isdir(OUTDIR) || mkpath(OUTDIR)

fn = family == "origin_zc" ? (find_smallest ? run_originzc_upper_checkpointed : run_originzc_lower_checkpointed) :
     (find_smallest ? run_cm_upper_checkpointed : run_cm_lower_checkpointed)

result = if family == "origin_zc"
    fn(w_last; W = parse(Int, ENV["CAMPAIGN_W"]), delta = delta, draw_design = :sobol_randomized,
        draw_seed = 20260719, maxtime_real = 45.0, ckpt_dir = OUTDIR, label = "reverify_$(family)",
        checkpoint_interval_s = 9999.0, distribution_restriction = :origin_specific_moments_zero_covariance,
        K_mean = 1, K_pair = 1, destination_sample = :exclude_row, verbose = true)
else
    fn(w_last; W = parse(Int, ENV["CAMPAIGN_W"]), delta = delta, draw_design = :sobol_randomized,
        draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[50],
        maxtime_real = 45.0, ckpt_dir = OUTDIR, label = "reverify_$(family)", checkpoint_interval_s = 9999.0,
        destination_sample = :exclude_row, verbose = true)
end

lp("RESULT: knitro_status=", result.knitro_status, " kappa=", result.kappa,
   " best=", result.best === nothing ? "NOTHING (still rejected under fixed gate)" : "Delta=$(result.best.Delta) gp=$(result.best.gp)")
