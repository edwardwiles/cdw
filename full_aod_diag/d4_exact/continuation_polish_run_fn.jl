# continuation_polish_run_fn.jl -- the real `run_fn` for continuation_polish_orchestrator.jl's
# `run_target_cell!`, dispatching to the actual production driver entry points with the same
# scientific-manifest kwargs the FULL W=100k 10x10 campaign used (see
# docs/audits/fullA-continuation-polish-2026-08-03/SCIENTIFIC_MANIFEST_VALIDATION_2026-08-03.md),
# plus `algo_kwargs(family, stage)` spliced in to select the exploration/polish algorithm.
#
# Caller must already have included the full driver chain (both campaign_unrestricted_runner.jl's
# and campaign_cm_family_runner.jl's include lists) AND continuation_polish_orchestrator.jl before
# including this file.

const CONTPOLISH_DRAW_DESIGN = :sobol_randomized
const CONTPOLISH_DRAW_SEED = 20260719
# W is a scientific parameter (per this repo's own rule: no function may silently default a
# parameter that changes what economic problem is being solved) -- required via CAMPAIGN_W env var,
# no fallback. All of this campaign's original W=100k runs set it explicitly; a later W=250k
# extension (2026-08-04, user-requested, testing whether W=100k's Monte Carlo draw count is itself
# limiting how far delta=2 upper bounds can be pushed) sets CAMPAIGN_W=250000.
const CONTPOLISH_W = parse(Int, get(ENV, "CAMPAIGN_W") do
    error("continuation_polish_run_fn.jl requires CAMPAIGN_W to be set in the environment (e.g. `export CAMPAIGN_W=100000`) -- no default, per this repo's scientific-parameter rule.")
end)
const CONTPOLISH_CM_L = 50
const CONTPOLISH_MEANZC_K = 1
const CONTPOLISH_ORIGINZC_K = 1
const CONTPOLISH_SNAPS = nested_grid_sequence([10, 20, 50])
const CONTPOLISH_PROBS_L50 = CONTPOLISH_SNAPS[CONTPOLISH_CM_L]
const CONTPOLISH_UNRESTRICTED_LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)

# run_target_cell! doesn't thread `delta` through to run_fn (the driver functions take `delta` as
# their own kwarg, matching the campaign runners' own convention of one delta per call), so it is
# passed via this small piece of call-scoped state instead of widening run_fn's signature just for
# this one value shared across all calls within a single run_target_cell! invocation.
const ACTIVE_TARGET_DELTA = Ref(NaN)

"""
    production_run_fn(family, w0, stage, budget_s, ckpt_dir; find_smallest) -> NamedTuple

The real (non-mock) run_fn. `ckpt_dir` must be a fresh directory per (family,direction,delta,stage)
invocation -- the driver's own `run_id`/`label` and checkpoint naming assume that.
"""
function production_run_fn(family::String, w0::Vector{Float64}, stage::AlgoStage, budget_s::Float64,
                            ckpt_dir::AbstractString; find_smallest::Bool)
    isdir(ckpt_dir) || mkpath(ckpt_dir)
    label = "$(family)_$(find_smallest ? "upper" : "lower")_$(stage)"
    extra_algo = algo_kwargs(family, stage)

    if family == "unrestricted"
        return run_polish_checkpointed_unified(label, find_smallest, w0;
            layout = CONTPOLISH_UNRESTRICTED_LAYOUT, maxtime_real = budget_s,
            W_in = CONTPOLISH_W, delta_in = ACTIVE_TARGET_DELTA[], draw_seed_in = CONTPOLISH_DRAW_SEED,
            draw_design_in = CONTPOLISH_DRAW_DESIGN, ckpt_dir = ckpt_dir, checkpoint_interval_s = 900.0,
            resume_from = nothing, destination_sample = :exclude_row, extra_algo...)
    elseif family == "origin_zc"
        fn = find_smallest ? run_originzc_upper_checkpointed : run_originzc_lower_checkpointed
        return fn(w0; W = CONTPOLISH_W, delta = ACTIVE_TARGET_DELTA[], draw_design = CONTPOLISH_DRAW_DESIGN,
            draw_seed = CONTPOLISH_DRAW_SEED, distribution_restriction = :origin_specific_moments_zero_covariance,
            K_mean = CONTPOLISH_ORIGINZC_K, K_pair = CONTPOLISH_ORIGINZC_K,
            ckpt_dir = ckpt_dir, run_id = label, label = label,
            checkpoint_interval_s = 900.0, maxtime_real = budget_s, verbose = true, extra_algo...)
    else
        fn = find_smallest ? run_cm_upper_checkpointed : run_cm_lower_checkpointed
        family_extra = family == "common_frechet" ? (marginal_restriction = :common_frechet, cm_hessian_backend = :structured, cm_gradient_backend = :cplus) :
                       family == "cm_meanzc"      ? (cm_extension = :cm_plus_moments, meanzc_K_mean = CONTPOLISH_MEANZC_K, meanzc_K_pair = CONTPOLISH_MEANZC_K) :
                       NamedTuple()
        return fn(w0; W = CONTPOLISH_W, delta = ACTIVE_TARGET_DELTA[], draw_design = CONTPOLISH_DRAW_DESIGN,
            draw_seed = CONTPOLISH_DRAW_SEED, L = CONTPOLISH_CM_L, contrasts = :orthonormal, probs = CONTPOLISH_PROBS_L50,
            ckpt_dir = ckpt_dir, run_id = label, label = label,
            checkpoint_interval_s = 900.0, maxtime_real = budget_s, verbose = true, family_extra..., extra_algo...)
    end
end
