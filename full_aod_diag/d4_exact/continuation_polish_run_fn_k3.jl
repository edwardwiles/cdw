# continuation_polish_run_fn_k3.jl -- K=3 variant of continuation_polish_run_fn.jl, for the
# TARGETED_K3_EXTENSIONS_2026-08-04 origin-ZC / CM+ZC (cm_meanzc) K=3 campaign. Deliberately a
# SEPARATE file, not a parametrization of the K=1 file via an env var: this repo's own rule (never
# let a function default a scientific parameter) plus the task's explicit instruction ("Do not
# overwrite or relabel the completed K=1 campaign") both argue for two small, static, side-by-side
# files over one file with a mutable K knob that could silently drift the completed K=1 campaign's
# behavior. Only the K_mean/K_pair constants and the checkpoint/label namespace differ from the K=1
# file -- everything else (manifest fields, algorithm dispatch) is identical by construction.
#
# Caller must already have included the full driver chain AND continuation_polish_orchestrator.jl,
# exactly as continuation_polish_run_fn.jl requires.

const CONTPOLISH_K3_DRAW_DESIGN = :sobol_randomized
const CONTPOLISH_K3_DRAW_SEED = 20260719
const CONTPOLISH_K3_W = parse(Int, get(ENV, "CAMPAIGN_W") do
    error("continuation_polish_run_fn_k3.jl requires CAMPAIGN_W to be set in the environment -- no default, per this repo's scientific-parameter rule.")
end)
const CONTPOLISH_K3_CM_L = 50
const CONTPOLISH_K3_MEANZC_K = 3
const CONTPOLISH_K3_ORIGINZC_K = 3
const CONTPOLISH_K3_SNAPS = nested_grid_sequence([10, 20, 50])
const CONTPOLISH_K3_PROBS_L50 = CONTPOLISH_K3_SNAPS[CONTPOLISH_K3_CM_L]

"""
    production_run_fn_k3(family, w0, stage, budget_s, ckpt_dir; find_smallest) -> NamedTuple

Only `origin_zc` and `cm_meanzc` are valid families for this K=3 entry point (unrestricted and the
two non-ZC CM families have no K_mean/K_pair axis at all -- calling this for them would silently
duplicate the K=1 run_fn's behavior under a misleading name, which this repo's "no default that
changes what problem is being solved" rule argues against doing implicitly).
"""
function production_run_fn_k3(family::String, w0::Vector{Float64}, stage::AlgoStage, budget_s::Float64,
                               ckpt_dir::AbstractString; find_smallest::Bool)
    family in ("origin_zc", "cm_meanzc") ||
        error("production_run_fn_k3: only origin_zc/cm_meanzc have a K_mean/K_pair axis, got family=$family")
    isdir(ckpt_dir) || mkpath(ckpt_dir)
    label = "$(family)_K3_$(find_smallest ? "upper" : "lower")_$(stage)"
    extra_algo = algo_kwargs(family, stage)

    if family == "origin_zc"
        fn = find_smallest ? run_originzc_upper_checkpointed : run_originzc_lower_checkpointed
        return fn(w0; W = CONTPOLISH_K3_W, delta = ACTIVE_TARGET_DELTA[], draw_design = CONTPOLISH_K3_DRAW_DESIGN,
            draw_seed = CONTPOLISH_K3_DRAW_SEED, distribution_restriction = :origin_specific_moments_zero_covariance,
            K_mean = CONTPOLISH_K3_ORIGINZC_K, K_pair = CONTPOLISH_K3_ORIGINZC_K,
            ckpt_dir = ckpt_dir, run_id = label, label = label,
            checkpoint_interval_s = 900.0, maxtime_real = budget_s, verbose = true, extra_algo...)
    else  # cm_meanzc
        fn = find_smallest ? run_cm_upper_checkpointed : run_cm_lower_checkpointed
        return fn(w0; W = CONTPOLISH_K3_W, delta = ACTIVE_TARGET_DELTA[], draw_design = CONTPOLISH_K3_DRAW_DESIGN,
            draw_seed = CONTPOLISH_K3_DRAW_SEED, L = CONTPOLISH_K3_CM_L, contrasts = :orthonormal,
            probs = CONTPOLISH_K3_PROBS_L50, cm_extension = :cm_plus_moments,
            meanzc_K_mean = CONTPOLISH_K3_MEANZC_K, meanzc_K_pair = CONTPOLISH_K3_MEANZC_K,
            ckpt_dir = ckpt_dir, run_id = label, label = label,
            checkpoint_interval_s = 900.0, maxtime_real = budget_s, verbose = true, extra_algo...)
    end
end
