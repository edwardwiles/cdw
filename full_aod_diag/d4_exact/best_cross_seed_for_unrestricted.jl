# best_cross_seed_for_unrestricted.jl -- finds the BEST already-completed restricted-family result
# (flexible_cm, common_frechet, origin_zc, cm_meanzc) at the given direction and at ANY delta <=
# the target delta, to offer as an extra_seed for unrestricted. unrestricted is a strict relaxation
# of every restricted family (identical [gp;A_nonpivot] 380-dim block, just fewer/no extra moment
# restrictions -- confirmed by continuation_campaign_cell_driver.jl's own pre-existing
# CROSS_SEED_FAMILY mechanism, which slices ANY family's final_w[1:380] the same way), so any
# restricted family's feasible point at delta<=target is AUTOMATICALLY feasible for unrestricted
# too. A smaller-delta result remains feasible at a larger target delta as well (same continuation
# logic the campaign already uses within one family).
#
# Usage: julia --project=. best_cross_seed_for_unrestricted.jl <direction> <target_delta>
# Prints either "NONE" or "<path>\t<role>\t<GT>" to stdout (last line only, for easy shell capture).
const D4E = @__DIR__
isdefined(Main, :OrchestratorRunState) || include(joinpath(D4E, "continuation_polish_orchestrator.jl"))
using Serialization

const CAMPAIGN_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/campaign_output"
const CANDIDATE_FAMILIES = ("flexible_cm", "common_frechet", "origin_zc", "cm_meanzc")
const DELTA_GRID = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)

direction = ARGS[1]
target_delta = parse(Float64, ARGS[2])
better(gt, cur) = direction == "upper" ? gt > cur : gt < cur

best_gt = nothing
best_path = nothing
best_role = nothing
for family in CANDIDATE_FAMILIES, delta in DELTA_GRID
    delta > target_delta + 1e-9 && continue
    p = joinpath(CAMPAIGN_ROOT, family, direction, "delta_$(delta)", "report.jls")
    isfile(p) || continue
    try
        r = load_run_state(p).report
        length(r.final_w) >= 380 || continue
        if best_gt === nothing || better(r.final_GT, best_gt)
            global best_gt = r.final_GT
            global best_path = p
            global best_role = "cross_$(family)_delta_$(delta)"
        end
    catch
        continue
    end
end

if best_path === nothing
    println("NONE")
else
    println(best_path, "\t", best_role, "\t", best_gt)
end
