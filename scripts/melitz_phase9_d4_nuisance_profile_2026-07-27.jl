# Governing prompt (outer-search session), Phase 9: scaled nuisance-profile formulation.
#
# DeltaProfile(g) = min_eta DeltaStar(g, eta), eta = complete A/f nuisance vector, using the
# EXISTING production-fast nuisance-profile infrastructure (`solve_melitz_nuisance_min_delta`,
# nuisance_profile.jl) -- no custom optimizer. Predetermined ORDERED grid (not root-finding):
# starts at the interior fixed-A/f point (g=0, this D=4 fixture's own calibration point) and
# walks outward toward more ambitious (more negative -- the :upper/find_smallest direction's
# own welfare-improving sign, confirmed by Phase 8's own best_g<0 finding) g values, each
# warm-started from the PRECEDING point's own verified nuisance coordinates + inner dual --
# never from an AboveEvaluationCap/NumericalFailure state.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)

data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free)
ctx = obj.γ
n = length(theta0)

r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
@assert r0.verified
println("theta0 Delta0=", r0.Delta, "  g0=", theta0[1])

inner_opt = joinpath(REPO, "melitz_inner_loop_options.opt")
# Governing prompt Phase 2 / user-flagged recurrence (2026-07-XX): :full_value is DELIBERATELY
# uncapped by design (its own docstring/Base.show say so) -- wrong choice for a nuisance-profile
# search, which runs MANY repeated/nested inner solves at trial (g, eta) points, some of which
# can be poorly conditioned. solve_melitz_nuisance_min_delta unconditionally overwrites
# obj_inner.lower_limit = inner_solve_config.lower_limit at entry (nuisance_profile.jl:393)
# regardless of how obj_inner was originally built, so :evaluation_cap here is sufficient on
# its own -- confirmed live: an earlier run of this script using :full_value by mistake
# diverged into a numerically-garbage KNITRO trajectory (objective values ~1e16-1e19) and ran
# for 35+ minutes before being killed, exactly the failure mode this mode exists to prevent.
inner_cfg = MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=10.0)
melitz_assert_evaluation_cap_active(inner_cfg)
free_mask = trues(n); free_mask[1] = false   # gamma FIXED at each grid g; every A/f nuisance coordinate free

# Predetermined ordered grid: 0 (interior) -> increasingly negative g (the welfare-improving
# direction, per Phase 8's own best_g<0 finding), 14 points, continuation from less to more
# ambitious.
g_grid = [0.0, -0.005, -0.01, -0.015, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10, -0.12, -0.15]

rows = NamedTuple[]
theta_prev = copy(theta0)
x_prev = copy(r0.dual_x)
last_good_theta = copy(theta0)
last_good_x = copy(r0.dual_x)

for (i, g) in enumerate(g_grid)
    theta_start = copy(last_good_theta)
    theta_start[1] = g
    println("\n=== grid point $i: g=$g ===")
    local res
    local wall = @elapsed begin
        res = solve_melitz_nuisance_min_delta(ctx, obj, theta_start; free_mask=free_mask,
            radius=0.3, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
            inner_loop_opt=inner_opt, warm_start_x=last_good_x,
            inner_solve_config=inner_cfg, forbid_dense_fallback=true)
    end
    ok = res.nStatus in (0, -100, -101, -103) && isfinite(res.Delta_min) && res.Delta_min > 0
    wm = melitz_welfare_metrics_from_g(res.theta_final[1], ctx)
    push!(rows, (g_target=g, g_final=res.theta_final[1], nStatus=res.nStatus, ok=ok,
        Delta_min=res.Delta_min, kappa_ratio=wm.kappa_ratio, GT=wm.gains_from_trade,
        wall=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls))
    println(@sprintf("  nStatus=%d ok=%s Delta_min=%.6e kappa_ratio=%.6f GT=%.6f wall=%.2fs",
        res.nStatus, ok, res.Delta_min, wm.kappa_ratio, wm.gains_from_trade, wall))
    if ok
        global last_good_theta = copy(res.theta_final)
        global last_good_x = copy(res.r_final.dual_x)
    else
        println("  (grid point NOT verified -- NOT used as warm start for the next point)")
    end
end

outfile = joinpath(OUTDIR, "melitz_phase9_d4_nuisance_profile_2026-07-27.csv")
open(outfile, "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join([r[c] for c in cols], ","))
    end
end
println("\nWrote ", outfile)

# Compare against fixed A/f (no nuisance flexibility) at the SAME g grid, for the "fixed vs
# flexible" comparison the governing prompt's own Phase 9 asks for where informative.
println("\n== fixed A/f comparison (same g grid, eta held at theta0's own values) ==")
rows_fixed = NamedTuple[]
for g in g_grid
    theta_fixed = copy(theta0); theta_fixed[1] = g
    r = evaluate_melitz_delta(theta_fixed, ctx, obj; cold=true, store_G=false)
    wm = melitz_welfare_metrics_from_g(g, ctx)
    push!(rows_fixed, (g=g, nStatus=r.nStatus, verified=r.verified,
        Delta=r.nStatus == 0 ? r.Delta : NaN, kappa_ratio=wm.kappa_ratio, GT=wm.gains_from_trade))
    println(@sprintf("g=%.3f nStatus=%d verified=%s Delta=%s kappa_ratio=%.6f GT=%.6f",
        g, r.nStatus, r.verified, r.nStatus == 0 ? @sprintf("%.6e", r.Delta) : "NA", wm.kappa_ratio, wm.gains_from_trade))
end
outfile2 = joinpath(OUTDIR, "melitz_phase9_d4_fixed_af_comparison_2026-07-27.csv")
open(outfile2, "w") do io
    cols = keys(rows_fixed[1])
    println(io, join(cols, ","))
    for r in rows_fixed
        println(io, join([r[c] for c in cols], ","))
    end
end
println("Wrote ", outfile2)
