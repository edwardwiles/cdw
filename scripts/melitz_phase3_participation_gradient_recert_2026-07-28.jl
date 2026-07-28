# Post-consolidation validation, Phase 3: replay the central participation-gradient diagnostic
# at exactly 2 real-D20 points (interior Delta~0.5, near-budget Delta~1), 3 direction families
# (pure gamma, participation/fixed-cost block, one mixed gamma+participation direction), 3 raw
# steps (1e-6,1e-5,1e-4) -- 18 fully-reoptimized inner solves total (governing prompt's own
# budget), via the NEW consolidated `solve_melitz_delta!` API. Adapted from
# `scripts/melitz_anomaly_phase7_8_9_diagnostics_2026-07-28.jl` (drops the technology_block
# direction, not requested by this validation's own narrower Phase 3 spec).

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)

d20 = build_realD20_fixture(; policy=policy)
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D; nA = D^2 - 1

profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]

function g_for_target(target)
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    return gs[k] + t * (gs[k+1] - gs[k])
end

g_interior = g_for_target(0.5)
g_nearbudget = g_for_target(1.0)
println("g_interior (DeltaStar~0.5) = ", g_interior)
println("g_nearbudget (DeltaStar~1) = ", g_nearbudget)
flush(stdout)

theta_interior = copy(theta0); theta_interior[1] = g_interior
theta_nearbudget = copy(theta0); theta_nearbudget[1] = g_nearbudget

h_fd = 1e-4
direct_gradient_fn = make_melitz_gradient_delta_direct_parallel(h_fd)

gt_pct(g) = 100.0 * melitz_welfare_metrics_from_g(g, ctx).gains_from_trade

rows7 = NamedTuple[]
n_solves = Ref(0)

function run_point(label, theta_pt)
    println("\n" * "="^100)
    println("POINT: $label  (theta[1]=g=$(theta_pt[1]))")
    println("="^100)
    session0 = MelitzInnerSession(obj, ctx, policy)
    r0 = solve_melitz_delta!(session0, theta_pt, policy; warm_start_source=:previous)
    n_solves[] += 1
    @assert r0 isa FiniteSolved "base point $label not FiniteSolved: $(typeof(r0))"
    Delta0 = r0.Delta
    x0 = copy(r0.x)
    GT0 = gt_pct(theta_pt[1])
    println("  Delta0=$Delta0  GT0=$(GT0)pp  ||x0||=$(norm(x0))")
    flush(stdout)

    snap = base_active_mask(theta_pt, ctx, obj)

    g_grad = zeros(n)
    direct_gradient_fn(g_grad, theta_pt, ctx, obj, x0)
    g_grad ./= 1e10   # convert d(1e10*Delta)/dtheta -> dDelta/dtheta

    # direction 1: pure gamma (welfare-improving). Sign via a tiny probe on GT itself.
    sign1 = gt_pct(theta_pt[1] + 1e-6) > GT0 ? 1.0 : -1.0
    v1 = zeros(n); v1[1] = sign1

    # direction 2: normalized participation/fixed-cost (logf) block, random unit vector
    # (seeded, deterministic), oriented DeltaStar-decreasing per the assembled gradient.
    rawF = zeros(n); rawF[2+nA:end] .= randn(MersenneTwister(3001 + Int(round(1000*theta_pt[1]))), n - 1 - nA)
    rawF ./= norm(rawF)
    v2 = (dot(g_grad, rawF) > 0 ? -1.0 : 1.0) .* rawF

    # direction 3: mixed gamma+participation -- the production gradient's own restriction to
    # those coordinates (technology/A-block zeroed), negated (descent on predicted DeltaStar).
    v3raw = zeros(n); v3raw[1] = -g_grad[1]; v3raw[2+nA:end] .= -g_grad[2+nA:end]
    v3 = v3raw ./ norm(v3raw)

    directions = [("pure_gamma", v1, false), ("participation_block", v2, true),
                  ("mixed_gamma_participation", v3, true)]
    steps = (1e-6, 1e-5, 1e-4)

    for (dname, v, has_secant) in directions
        for step in steps
            theta_new = theta_pt .+ step .* v
            predicted_A = dot(g_grad, step .* v)   # assembled coordinatewise gradient prediction

            predicted_B = NaN
            if has_secant
                melitz_bundle_prepare_at_theta!(obj, theta_new)
                localc = zeros(1)
                obj(x0, constr=localc)   # FIXED dual x0, moved moment matrix -- one-shot block secant
                Delta_fixed_new = localc[1] / 1e10
                predicted_B = Delta_fixed_new - Delta0
            end

            session_i = MelitzInnerSession(obj, ctx, policy)   # fresh bank per trial (matches original convention)
            r_new = solve_melitz_delta!(session_i, theta_new, policy; warm_start_source=:previous)
            n_solves[] += 1
            reoptimized = r_new isa FiniteSolved ? (r_new.Delta - Delta0) : NaN
            nsw, _, nauk = count_switches(snap, theta_new, ctx, obj)
            dGT = gt_pct(theta_new[1]) - GT0

            push!(rows7, (point=label, direction=dname, step=step, dGT_pp=dGT,
                pred_dDelta_assembled_gradient=predicted_A, pred_dDelta_block_secant=predicted_B,
                reoptimized_dDelta=reoptimized, n_switches=nsw, n_autarky_switches=nauk,
                classification=string(typeof(r_new))))
            @printf("  [%-26s step=%.0e] dGT=%+.4fpp  A=%+.4e  B=%s  reopt=%+.4e  nsw=%d  class=%s\n",
                dname, step, dGT, predicted_A, has_secant ? @sprintf("%+.4e", predicted_B) : "NA",
                reoptimized, nsw, typeof(r_new))
            flush(stdout)
        end
    end
    return (theta=theta_pt, Delta0=Delta0)
end

run_point("interior_Delta0.5", theta_interior)
run_point("nearbudget_Delta1", theta_nearbudget)

open(joinpath(OUTDIR, "melitz_post_consolidation_phase3_participation_gradient_replay_2026-07-28.csv"), "w") do io
    cols = keys(rows7[1])
    println(io, join(cols, ","))
    for r in rows7
        println(io, join([r[c] for c in cols], ","))
    end
end
println("\nTotal fully-reoptimized inner solves this script (base points not counted in the " *
        "18-solve directional budget): ", n_solves[], "  (18 directional + 2 base-point)")
println("\nWrote docs/key_results/melitz_post_consolidation_phase3_participation_gradient_replay_2026-07-28.csv")
println("\nDONE Phase 3 (post-consolidation).")
