# 2026-07-25 local-geometry/continuation session, Phase 7 live probe: a SHORT, bounded run
# of the experimental predictor-corrector continuation driver (src/melitz/predictor_corrector.jl)
# at the real D=20/W=80,000/seed=1 calibration, starting from the same near-boundary point
# `scripts/melitz_local_geometry_lab_2026-07-25.jl` characterizes. Deliberately SHORT
# (n_steps small) -- this is a correctness/behavior probe, not a production campaign
# (governing prompt: "do not launch a long production campaign... in this session").
#
# Usage: julia --project=. -t 16 scripts/melitz_predictor_corrector_probe_2026-07-25.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
const OUTER_OPT = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
const OUTER_BUDGET_DELTA = 1.0
const DELTA_EVALUATION_CAP = 10.0
const CAMPAIGN_START_G = -0.497333
const W = 80_000
const N_STEPS = parse(Int, get(ENV, "MELITZ_PC_N_STEPS", "10"))

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
end

function main()
    println("="^100)
    println("Melitz predictor-corrector continuation probe -- 2026-07-25 (SHORT, n_steps=$N_STEPS)")
    println("="^100)
    BLAS.set_num_threads(16)
    calib = load_calibration()
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ
    theta0 = copy(theta_calib); theta0[1] = CAMPAIGN_START_G
    r0 = evaluate_melitz_delta(theta0, ctx, obj_inner; cold=true, store_G=false)
    @printf("start: g=%.6f  Delta=%.6e  kappa=%.6f  nStatus=%d\n",
        CAMPAIGN_START_G, r0.Delta, kappa_of_g(CAMPAIGN_START_G, ctx), r0.nStatus)
    flush(stdout)

    # Radii seeded from the local-geometry lab's own finding (Phase 4B/4C): a pure nuisance
    # step of aggregate norm ~1e-5 stayed finite, ~1e-4 broke -- start comfortably inside
    # that, at 3e-6, and a modest r_g. These are STARTING points for the adaptive scheme,
    # not fixed for the whole run (Phase 6: expand after well-predicted accepted steps,
    # shrink on any rejection).
    radii = MelitzBlockTrustRadii(1e-4, 3e-6)

    t0 = time()
    result = melitz_predictor_corrector_continuation(ctx, obj_inner, theta0;
        delta=OUTER_BUDGET_DELTA, cap=DELTA_EVALUATION_CAP, radii=radii, dg_init=1e-4,
        n_steps=N_STEPS, inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT,
        min_slack_floor=0.0, prediction_error_tol=0.5, expand_after=2)
    wall = time() - t0

    println("\n-- per-step record --")
    for (i, s) in enumerate(result.steps)
        @printf("  [%2d] kind=%-9s accepted=%-5s dg=%10.3e ||deta||=%10.3e Delta=%12s nStatus=%5d kappa=%10s min_slack=%9.5f pred_dD=%10.3e r_g=%9.3e r_eta=%9.3e wall=%5.2fs\n",
            i, string(s.kind), s.accepted, s.dg, s.deta_norm,
            isnan(s.Delta) ? "NaN" : @sprintf("%.6e", s.Delta), s.nStatus,
            isnan(s.kappa) ? "NaN" : @sprintf("%.6f", s.kappa), s.min_slack, s.predicted_dDelta,
            s.r_g, s.r_eta, s.wall)
    end
    flush(stdout)

    @printf("\nFINAL: g=%.6f  Delta=%.6e  kappa=%.6f  n_accepted=%d  n_rejected=%d  wall=%.1fs\n",
        result.final_theta[1], result.final_Delta, result.final_kappa,
        result.n_accepted, result.n_rejected, wall)
    @printf("start kappa=%.6f -> final kappa=%.6f  (lower kappa = higher welfare gain)\n",
        kappa_of_g(CAMPAIGN_START_G, ctx), result.final_kappa)

    println("\n" * "="^100)
    println("Predictor-corrector probe complete.")
    println("="^100)
end

main()
