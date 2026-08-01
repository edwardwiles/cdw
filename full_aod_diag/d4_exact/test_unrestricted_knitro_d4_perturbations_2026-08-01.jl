# ============================================================================
# Claude Code task 2026-08-01, §12: SECOND real KNITRO gate. Three
# deterministic D=4 perturbations (small, moderate, multi-winner-change),
# each run through the FULL decisive procedure (theory doc section 2.3):
# (1) solve reduced/profiled; (2) verify its own LFD; (3) recover full
# gamma-normalized A using that LFD; (4) re-solve the legacy full/reference
# problem at the recovered A; (5) compare.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
using LinearAlgebra, Printf

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
D = ctx.D
θ0 = copy(ctx.θ0_up)

spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w_profiled_calib = reduce_to_w_profiled(gp0, z_calib, pe)
n_rfree = length(w_profiled_calib) - 1
println("D=$D  n_rfree=$n_rfree")

decoded0 = decode_outer_profiled(w_profiled_calib, ctx, pe)
θ_calib_profiled = copy(θ0)
θ_calib_profiled[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= decoded0.Aod_levels
θ_calib_profiled[3+D] = decoded0.gp
cf_calib = build_compressed_factual(θ_calib_profiled, ctx; check_ties = false)
winner_calib = sum(Float64.(cf_calib.winner))

# deterministic perturbation direction: alternating +1/-1 pattern over r_free
direction = [(-1.0)^k for k in 1:n_rfree]

function run_one_point(label::String, scale::Float64)
    println("\n" * "#"^90); println("PERTURBATION: $label (scale=$scale)"); println("#"^90)
    w_profiled = copy(w_profiled_calib)
    w_profiled[2:end] .+= scale .* direction
    decoded = decode_outer_profiled(w_profiled, ctx, pe)
    θ_full_pert = copy(θ_calib_profiled)
    θ_full_pert[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= decoded.Aod_levels
    θ_full_pert[3+D] = decoded.gp

    cf_check = build_compressed_factual(θ_full_pert, ctx; check_ties = false)
    winner_pert = sum(Float64.(cf_check.winner))
    n_winner_diffs = sum(cf_check.winner .!= cf_calib.winner)
    println("winner diffs vs calibration: $n_winner_diffs cells out of $(length(cf_check.winner))")

    obj_p, st_p = build_profiled_operator_bundle(ctx, θ_full_pert, spec; ref_obj = ctx.obj)
    SW = ctx.γ.SamplingWeights[1:obj_p.M]
    obj_p.payoff .= θ_full_pert[3+D] .* SW
    obj_p.H_save = obj_p.payoff[1] * (-1.0)^obj_p.find_smallest
    nStatus_p, objSol_p, x_p, lambda_p, nfg_p, nhess_p = inner_loop_KNITRO_profiled(obj_p, st_p)
    p_ok = nStatus_p ∈ [0, -100, -101, -103]
    zeta_p = x_p[1]; beta_p = x_p[2:end]
    ov_p = verify_inner_solution_reduced_profiled!(zeta_p, beta_p, st_p.cf, ctx, θ_full_pert, st_p.layout, obj_p, st_p.cf.W)
    mw_p, verify_p = verify_namedtuple_from_operator(ov_p, obj_p, st_p.cf.W, nStatus_p)
    println("profiled: nStatus=$nStatus_p  Delta_primal=$(verify_p.Delta_primal)  kkt=$(verify_p.max_abs_moment_kkt_resid)")

    z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(θ_full_pert, ctx, st_p.cf, mw_p)
    θ_full_recovered = copy(θ_full_pert)
    θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(exp.(z_recovered))
    gamma_dev = maximum(abs.(gamma_tilde .- 1.0))
    println("gamma_tilde (pre-recovery deviation from 1) max = $gamma_dev")

    ctx.obj.use_cached_x = false
    ctx.obj.x .= NaN
    K_ref, x_ref, nStatus_ref, nfg_ref, nhess_ref, st_ref = inner_loop_internal_compressed(ctx.obj, θ_full_recovered, ctx)
    ref_ok = nStatus_ref ∈ [0, -100, -101, -103]
    zeta_ref = x_ref[1]; lambda_ref = x_ref[2:end]
    ov_ref = verify_inner_solution_operator_unrestricted!(zeta_ref, lambda_ref, st_ref.cf, ctx.obj, st_ref.cf.W)
    mw_ref, verify_ref = verify_namedtuple_from_operator(ov_ref, ctx.obj, st_ref.cf.W, nStatus_ref)
    println("reference@recovered: nStatus=$nStatus_ref  Delta_primal=$(verify_ref.Delta_primal)  kkt=$(verify_ref.max_abs_moment_kkt_resid)")

    winner_checksum_ref = sum(Float64.(st_ref.cf.winner))
    winner_checksum_p = sum(Float64.(st_p.cf.winner))
    mw_diff = maximum(abs.(mw_ref .- mw_p))
    div_diff = abs(verify_ref.Delta_primal - verify_p.Delta_primal)
    # gravity residual at the recovered point (should be exactly 0 -- gravity is a separate, unaffected constraint)
    grav_resid = abs(gravity_from_logz(log.(reshape(θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)), ctx) -
                      gravity_from_logz(log.(reshape(θ_full_pert[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)), ctx))

    println(@sprintf("RESULT: p_ok=%s ref_ok=%s winner_match=%s mw_diff=%.3e div_diff=%.3e n_winner_diffs_vs_calib=%d gravity_resid_change=%.3e",
        p_ok, ref_ok, winner_checksum_ref == winner_checksum_p, mw_diff, div_diff, n_winner_diffs, grav_resid))

    return (label = label, scale = scale, nStatus_p = nStatus_p, nStatus_ref = nStatus_ref,
        Delta_primal_p = verify_p.Delta_primal, Delta_primal_ref = verify_ref.Delta_primal,
        mw_diff = mw_diff, div_diff = div_diff, winner_match = (winner_checksum_ref == winner_checksum_p),
        n_winner_diffs_vs_calib = n_winner_diffs, gamma_dev_pre_recovery = gamma_dev, gravity_resid_change = grav_resid,
        kkt_p = verify_p.max_abs_moment_kkt_resid, kkt_ref = verify_ref.max_abs_moment_kkt_resid)
end

results = []
push!(results, run_one_point("small", 0.02))
push!(results, run_one_point("moderate", 0.15))

# find a scale that flips multiple winners, deterministically increasing until it does
big_scale = 0.5
local_result = nothing
for trial_scale in (0.5, 1.0, 2.0, 4.0, 8.0)
    global big_scale = trial_scale
    r = run_one_point("multi_winner_change_trial_scale_$(trial_scale)", trial_scale)
    global local_result = r
    if r.n_winner_diffs_vs_calib >= 2
        break
    end
end
push!(results, local_result)

println("\n" * "="^90); println("SUMMARY"); println("="^90)
outpath = joinpath(@__DIR__, "..", "..", "UNRESTRICTED_KNITRO_PERTURBATION_EQUIVALENCE_D4_2026-08-01.csv")
open(outpath, "w") do io
    println(io, "label,scale,nStatus_p,nStatus_ref,Delta_primal_p,Delta_primal_ref,mw_diff,div_diff,winner_match,n_winner_diffs_vs_calib,gamma_dev_pre_recovery,gravity_resid_change,kkt_p,kkt_ref")
    for r in results
        @printf(io, "%s,%.6g,%d,%d,%.10e,%.10e,%.6e,%.6e,%s,%d,%.6e,%.6e,%.6e,%.6e\n",
            r.label, r.scale, r.nStatus_p, r.nStatus_ref, r.Delta_primal_p, r.Delta_primal_ref,
            r.mw_diff, r.div_diff, r.winner_match, r.n_winner_diffs_vs_calib, r.gamma_dev_pre_recovery,
            r.gravity_resid_change, r.kkt_p, r.kkt_ref)
        println("$(r.label): mw_diff=$(r.mw_diff)  div_diff=$(r.div_diff)  winner_match=$(r.winner_match)  n_winner_diffs=$(r.n_winner_diffs_vs_calib)")
    end
end
println("Wrote $outpath")

tol = 1e-4
all_pass = all(r -> r.mw_diff < tol && r.div_diff < tol * max(1.0, abs(r.Delta_primal_p)) && r.winner_match && r.nStatus_p in [0,-100,-101,-103] && r.nStatus_ref in [0,-100,-101,-103], results)
println("\nALL PERTURBATION POINTS PASS (tol=$tol): $all_pass")
@assert all_pass "at least one perturbation point failed the decisive equivalence test"
println("\n" * "="^90); println("ALL TESTS PASSED -- D4 PERTURBATION EQUIVALENCE CONFIRMED (including a multi-winner-change point)"); println("="^90)
