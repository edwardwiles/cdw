# ============================================================================
# Claude Code task 2026-08-01, §15: small real-D20 omit-ROW gate.
# destination_sample=:exclude_row (D=20, Ddest=19, real ROW-excluded
# rectangular production sample), unrestricted family, fixed theta, small
# diagnostic W (15,000). Calibration comparison + one modest perturbation,
# both via the SAME decisive recover-then-resolve procedure validated at D4.
# Diagnostic only -- does not establish readiness on any other branch.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
flush(stdout)

const W_SMALL = 80_000   # user correction 2026-08-01: W=15,000 is too low to be numerically reliable
                          # at real D=20 (matches this repo's own prior memory: "D=20 real-data
                          # W-sensitivity" -- W=8000 understates kappa vs W>=80000; "Melitz small-W
                          # numerically finicky" -- use W>=80,000 for anything reported). The
                          # original W=15000 run's KNITRO non-convergence (-400, kkt~0.12-0.36) may
                          # simply have been an artifact of an unreliably small W, not a scaling or
                          # correctness problem -- retest at the size this project actually trusts.
println("="^90); println("Building real D=20 :exclude_row context at W=$W_SMALL ..."); flush(stdout)
ctx0 = d20_real_setup(W = W_SMALL, destination_sample = :exclude_row)
ctx = build_unrestricted_operator_ctx(ctx0)
D = ctx.D; Ddest = ctx.D_dest
println("D=$D  Ddest=$Ddest  bi=$(ctx.bi)  row_idx=$(ctx.row_idx)  destination_sample=$(ctx.destination_sample)")
flush(stdout)
@assert D == 20 && Ddest == 19 "expected live D=20, Ddest=19 -- got D=$D, Ddest=$Ddest"

θ0 = copy(ctx.θ0_up)

println("\n" * "="^90); println("CONTROL TEST: reference (old/production) formulation, cold-started at calibration, independent of the profiled/recovery pipeline -- isolates whether any convergence difficulty is specific to the reduced kernels or a general cold-start-at-D20-scale property"); println("="^90); flush(stdout)
ctx.obj.use_cached_x = false
ctx.obj.x .= NaN
t_solve_ctrl = @elapsed (K_ctrl, x_ctrl, nStatus_ctrl, nfg_ctrl, nhess_ctrl, st_ctrl) = inner_loop_internal_compressed(ctx.obj, θ0, ctx)
zeta_ctrl = x_ctrl[1]; lambda_ctrl = x_ctrl[2:end]
ov_ctrl = verify_inner_solution_operator_unrestricted!(zeta_ctrl, lambda_ctrl, st_ctrl.cf, ctx.obj, st_ctrl.cf.W)
mw_ctrl, verify_ctrl = verify_namedtuple_from_operator(ov_ctrl, ctx.obj, st_ctrl.cf.W, nStatus_ctrl)
println("CONTROL (reference, cold start, calibration): nStatus=$nStatus_ctrl  time=$(t_solve_ctrl)s  n_fg=$nfg_ctrl  n_hess=$nhess_ctrl  Delta_primal=$(verify_ctrl.Delta_primal)  kkt=$(verify_ctrl.max_abs_moment_kkt_resid)")
flush(stdout)
korea_idx = 14; brazil_idx = 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
println("anchor_origin_by_slot (first 5) = ", spec.anchor_origin[1:5], " ... slot14(korea)=", spec.anchor_origin[14])
flush(stdout)

cf0 = build_compressed_factual(θ0, ctx; check_ties = false)
has_france = cf0.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
println("LIVE DIMENSIONS: active_A=$(D*Ddest)  retained_factual=$(length(layout.retained_full_factual_j))  france_ratio_present=$has_france  total_reduced=$(layout.total_reduced_economic_moments)")
@assert D * Ddest == 380
@assert length(layout.retained_full_factual_j) == 361
@assert layout.total_reduced_economic_moments == (has_france ? 362 : 361)
flush(stdout)

z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
n_free_after_pivot = length(pe.other_pos)
println("free_A_after_gravity_pivot = $n_free_after_pivot  (expect 360)")
@assert n_free_after_pivot == 360
flush(stdout)
gp0 = θ0[3+D]
w_profiled_calib = reduce_to_w_profiled(gp0, z_calib, pe)
decoded0 = decode_outer_profiled(w_profiled_calib, ctx, pe)
θ_calib_profiled = copy(θ0)
θ_calib_profiled[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest] .= decoded0.Aod_levels
θ_calib_profiled[3+D] = decoded0.gp
# NOTE: real D20 calibrated A_od spans ~15 orders of magnitude here (extrema of decoded0.Aod_levels
# below) -- this repo's own CLAUDE.md standing warning applies: comparing round-trip fidelity via an
# ABSOLUTE level-space difference is meaningless at this scale (a perfect ~1e-14 LOG-space match
# produces an absolute level-space difference of ~1e9 purely because exp(53.8)~2e23). Compare in
# LOG-SPACE (z, where the round-trip actually operates) instead, matching decode_full_z_on_retained's
# own domain.
println("extrema(decoded0.Aod_levels) = ", extrema(decoded0.Aod_levels))
roundtrip_diff_logspace = maximum(abs.(decoded0.z_full .- z_calib))
roundtrip_diff_gp = abs(decoded0.gp - gp0)
println("round-trip diff (log-space z) = $roundtrip_diff_logspace ; gp diff = $roundtrip_diff_gp")
@assert roundtrip_diff_logspace < 1e-8 "D20 profiled outer round-trip did not reproduce calibration (log-space)"
@assert roundtrip_diff_gp < 1e-9 "D20 profiled outer round-trip did not reproduce calibration gp"
flush(stdout)

function run_point(label::String, θ_full_point::Vector{Float64})
    println("\n" * "#"^90); println("D20 POINT: $label"); println("#"^90); flush(stdout)
    obj_p, st_p = build_profiled_operator_bundle(ctx, θ_full_point, spec; ref_obj = ctx.obj)
    SW = ctx.γ.SamplingWeights[1:obj_p.M]
    obj_p.payoff .= θ_full_point[3+D] .* SW
    obj_p.H_save = obj_p.payoff[1] * (-1.0)^obj_p.find_smallest
    println("outer_constr_index=$(obj_p.outer_constr_index)  W=$(obj_p.M)"); flush(stdout)
    # Using the SHARED default maxit=100 (no override) this time -- testing whether the earlier
    # W=15,000 non-convergence was simply a too-small-W artifact (per user correction) rather than
    # a real scaling/conditioning problem needing a larger iteration budget.
    t_solve_p = @elapsed (nStatus_p, objSol_p, x_p, lambda_p, nfg_p, nhess_p) = inner_loop_KNITRO_profiled(obj_p, st_p)
    println("profiled solve: nStatus=$nStatus_p  time=$(t_solve_p)s  n_fg=$nfg_p  n_hess=$nhess_p"); flush(stdout)
    p_ok = nStatus_p ∈ [0, -100, -101, -103, -400, -401, -402]
    zeta_p = x_p[1]; beta_p = x_p[2:end]
    ov_p = verify_inner_solution_reduced_profiled!(zeta_p, beta_p, st_p.cf, ctx, θ_full_point, st_p.layout, obj_p, st_p.cf.W)
    mw_p, verify_p = verify_namedtuple_from_operator(ov_p, obj_p, st_p.cf.W, nStatus_p)
    println("profiled verify: Delta_primal=$(verify_p.Delta_primal)  kkt=$(verify_p.max_abs_moment_kkt_resid)"); flush(stdout)

    z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(θ_full_point, ctx, st_p.cf, mw_p)
    gamma_dev = maximum(abs.(gamma_tilde .- 1.0))
    println("gamma_tilde max deviation from 1 (pre-recovery) = $gamma_dev"); flush(stdout)
    θ_full_recovered = copy(θ_full_point)
    θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest] .= vec(exp.(z_recovered))

    ctx.obj.use_cached_x = false
    ctx.obj.x .= NaN
    t_solve_ref = @elapsed (K_ref, x_ref, nStatus_ref, nfg_ref, nhess_ref, st_ref) = inner_loop_internal_compressed(ctx.obj, θ_full_recovered, ctx)
    println("reference@recovered solve: nStatus=$nStatus_ref  time=$(t_solve_ref)s"); flush(stdout)
    ref_ok = nStatus_ref ∈ [0, -100, -101, -103, -400, -401, -402]
    zeta_ref = x_ref[1]; lambda_ref = x_ref[2:end]
    ov_ref = verify_inner_solution_operator_unrestricted!(zeta_ref, lambda_ref, st_ref.cf, ctx.obj, st_ref.cf.W)
    mw_ref, verify_ref = verify_namedtuple_from_operator(ov_ref, ctx.obj, st_ref.cf.W, nStatus_ref)
    println("reference verify: Delta_primal=$(verify_ref.Delta_primal)  kkt=$(verify_ref.max_abs_moment_kkt_resid)"); flush(stdout)

    winner_checksum_ref = sum(Float64.(st_ref.cf.winner))
    winner_checksum_p = sum(Float64.(st_p.cf.winner))
    mw_diff = maximum(abs.(mw_ref .- mw_p))
    div_diff = abs(verify_ref.Delta_primal - verify_p.Delta_primal)
    println(@sprintf("RESULT %s: p_ok=%s ref_ok=%s winner_match=%s mw_diff=%.3e div_diff=%.3e gamma_dev=%.3e",
        label, p_ok, ref_ok, winner_checksum_ref == winner_checksum_p, mw_diff, div_diff, gamma_dev))
    flush(stdout)
    return (label = label, p_ok = p_ok, ref_ok = ref_ok, nStatus_p = nStatus_p, nStatus_ref = nStatus_ref,
        Delta_primal_p = verify_p.Delta_primal, Delta_primal_ref = verify_ref.Delta_primal,
        mw_diff = mw_diff, div_diff = div_diff, winner_match = (winner_checksum_ref == winner_checksum_p),
        gamma_dev = gamma_dev, kkt_p = verify_p.max_abs_moment_kkt_resid, kkt_ref = verify_ref.max_abs_moment_kkt_resid,
        t_solve_p = t_solve_p, t_solve_ref = t_solve_ref)
end

results = []
push!(results, run_point("calibration", θ_calib_profiled))

# modest perturbation: small deterministic additive shift on r_free
n_rfree = length(w_profiled_calib) - 1
direction = [(-1.0)^k for k in 1:n_rfree]
w_profiled_pert = copy(w_profiled_calib)
w_profiled_pert[2:end] .+= 0.02 .* direction
decoded_pert = decode_outer_profiled(w_profiled_pert, ctx, pe)
θ_pert = copy(θ_calib_profiled)
θ_pert[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest] .= decoded_pert.Aod_levels
θ_pert[3+D] = decoded_pert.gp
push!(results, run_point("modest_perturbation", θ_pert))

outpath = joinpath(@__DIR__, "..", "..", "UNRESTRICTED_KNITRO_OMIT_ROW_D20_SMALLW_2026-08-01.csv")
open(outpath, "w") do io
    println(io, "label,p_ok,ref_ok,nStatus_p,nStatus_ref,Delta_primal_p,Delta_primal_ref,mw_diff,div_diff,winner_match,gamma_dev,kkt_p,kkt_ref,t_solve_p_sec,t_solve_ref_sec")
    for r in results
        @printf(io, "%s,%s,%s,%d,%d,%.10e,%.10e,%.6e,%.6e,%s,%.6e,%.6e,%.6e,%.3f,%.3f\n",
            r.label, r.p_ok, r.ref_ok, r.nStatus_p, r.nStatus_ref, r.Delta_primal_p, r.Delta_primal_ref,
            r.mw_diff, r.div_diff, r.winner_match, r.gamma_dev, r.kkt_p, r.kkt_ref, r.t_solve_p, r.t_solve_ref)
    end
end
println("Wrote $outpath")

tol = 1e-3
all_pass = all(r -> r.mw_diff < tol && r.div_diff < tol * max(1.0, abs(r.Delta_primal_p)) && r.winner_match && r.p_ok && r.ref_ok, results)
println("\nALL D20 POINTS PASS (tol=$tol): $all_pass")
for r in results
    println("  $(r.label): mw_diff=$(r.mw_diff)  div_diff=$(r.div_diff)  winner_match=$(r.winner_match)")
end
if all_pass
    println("\n" * "="^90); println("D20 OMIT-ROW SMALL-W GATE: EQUIVALENT"); println("="^90)
else
    println("\n" * "="^90); println("D20 OMIT-ROW SMALL-W GATE: DID NOT PASS AT STATED TOLERANCE"); println("="^90)
end
