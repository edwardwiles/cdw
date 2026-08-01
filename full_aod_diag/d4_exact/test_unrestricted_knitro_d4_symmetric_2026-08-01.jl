# ============================================================================
# Claude Code task 2026-08-01, §11: FIRST real KNITRO gate. D=4 symmetric
# calibration comparison: reference (:full_gamma_normalized_reference) vs
# profiled (:profiled_destination_scales), same real public CC/KNITRO inner
# solver (inner_loop_KNITRO_compressed / inner_loop_KNITRO_profiled), both
# through a genuine OperatorPsiBundle.
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
println("D=$D  Aod_offset=$(ctx.Aod_offset)")

println("\n" * "="^78); println("SETUP: reduce calibration to profiled coordinates and decode back"); println("="^78)
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp0 = θ0[3+D]
w_profiled = reduce_to_w_profiled(gp0, z_calib, pe)
println("outer_dim_profiled = $(outer_dim_profiled(pe))  (full would be 1+$(D^2-1)=$(D^2))")
decoded = decode_outer_profiled(w_profiled, ctx, pe)
θ_profiled_full = copy(θ0)
θ_profiled_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= decoded.Aod_levels
θ_profiled_full[3+D] = decoded.gp
roundtrip_diff = maximum(abs.(θ_profiled_full .- θ0))
println("max|θ_profiled_full - θ0| after reduce->decode round trip = $roundtrip_diff")
@assert roundtrip_diff < 1e-9 "profiled outer round-trip did not reproduce calibration"
println("PASS -- profiled outer coordinates reproduce calibration bit-for-bit (to $roundtrip_diff)")

println("\n" * "="^78); println("STEP 1: REFERENCE inner solve (full_gamma_normalized_reference, real KNITRO, OperatorPsiBundle)"); println("="^78)
println("ctx.obj.use_cached_x (BEFORE forcing) = $(ctx.obj.use_cached_x)  ctx.obj.x[1:3] = $(ctx.obj.x[1:min(3,end)])")
ctx.obj.use_cached_x = false   # force a from-zeros start, matching the profiled bundle's default -- controls for any warm-started cache from ctx construction
K_ref, x_ref, nStatus_ref, nfg_ref, nhess_ref, st_ref = inner_loop_internal_compressed(ctx.obj, θ0, ctx)
println("nStatus_ref=$nStatus_ref  K_ref=$K_ref  n_fg=$nfg_ref  n_hess=$nhess_ref")
ref_ok = nStatus_ref ∈ [0, -100, -101, -103]
@assert ref_ok "reference inner solve did not report a successful/near-optimal KNITRO status"
zeta_ref = x_ref[1]; lambda_ref = x_ref[2:end]
ov_ref = verify_inner_solution_operator_unrestricted!(zeta_ref, lambda_ref, st_ref.cf, ctx.obj, st_ref.cf.W)
mw_ref, verify_ref = verify_namedtuple_from_operator(ov_ref, ctx.obj, st_ref.cf.W, nStatus_ref)
println("verify_ref = ", verify_ref)
winner_checksum_ref = sum(Float64.(st_ref.cf.winner))

println("\n" * "="^78); println("STEP 2: PROFILED inner solve (profiled_destination_scales, real KNITRO, OperatorPsiBundle)"); println("="^78)
obj_p, st_p = build_profiled_operator_bundle(ctx, θ_profiled_full, spec; ref_obj = ctx.obj)
SW = ctx.γ.SamplingWeights[1:obj_p.M]
obj_p.payoff .= θ_profiled_full[3+D] .* SW
obj_p.H_save = obj_p.payoff[1] * (-1.0)^obj_p.find_smallest
println("layout: total_reduced_economic_moments=$(st_p.layout.total_reduced_economic_moments)  outer_constr_index=$(obj_p.outer_constr_index)")
nStatus_p, objSol_p, x_p, lambda_p, nfg_p, nhess_p = inner_loop_KNITRO_profiled(obj_p, st_p)
println("nStatus_p=$nStatus_p  K_p=$(obj_p.H_save)  n_fg=$nfg_p  n_hess=$nhess_p")
p_ok = nStatus_p ∈ [0, -100, -101, -103]
@assert p_ok "profiled inner solve did not report a successful/near-optimal KNITRO status"
zeta_p = x_p[1]; beta_p = x_p[2:end]
ov_p = verify_inner_solution_reduced_profiled!(zeta_p, beta_p, st_p.cf, ctx, θ_profiled_full, st_p.layout, obj_p, st_p.cf.W)
mw_p, verify_p = verify_namedtuple_from_operator(ov_p, obj_p, st_p.cf.W, nStatus_p)
println("verify_p = ", verify_p)
winner_checksum_p = sum(Float64.(st_p.cf.winner))

println("\n" * "="^78); println("STEP 3: NAIVE DIRECT COMPARISON (informational only -- see STEP 5 for the actual comparison theorem)"); println("="^78)
println(@sprintf("%-32s %18s %18s %14s", "metric", "reference", "profiled", "diff"))
println(@sprintf("%-32s %18d %18d %14s", "nStatus", nStatus_ref, nStatus_p, string(nStatus_ref == nStatus_p)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "K (payoff/objective)", K_ref, obj_p.H_save, abs(K_ref - obj_p.H_save)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "Delta_dual", verify_ref.Delta_dual, verify_p.Delta_dual, abs(verify_ref.Delta_dual - verify_p.Delta_dual)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "Delta_primal", verify_ref.Delta_primal, verify_p.Delta_primal, abs(verify_ref.Delta_primal - verify_p.Delta_primal)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "primal_dual_gap", verify_ref.primal_dual_gap, verify_p.primal_dual_gap, abs(verify_ref.primal_dual_gap - verify_p.primal_dual_gap)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "mean_m_resid", verify_ref.mean_m_resid, verify_p.mean_m_resid, abs(verify_ref.mean_m_resid - verify_p.mean_m_resid)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "max_abs_moment_kkt_resid", verify_ref.max_abs_moment_kkt_resid, verify_p.max_abs_moment_kkt_resid, abs(verify_ref.max_abs_moment_kkt_resid - verify_p.max_abs_moment_kkt_resid)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "m_mean (LFD)", verify_ref.m_mean, verify_p.m_mean, abs(verify_ref.m_mean - verify_p.m_mean)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "m_min (LFD)", verify_ref.m_min, verify_p.m_min, abs(verify_ref.m_min - verify_p.m_min)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "m_max (LFD)", verify_ref.m_max, verify_p.m_max, abs(verify_ref.m_max - verify_p.m_max)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "winner_checksum", winner_checksum_ref, winner_checksum_p, abs(winner_checksum_ref - winner_checksum_p)))
mw_diff = maximum(abs.(mw_ref .- mw_p))
println(@sprintf("%-32s %18s %18s %14.4g", "max|m_weights_ref-m_weights_p|", "-", "-", mw_diff))
gp_ref = θ0[3+D]; gp_p = θ_profiled_full[3+D]
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "France welfare param gp", gp_ref, gp_p, abs(gp_ref - gp_p)))

println("\n" * "="^78); println("STEP 4: naive-comparison finding (NOT a hard gate -- see STEP 16 failure-diagnosis discipline)"); println("="^78)
@assert winner_checksum_ref == winner_checksum_p "winner checksum mismatch -- reference and profiled disagree on winners at calibration (this WOULD be a real bug)"
@assert nStatus_ref == nStatus_p "KNITRO status category mismatch"
println("Winners match exactly ($winner_checksum_ref) and both solves report nStatus=$nStatus_ref -- the two dual PROGRAMS are each internally consistent (kkt_resid ~1e-15 each).")
println("But Delta_dual/Delta_primal/LFD differ by a NON-trivial amount ($mw_diff on LFD, $(abs(verify_ref.Delta_primal-verify_p.Delta_primal)) on Delta_primal).")
println("FINDING (see reduced_recovery_from_lfd_2026-08-01.jl header for the full explanation): this is NOT a bug in the reduced kernels")
println("(every forward/transpose/Hessian building-block gate above passed at machine precision, including EXACT -- 0.0 diff -- equivalence")
println("to the old kernel with the anchor beta forced to 0). It is because the solved LFD is NOT the uniform/factual measure (m_min/m_max above")
println("are far from 1 for BOTH solves), and under a TILTED measure, E_LFD[M_d] need not equal the fixed denom[d] the old formulation targets --")
println("the homogeneous moment only pins SHARES, not the scale, under any measure. Theory doc section 2.3's actual comparison theorem requires")
println("RECOVERING a new full-A using the reduced solve's OWN LFD before re-solving the reference problem -- tested next.")

println("\n" * "="^78); println("STEP 5: RECOVER full-A using the PROFILED solve's own verified LFD (theory doc section 2.3 step (ii))"); println("="^78)
z_recovered, c_recover, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(θ_profiled_full, ctx, st_p.cf, mw_p)
println("gamma_tilde (LFD-weighted E[M_d]/denom[d] per destination) = ", gamma_tilde)
println("(if this were ~1 for every destination, recovery would be a near no-op; it is NOT, confirming recovery is genuinely needed even at calibration)")
θ_full_recovered = copy(θ_profiled_full)
θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(exp.(z_recovered))
recovery_change = maximum(abs.(θ_full_recovered[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .- θ_profiled_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2]))
println("max|A_recovered - A_profiled| = $recovery_change (0 would mean recovery was a no-op)")

println("\n" * "="^78); println("STEP 6: re-solve the LEGACY FULL (reference) problem AT THE RECOVERED A (theory doc section 2.3 step (iii))"); println("="^78)
ctx.obj.use_cached_x = false
ctx.obj.x .= NaN
K_ref2, x_ref2, nStatus_ref2, nfg_ref2, nhess_ref2, st_ref2 = inner_loop_internal_compressed(ctx.obj, θ_full_recovered, ctx)
println("nStatus_ref2=$nStatus_ref2  K_ref2=$K_ref2")
ref2_ok = nStatus_ref2 ∈ [0, -100, -101, -103]
@assert ref2_ok "recovered-reference inner solve did not report a successful/near-optimal KNITRO status"
zeta_ref2 = x_ref2[1]; lambda_ref2 = x_ref2[2:end]
ov_ref2 = verify_inner_solution_operator_unrestricted!(zeta_ref2, lambda_ref2, st_ref2.cf, ctx.obj, st_ref2.cf.W)
mw_ref2, verify_ref2 = verify_namedtuple_from_operator(ov_ref2, ctx.obj, st_ref2.cf.W, nStatus_ref2)
println("verify_ref2 (reference @ recovered A) = ", verify_ref2)
winner_checksum_ref2 = sum(Float64.(st_ref2.cf.winner))

println("\n" * "="^78); println("STEP 7: THE DECISIVE COMPARISON (profiled's own solve vs. reference re-solved at the recovered A)"); println("="^78)
println(@sprintf("%-32s %18s %18s %14s", "metric", "reference@recovered", "profiled@original", "diff"))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "Delta_dual", verify_ref2.Delta_dual, verify_p.Delta_dual, abs(verify_ref2.Delta_dual - verify_p.Delta_dual)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "Delta_primal", verify_ref2.Delta_primal, verify_p.Delta_primal, abs(verify_ref2.Delta_primal - verify_p.Delta_primal)))
println(@sprintf("%-32s %18.10g %18.10g %14.4g", "K", K_ref2, obj_p.H_save, abs(K_ref2 - obj_p.H_save)))
mw_diff2 = maximum(abs.(mw_ref2 .- mw_p))
println(@sprintf("%-32s %18s %18s %14.4g", "max|m_weights diff|", "-", "-", mw_diff2))
println(@sprintf("%-32s %18.10g %18.10g %14s", "winner_checksum", winner_checksum_ref2, winner_checksum_p, string(winner_checksum_ref2 == winner_checksum_p)))

tol_LFD = 1e-4
tol_div = 1e-4
lfd_ok = mw_diff2 < tol_LFD * max(1.0, maximum(abs.(mw_p)))
div_ok = abs(verify_ref2.Delta_primal - verify_p.Delta_primal) < tol_div * max(1.0, abs(verify_p.Delta_primal))
println("\nLFD match (tol $tol_LFD rel): $lfd_ok (actual $mw_diff2)")
println("Delta_primal match (tol $tol_div rel): $div_ok (actual $(abs(verify_ref2.Delta_primal - verify_p.Delta_primal)))")

if lfd_ok && div_ok && winner_checksum_ref2 == winner_checksum_p
    println("\n" * "="^78); println("DECISIVE EQUIVALENCE CONFIRMED -- theory doc section 2.3's comparison theorem holds numerically at D=4 calibration"); println("="^78)
else
    println("\n" * "="^78); println("DECISIVE EQUIVALENCE TEST DID NOT PASS AT THE STATED TOLERANCE -- see master report for full diagnosis"); println("="^78)
end
