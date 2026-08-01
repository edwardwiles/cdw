# ============================================================================
# Claude Code task 2026-08-01, follow-up (live user request, verification):
# "verify your profiled solutions by using the LFD to evaluate all of the
# trade shares and the gravity regression. I am a bit suspicious of
# differences that large [509x]."
#
# For BOTH arms' completed gp=0.99 best points:
#   (a) reconstruct the LFD (m_weights) at the final solution;
#   (b) compute the LFD-weighted implied trade share for EVERY (o, active
#       destination) bilateral cell: E_LFD[Q_od]/E_LFD[M_d], compared against
#       the FACTUAL data share lambda_od (cf.Pmat) -- if Delta_dual is small,
#       these should be close (that IS what Delta_dual measures);
#   (c) recover the gamma-normalized full A from that LFD
#       (recover_gamma_normalized_full_A_from_lfd, already-gated) and check
#       the gravity regression residual on it (gravity_from_logz) -- both at
#       the WORKING-GAUGE point (should be exactly 0 by pivot construction)
#       and at the RECOVERED point (a genuinely independent check).
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
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
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
using Printf, Statistics, Serialization

const W = 80_000
const DELTA = 1.0

println("Building ctx..."); flush(stdout)
ctx = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
D = ctx.D; Ddest = ctx.D_dest
spec, gauge, pe_profiled = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
pe_full = build_pivot_elimination(ctx)

"""
    lfd_share_report(cf, m_weights, label) -> NamedTuple

For every (o, active-destination) bilateral cell, computes the LFD-weighted
implied share E_LFD[Q_od]/E_LFD[M_d] and compares to the factual data share
cf.Pmat[o,slot]. Q_od[w] = wval[w,slot] if winner[w,slot]==o else 0 (the raw,
un-normalized bilateral indicator*value the homogeneous moment is built from).
"""
function lfd_share_report(cf::CompressedFactual, m_weights::AbstractVector{Float64}, label::String)
    Dl = cf.D; Ddl = cf.D_dest; Wl = cf.W
    E_M = zeros(Ddl)   # E_LFD[M_d]
    E_Q = zeros(Dl, Ddl)  # E_LFD[Q_od]
    @inbounds for slot in 1:Ddl
        for w in 1:Wl
            wm = cf.SW[w] * m_weights[w]
            v = wm * cf.wval[w, slot]
            E_M[slot] += v
            E_Q[cf.winner[w, slot], slot] += v
        end
    end
    total_w = sum(cf.SW[w] * m_weights[w] for w in 1:Wl)
    E_M ./= total_w; E_Q ./= total_w

    diffs = Float64[]
    worst = (diff = 0.0, o = 0, s = 0, implied = 0.0, factual = 0.0)
    for slot in 1:Ddl, o in 1:Dl
        implied_share = E_M[slot] > 0 ? E_Q[o, slot] / E_M[slot] : NaN
        factual_share = cf.Pmat[o, slot]
        d = abs(implied_share - factual_share)
        push!(diffs, d)
        if d > worst.diff
            worst = (diff = d, o = o, s = slot, implied = implied_share, factual = factual_share)
        end
    end
    @printf("[%s] LFD-implied vs factual bilateral shares: mean|diff|=%.3e  max|diff|=%.3e  median|diff|=%.3e\n",
        label, mean(diffs), maximum(diffs), median(diffs))
    @printf("[%s]   worst cell: o=%d slot=%d implied=%.6f factual=%.6f diff=%.3e\n",
        label, worst.o, worst.s, worst.implied, worst.factual, worst.diff)
    return (mean_diff = mean(diffs), max_diff = maximum(diffs), median_diff = median(diffs), worst = worst,
            E_M = E_M, E_Q = E_Q)
end

# ---------------------------------------------------------------------------
# PROFILED arm: parse the best w from the completed gp=0.99 run's log.
# ---------------------------------------------------------------------------
println("\n" * "="^90); println("PROFILED ARM (gp=0.99 fixed-iteration best point)"); println("="^90); flush(stdout)
w_str = read(joinpath(@__DIR__, "profiled_w_best_gp099.txt"), String)
w_profiled_best = Float64.(eval(Meta.parse(w_str)))
@assert length(w_profiled_best) == outer_dim_profiled(pe_profiled) "parsed w has length $(length(w_profiled_best)), expected $(outer_dim_profiled(pe_profiled))"
println("gp (profiled best) = ", w_profiled_best[1])

ev_p = evaluate_profiled_point(w_profiled_best, ctx, spec, pe_profiled)
@printf("Re-evaluated: Delta_dual=%.6e  inner_status=%d  (expect ~2.976e-4, status=0, matching the completed run)\n",
    ev_p.result.Delta_dual, ev_p.result.inner_status)
flush(stdout)

rep_p = lfd_share_report(ev_p.st.cf, ev_p.m_weights, "PROFILED")

# gravity check at the WORKING-GAUGE point (should be exactly 0 by pivot construction)
grav_working_p = gravity_from_logz(ev_p.decoded.z_full, ctx)
println("[PROFILED] gravity residual at working-gauge z_full = ", grav_working_p, "  (expect ~0, pivot-feasible by construction)")

# gravity check at the RECOVERED (gamma-normalized) point -- independent check
z_recovered_p, c_recover_p, gamma_tilde_p = recover_gamma_normalized_full_A_from_lfd(ev_p.theta_full, ctx, ev_p.st.cf, ev_p.m_weights)
grav_recovered_p = gravity_from_logz(z_recovered_p, ctx)
println("[PROFILED] gravity residual at RECOVERED z_full = ", grav_recovered_p, "  (independent check -- recovery is a per-destination rescale, not obviously gravity-neutral until checked)")
println("[PROFILED] gamma_tilde (LFD-weighted E[M_d]/denom[d], deviation from 1 means recovery is non-trivial): max|dev|=", maximum(abs.(gamma_tilde_p .- 1)))
flush(stdout)

# ---------------------------------------------------------------------------
# FULL/REFERENCE arm: load the best checkpoint from disk (D20CheckpointV4).
# ---------------------------------------------------------------------------
println("\n" * "="^90); println("FULL/REFERENCE ARM (gp=0.99 fixed-iteration best point)"); println("="^90); flush(stdout)
ckpt_path = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "full_fixediter_gplow", "full_fixediter_gplow_stage_complete_neval121.jls")
ckpt = load_checkpoint(ckpt_path)
println("Loaded checkpoint: gp=", ckpt.g, "  Delta_dual(verify)=", ckpt.verify_Delta_dual)
θ_full_ref = copy(ctx.θ0_up)
θ_full_ref[3+D] = ckpt.g
θ_full_ref[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest] .= vec(exp.(ckpt.logA_full))

ctx.obj.use_cached_x = false; ctx.obj.x .= NaN
K_ref, x_ref, nStatus_ref, nfg_ref, nhess_ref, st_ref = inner_loop_internal_compressed(ctx.obj, θ_full_ref, ctx)
zeta_ref = x_ref[1]; lambda_ref = x_ref[2:end]
ov_ref = verify_inner_solution_operator_unrestricted!(zeta_ref, lambda_ref, st_ref.cf, ctx.obj, st_ref.cf.W)
mw_ref, verify_ref = verify_namedtuple_from_operator(ov_ref, ctx.obj, st_ref.cf.W, nStatus_ref)
@printf("Re-evaluated: Delta_dual=%.6e  inner_status=%d  (expect ~0.1517, status=0, matching the completed run)\n",
    verify_ref.Delta_dual, nStatus_ref)
flush(stdout)

rep_ref = lfd_share_report(st_ref.cf, mw_ref, "FULL/REFERENCE")

grav_working_ref = gravity_from_logz(ckpt.logA_full, ctx)
println("[FULL] gravity residual at working-gauge logA_full = ", grav_working_ref, "  (expect ~0, pivot-feasible by construction)")

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------
println("\n" * "="^90); println("SUMMARY"); println("="^90)
@printf("PROFILED: Delta_dual=%.4e  mean_share_diff=%.3e  max_share_diff=%.3e  gravity(working)=%.3e  gravity(recovered)=%.3e\n",
    ev_p.result.Delta_dual, rep_p.mean_diff, rep_p.max_diff, grav_working_p, grav_recovered_p)
@printf("FULL:     Delta_dual=%.4e  mean_share_diff=%.3e  max_share_diff=%.3e  gravity(working)=%.3e\n",
    verify_ref.Delta_dual, rep_ref.mean_diff, rep_ref.max_diff, grav_working_ref)
println("\nIf PROFILED's mean/max share diffs are correspondingly SMALLER than FULL's (proportional to the")
println("Delta_dual gap), the 509x Delta_dual gap reflects a genuinely better-fitting solution, not a bug.")
println("If PROFILED's share diffs are NOT correspondingly smaller (e.g. comparable to or larger than FULL's")
println("despite a much smaller Delta_dual), that is a red flag worth investigating further.")
