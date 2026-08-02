# ============================================================================
# Claude Code task 2026-08-01, §11/§12 (theory doc §2.3, the actual comparison
# theorem): full-A recovery using the REDUCED/PROFILED SOLVE'S OWN VERIFIED
# LFD, not the factual/uniform measure `recover_gamma_normalized_full_A`
# (recover_full_a_2026-07-31.jl) uses.
#
# WHY THIS FILE EXISTS (a real finding from this session, not assumed): a
# direct comparison of the reference (old, fixed-denom) and profiled (new,
# homogeneous) inner solves AT THE SAME calibration theta, with NO recovery
# step, does NOT numerically agree (~1e-4 absolute / ~11% relative on
# Delta_dual, up to ~10% on individual LFD weights, in the D=4 gate -- see
# UNRESTRICTED_KNITRO_CALIBRATION_EQUIVALENCE_D4_2026-08-01.csv). This is
# NOT a bug: every building-block gate (forward/transpose exact adjointness,
# exact zero-anchor equivalence to the OLD full kernel, Hessian vs finite
# difference AND vs an exact dense reduced-G construction, rank/redundancy)
# passed at machine precision. The reason is that BOTH the old (fixed
# `denom[d]`) and new (homogeneous, `M_d(omega)`) moment systems are
# satisfied EXACTLY by the factual/uniform measure at calibration -- but the
# actual solved LFD (`m_weights = dPsi(r)`) is NOT the uniform measure (this
# session's own gate output shows `m_min`/`m_max` far from 1 for both solves)
# it is a TILTED distribution chosen to rationalize the outer `gp` target via
# the divergence-minimization criterion. Under that tilted measure, `E_LFD
# [M_d(omega)]` need not equal the fixed `denom[d]` (the homogeneous moment
# only pins the SHARES `E_LFD[Q_od]/E_LFD[M_d]`, not the scale `E_LFD[M_d]`
# itself -- exactly the scale-invariance property this whole reparameterization
# exploits). So the two dual programs are genuinely different constraint sets
# on the primal weights whenever the solved LFD departs from uniform, and
# equality is NOT expected from a naive same-theta comparison.
#
# Theory doc `PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md` section 2.3
# states the ACTUAL comparison theorem precisely: (i) solve the reduced
# problem; (ii) recover the full gamma-normalized A using the REDUCED
# problem's OWN VERIFIED LFD (not the factual measure); (iii) solve the
# LEGACY FULL problem AT THAT RECOVERED A. This file implements step (ii).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :destination_M_d) || error("reduced_recovery_from_lfd_2026-08-01.jl requires recover_full_a_2026-07-31.jl to be included first.")

"""
    recover_gamma_normalized_full_A_from_lfd(θ_working, ctx, cf, m_weights; d_list=1:ctx.D) -> (z_full, c, gamma_tilde)

Same recovery formula as `recover_gamma_normalized_full_A`
(`c[d] = gamma_tilde[d]^(-1/(mu*(sigma-1)))`, `z_full[:,d] = z_working[:,d] +
log(c[d])`), but `gamma_tilde[d] = E_LFD[M_d(omega)] / denom[d]`, where
`E_LFD` is the LFD-WEIGHTED expectation (`sum_w SW[w]*m_weights[w]*M_d(w) /
sum_w SW[w]*m_weights[w]`) instead of the plain factual-measure average
`recover_gamma_normalized_full_A` uses. `cf` must already be built at
`θ_working` (same point `m_weights` was solved at); `m_weights` is
`dPsi(r)` from a `verify_inner_solution_*!` call at that same solve.
"""
function recover_gamma_normalized_full_A_from_lfd(θ_working::AbstractVector{Float64}, ctx, cf::CompressedFactual,
        m_weights::AbstractVector{Float64}; d_list = active_destinations(ctx))
    D = ctx.D
    Ddest = length(active_destinations(ctx))
    Aod_offset = ctx.Aod_offset
    μ = θ_working[1]; σ = θ_working[2]
    e_exponent = μ * (σ - 1)
    W = cf.W
    weight_total = sum(cf.SW[w] * m_weights[w] for w in 1:W)

    # RECTANGULAR-SAFE (task §5): D*Ddest block size / (D,Ddest) reshape / per-SLOT column indexing
    # -- the original recover_gamma_normalized_full_A (recover_full_a_2026-07-31.jl) this mirrors
    # hardcodes D^2 and a (D,D) reshape, which is wrong whenever Ddest != D (real D=20 :exclude_row,
    # Ddest=19). Fixed here rather than in that file (additive-only convention).
    c = ones(Ddest)
    gamma_tilde = ones(Ddest)
    for d in d_list
        s = dest_slot(ctx, d)
        Md = cf.wval[:, s]
        weighted_sum = sum(cf.SW[w] * m_weights[w] * Md[w] for w in 1:W)
        e_lfd_Md = weighted_sum / weight_total
        denom_d = cf.denom[s]
        gt = e_lfd_Md / denom_d
        gamma_tilde[s] = gt
        c[s] = gt^(-1 / e_exponent)
    end
    Aod_θ_working = reshape(θ_working[Aod_offset+1:Aod_offset+D*Ddest], (D, Ddest))
    z_working = log.(Aod_θ_working)
    z_full = copy(z_working)
    for d in d_list
        s = dest_slot(ctx, d)
        z_full[:, s] .+= log(c[s])
    end
    return z_full, c, gamma_tilde
end
