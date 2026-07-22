# Part A / remediation task: live verification of the F1 identity at a real D=20/W=80,000/L=50
# CM point. Confirms or refutes, against the CURRENT (pre-fix) production code, that
#   Delta_dual = -(mean(Psi(q*)) + zeta*)
#   (-zeta*) - Delta_dual == mean(Psi(q*)) > 0   at a tail-active point (m_max > e)
# and prints the quantity the CURRENT cb_F! stores as "Delta" in the checkpoint (-zeta*) next to
# the canonical verify.Delta_dual, so the two can be compared directly. Nothing in production is
# modified by this script.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
using Printf, LinearAlgebra, Statistics

lp(xs...) = (println(xs...); flush(stdout))

W = 80000; DELTA = 1.0; L = 50
lp(">>> building D20 real-data context, W=$W ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
lp(@sprintf(">>> ctx built in %.1fs. D=%d", time() - t0, D))

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]
lp(">>> building CM production context (Architecture C, archB moments, L=$L, anchored contrasts) ...")
t1 = time()
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = probs)
lp(@sprintf(">>> pcx built in %.1fs", time() - t1))

x_free_calib = ctx.θ0_up[ctx.free_idx]
gp_calib = x_free_calib[1]

function report_point(label, x_free)
    lp(">>> ", label, " x_free[1] (gamma'_focal) = ", x_free[1])
    K, base, verify = cm_production_value_verified(x_free, pcx)
    zeta_star = base.ζstar
    Delta_dual = verify.Delta_dual
    naive_minus_zeta = -zeta_star

    # recompute mean(Psi(q*)) independently (not reusing verify's internals) to cross-check the
    # identity from a second code path
    obj = pcx.ctx_cm.obj
    oci = obj.outer_constr_index
    θ_full = CS.reconstruct_full(x_free, pcx.ctx_cm.m)
    Wn = size(obj.U, 1)
    Kk = zeros(Wn); G = zeros(Wn, obj.d)
    obj.moments!(Kk, G, θ_full, obj.U, obj)
    q = [-zeta_star - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:Wn]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    mean_Psi_q = sum(Psi_q) / Wn
    mean_m = mean(base.m_star)
    m_max = maximum(base.m_star)
    frac_tail = count(>(ℯ), base.m_star) / Wn

    identity1_resid = abs(Delta_dual - (-(mean_Psi_q + zeta_star)))
    diff = naive_minus_zeta - Delta_dual

    lp(@sprintf("    inner_status         = %d", base.inner_status))
    lp(@sprintf("    zeta_star            = %.10f", zeta_star))
    lp(@sprintf("    Delta_dual (verify)  = %.10f", Delta_dual))
    lp(@sprintf("    -zeta_star (checkpoint's current stored 'Delta')  = %.10f", naive_minus_zeta))
    lp(@sprintf("    mean(Psi(q*))        = %.10f", mean_Psi_q))
    lp(@sprintf("    mean(m*)             = %.10f  (should be ~1 at a converged solve)", mean_m))
    lp(@sprintf("    m_max                = %.6f   frac(m*>e)=%.4f", m_max, frac_tail))
    lp(@sprintf("    primal_dual_gap      = %.3e", verify.primal_dual_gap))
    lp(@sprintf("    identity check |Delta_dual - (-(mean(Psi(q*))+zeta*))| = %.3e  (should be ~0)", identity1_resid))
    lp(@sprintf("    (-zeta*) - Delta_dual = %.10f   vs mean(Psi(q*)) = %.10f   |diff| = %.3e",
                diff, mean_Psi_q, abs(diff - mean_Psi_q)))
    return (label = label, zeta_star = zeta_star, Delta_dual = Delta_dual, naive_minus_zeta = naive_minus_zeta,
            mean_Psi_q = mean_Psi_q, mean_m = mean_m, m_max = m_max, frac_tail = frac_tail,
            identity1_resid = identity1_resid, diff = diff)
end

lp("=== Point 1: calibration point (gp=gp_calib, A=A_calib) ===")
r1 = report_point("calibration", x_free_calib)

lp("")
lp("=== Point 2: perturbed gp (away from calibration, toward delta=1 frontier -- more likely tail-active) ===")
# perturb gamma'_focal downward (tighter divergence budget => more mass pushed into the tail)
x_free_pert = copy(x_free_calib)
x_free_pert[1] = gp_calib * 0.9997
r2 = report_point("perturbed(0.9997x)", x_free_pert)

lp("")
lp("=== Point 3: further perturbed gp ===")
x_free_pert2 = copy(x_free_calib)
x_free_pert2[1] = gp_calib * 0.999
r3 = report_point("perturbed(0.999x)", x_free_pert2)

lp("")
lp(">>> DONE.")

lp("")
lp("=== Point 4: aggressive perturbation (gp far below calibration, testing whether F1's magnitude can plausibly reach the final-gates report's 0.03-0.18 checkpoint discrepancy) ===")
x_free_pert3 = copy(x_free_calib)
x_free_pert3[1] = gp_calib * 0.97
try
    r4 = report_point("perturbed(0.97x)", x_free_pert3)
catch e
    lp("Point 4 FAILED (likely infeasible at this scale): ", sprint(showerror, e))
end

lp("")
lp("=== Point 5: even more aggressive (gp*0.9) ===")
x_free_pert4 = copy(x_free_calib)
x_free_pert4[1] = gp_calib * 0.9
try
    r5 = report_point("perturbed(0.9x)", x_free_pert4)
catch e
    lp("Point 5 FAILED (likely infeasible at this scale): ", sprint(showerror, e))
end

lp(">>> DONE2.")
