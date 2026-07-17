# ============================================================================
# Task §12: derivative methods, building on three_way_derivatives.jl (which
# already implements B/frozen-adjoint, C/fixed-dual, D/optimized).
# ============================================================================
using ForwardDiff

"""
    method_A_pathwise_ad(x_free, ctx, base::BaseDualState) -> Vector

Method A (task §12.A): naive ForwardDiff.gradient straight through
frozen_adjoint_Q's own moments!->hFunction!->MinInd! call chain. This is
EXACTLY what production's existing Method-B envelope-gradient path
(`full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl`,
`run_fullA_D4_production.jl::make_div_grad_fn!`) already computes -- a
baseline/bug-detector, expected (not merely suspected) to silently drop the
winner-boundary/Dirac term, since `MinInd!`'s hard Bool branch has zero
a.e. derivative (see docs/fullA_d4_code_audit.md sec 6).
"""
function method_A_pathwise_ad(x_free::AbstractVector, ctx, base::BaseDualState)
    f = x -> frozen_adjoint_Q(x, ctx, base)
    return ForwardDiff.gradient(f, x_free)
end

"""
    method_G_hybrid_refresh(x_free_history, ctx; refresh_every=5, gap_tol=0.05)

Task §12.G: use the cheap fixed_dual_L gradient (finite-differenced) between
periodic expensive optimized_Delta-gradient refreshes. This is a POLICY
function, not a one-shot evaluation: given a callback history (list of
accepted outer iterates), decides at each iterate whether to pay for a fresh
optimized-value gradient. Implemented here as a decision rule usable by a
future KNITRO driver (task sec 18), not wired into a live solve yet.

Triggers implemented (task's own suggested list, the ones cheaply computable
from information already in this diagnostic toolkit):
  - fixed iteration count since last refresh (`refresh_every`)
  - disagreement between a cheap Q_adj/L_fix finite-difference secant and
    the frozen base's own recorded Delta value beyond `gap_tol` (relative)
"""
function should_refresh(iters_since_refresh::Int, cheap_slope::Float64, last_optimized_slope::Float64;
        refresh_every::Int = 5, gap_tol::Float64 = 0.05)
    iters_since_refresh >= refresh_every && return (true, :iteration_count)
    denom = max(abs(last_optimized_slope), 1e-12)
    rel_gap = abs(cheap_slope - last_optimized_slope) / denom
    rel_gap > gap_tol && return (true, :fixed_dual_disagreement)
    return (false, :none)
end
