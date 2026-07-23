# ============================================================================
# Profiling helper: minimize Delta_dual over the eta coordinates ONLY, at a
# fixed economic outer point (x_free0). Used for (1) the mean-only-arm
# implementation-equivalence check (task brief Section 3: profiled
# origin-moments-only ~ unrestricted) and (2) the D=20 fixed-point gates
# (task brief Section 11.2: "minimize/profile over the origin-specific eta
# coordinates at fixed economic outer point").
#
# Uses Optim.jl's LBFGS with the ANALYTIC eta-gradient (d_delta_dual_d_eta_origin_vec,
# cheap: one dot product per moment block, no extra inner solve) via
# `only_fg!` -- each Optim iteration needs exactly ONE inner CC dual solve
# (archOZ_verified_state), not `n_eta+1` (which an FD-Jacobian root-finder,
# e.g. NLsolve with autodiff=:finite, would need). At D=20 with n_eta up to
# 40 (K=2), this difference is the difference between a few inner solves and
# a few dozen PER ITERATION -- decisive at production scale (task brief
# Section 6: "measure rather than assume" the cost of the enlarged outer
# loop, and don't let the profiling harness itself be the bottleneck).
#
# DIAGNOSTIC-ONLY (release note, 2026-07-23): Optim.jl is deliberately NOT a
# Project.toml/Manifest.toml dependency of this repo -- the production joint
# solver (run_originzc_upper_checkpointed, cm_originzc_checkpoint.jl) never
# needs this helper and does not include this file. Only
# d20_originzc_fixedpoint_gates.jl (a diagnostic gate script, not part of any
# production entry point) includes it. To run that script or anything else
# that loads this file, `] add Optim` in this project's environment locally
# first; do not add Optim to the committed Project.toml for that purpose.
# ============================================================================
using Optim

"""
    profile_eta_originzc(x_free0, η0, pcx; iterations=50, g_tol=1e-8, show_trace=false) -> (result, last)

Minimizes `Delta_dual(x_free0, exp.(η))` over `η` via `Optim.LBFGS`, using
the analytic gradient at every trial point (no finite differences). Returns
the Optim result object and `last = (base, verify, νfull)` at the LAST
SUCCESSFUL inner solve (so callers can immediately compute residuals/
derivatives there without a redundant extra inner solve).

A line-search trial `η` can push some `nu = exp(eta)` outside the region
where the inner CC dual problem is feasible (e.g. a large first LBFGS step
at an economic point far from where `eta0` was calibrated -- observed live
at the D=20 fixed-point gate's economic Point B, a large-magnitude
production incumbent very different in scale from the calibration point).
`archOZ_verified_state` signals this via `CMExpectedSolveFailure` (the same
expected-failure type the production KNITRO callback already handles via
`reject_point` -- cm_checkpoint.jl/cm_originzc_checkpoint.jl's `cb_F!`).
This function catches it the same way: returns `Inf` for `F` (a standard
"reject this point" signal most line searches, including Optim's default
HagerZhang, back off from) and an all-zero `G` (never used to accept a
step, only to satisfy `only_fg!`'s contract), WITHOUT updating
`last_state[]` -- the returned `last` is always a genuinely converged inner
solve, never a rejected probe.
"""
function profile_eta_originzc(x_free0::AbstractVector, η0::AbstractVector{Float64}, pcx;
                               iterations::Int = 50, g_tol::Float64 = 1e-8, show_trace::Bool = false)
    last_state = Ref{Any}(nothing)
    function fg!(F, G, η)
        νfull = exp.(η)
        local base, verify
        try
            base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            G !== nothing && fill!(G, 0.0)
            F !== nothing && return Inf
            return nothing
        end
        last_state[] = (base = base, verify = verify, νfull = νfull)
        if G !== nothing
            G .= d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull; mean_m = verify.m_mean)
        end
        F !== nothing && return verify.Delta_dual
        return nothing
    end
    res = Optim.optimize(Optim.only_fg!(fg!), η0, Optim.LBFGS(),
                          Optim.Options(iterations = iterations, g_tol = g_tol, show_trace = show_trace))
    last_state[] === nothing && error("profile_eta_originzc: every probed eta was infeasible -- no successful inner solve to report (eta0 itself may be a bad start for this economic point)")
    return res, last_state[]
end
