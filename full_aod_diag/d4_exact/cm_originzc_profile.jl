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
# ============================================================================
using Optim

"""
    profile_eta_originzc(x_free0, η0, pcx; iterations=50, g_tol=1e-8, show_trace=false) -> (result, last)

Minimizes `Delta_dual(x_free0, exp.(η))` over `η` via `Optim.LBFGS`, using
the analytic gradient at every trial point (no finite differences). Returns
the Optim result object and `last = (base, verify, νfull)` at the terminal
point (so callers can immediately compute residuals/derivatives there
without a redundant extra inner solve).
"""
function profile_eta_originzc(x_free0::AbstractVector, η0::AbstractVector{Float64}, pcx;
                               iterations::Int = 50, g_tol::Float64 = 1e-8, show_trace::Bool = false)
    last_state = Ref{Any}(nothing)
    function fg!(F, G, η)
        νfull = exp.(η)
        base, verify = archOZ_verified_state(x_free0, νfull, pcx.ctx_cm)
        last_state[] = (base = base, verify = verify, νfull = νfull)
        if G !== nothing
            G .= d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull; mean_m = verify.m_mean)
        end
        F !== nothing && return verify.Delta_dual
        return nothing
    end
    res = Optim.optimize(Optim.only_fg!(fg!), η0, Optim.LBFGS(),
                          Optim.Options(iterations = iterations, g_tol = g_tol, show_trace = show_trace))
    return res, last_state[]
end
