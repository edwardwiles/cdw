# ============================================================================
# Corrected wiring: `gradient_method = :fixed_dual_fd_full` (and, later,
# :boundary_full) into the ACTUAL production sequential outer loop, using the
# FULL (D+2)-moment fixed-dual criterion (full_fixed_dual_criterion.jl), not
# the reduced (D+1)-moment one the earlier `gradient_method_wiring.jl` used.
#
# DESIGN CHANGE vs the earlier (flawed) gradient_method_wiring.jl: that file
# computed a CORRECTION on the D+1-moment reduced problem (dropping
# lambda_R*G_R, the gravity-linearized moment's own dual multiplier and
# moment, entirely) and added it to the full AD gradient. Per the task's
# explicit instruction, that is not generally valid: lambda_R*G_R enters the
# conjugate argument arg0 = -zeta - lambda_b'G_b - lambda_R*G_R as a common
# offset inside the NONLINEAR Psi(arg0), so it generally shifts the
# winner-boundary jump even though G_R itself has no winner dependence.
#
# This file instead REPLACES the Acol block of g_free directly with the
# full-(D+2) fixed-dual FD gradient (computed via `fixed_dual_fd_gradient`
# from fixed_dual_fd.jl, unmodified/generic, called with d=D+2 and the frozen
# full-(D+2) moments function from full_fixed_dual_criterion.jl) -- no
# baseline subtraction, no reduced-model intermediate step. gamma'_focal
# (x_free[1]) keeps using the existing full-(D+2) AD gradient, which is
# already exact for that coordinate (verified: neither the winner rule nor
# the price-index moment depends on theta[3]=gamma'_focal at all -- see
# derivative_methods_report.md and focal_moments_directgp.jl).
#
# CORRECTED UNDERSTANDING (an earlier version of this file got this backwards
# -- recorded here so a future session doesn't re-derive it): `find_smallest`
# on a `PsiObjectiveBundleDelta` should ALWAYS be `true` (the struct's own
# default), REGARDLESS of which outer bound (lower/upper) is being solved.
# `PsiObjectiveBundleDelta` computes the genuine CC divergence delta*(theta),
# a quantity with NO dependence on which direction the OUTER problem extremizes
# gamma' in -- production's own `recover_lfd` (run_profiled_production.jl)
# NEVER varies it, always using the struct default. `find_smallest` only
# genuinely varies for the OUTER bundle types (PsiObjectiveBundleImplicit/
# MethodB), whose H_save/reported-objective is K[1]=gamma'_focal, a DIFFERENT
# quantity that legitimately depends on search direction. Threading
# `obj.find_smallest` (the OUTER MethodB bundle's value) into a
# `PsiObjectiveBundleDelta` construction, as an earlier version of this file
# did, silently NEGATES delta* for the lower-bound case (find_smallest=false)
# -- caught by comparing a `build_fixed_dual_bundle` call at the Frechet
# benchmark under find_smallest=true (delta*=+2.78e-4, matching every Part
# 1-7 validation) against find_smallest=false at the IDENTICAL theta
# (delta*=-2.78e-4, same magnitude, wrong sign) via `dual_criterion_fixed_x`
# cross-checked against `inner_loop` directly. This fed the outer KNITRO
# search a SIGN-FLIPPED Acol-block gradient for the lower bound specifically,
# which is consistent with that search's erratic behavior (objective value
# hitting a bogus 1e10 mid-solve, needing far more iterations than the upper
# bound) before this fix. `find_smallest=true` is now hardcoded everywhere a
# `PsiObjectiveBundleDelta` is built in this file (and in
# fixed_A_incumbent.jl), independent of the real outer bound direction.
# ============================================================================

"""
    make_seq_div_grad_fn_full!(obj, fpmap, γobj, U, D, gcol_st, lastθ_st, lastRcol_st, dRdθ_st, lastok_st, method;
                                 Acol_offset=3, h_fd=0.1, trade_moments_fn! =EK_moments_focal_norm_directgp!)

Drop-in replacement for `run_profiled_production.jl::make_seq_div_grad_fn!`,
using the FULL-(D+2) fixed-dual derivative for the A[.,focal] block.
`method` in (:pointwise_ad, :fixed_dual_fd_full). `gcol_st`/`lastθ_st`/
`lastRcol_st`/`dRdθ_st`/`lastok_st` are the Refs returned (additively) by
`make_stateful_moments` -- they hold the CURRENT sequential iterate's frozen
gravity linearization, i.e. exactly the state `obj.moments!` itself already
uses internally for ForwardDiff.Dual theta. Same closure signature
`(g_free, x_free, θ_full, inner_x) -> g_free` as the original.
"""
function make_seq_div_grad_fn_full!(obj, fpmap, γobj, U::Matrix{Float64}, D::Int,
        gcol_st, lastθ_st, lastRcol_st, dRdθ_st, lastok_st, method::Symbol;
        Acol_offset::Int=3, h_fd::Float64=0.1,
        trade_moments_fn!::Function=EK_moments_focal_norm_directgp!)
    d = obj.d; oci = obj.outer_constr_index
    @assert d == D + 2 "make_seq_div_grad_fn_full! requires the full (D+2)-moment problem (got d=$d)"
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        # ---- 1. full-(D+2) AD, exact and unmodified (gives g_free[1]=gamma'_focal exactly;
        #         gives g_free[2:D+1] too, but that block is OVERWRITTEN below for method != :pointwise_ad) ----
        obj(inner_x, Float64[], Float64[]; constr=zeros(1))   # trigger dPsi!, populate obj.arg1 at (θ_full, inner_x)
        λfull = collect(@view inner_x[2:end])                  # length D+2
        Usub = obj.U[1:obj.N, :]
        θ0 = reconstruct_full(x_free, fpmap)
        f_ad = x -> CS._methodB_envelope_scalar(reconstruct_full(x, fpmap), obj.moments!, obj.γ, Usub, λfull, obj.arg1, d, oci)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f_ad, x_free)
        end
        ForwardDiff.gradient!(g_free, f_ad, x_free, cfg_cache[])

        method == :pointwise_ad && return g_free

        if !lastok_st[]
            # No feasible frozen linearization available yet at this theta (mirrors production's own
            # dRdθ[]=zeros(...) fallback in make_stateful_moments -- AD-only, not a crash).
            return g_free
        end

        # ---- 2. Replace the Acol block with the full-(D+2) fixed-dual FD gradient ----
        Acol0 = θ0[Acol_offset+1:Acol_offset+D]

        frozen_moments = make_frozen_gravity_moments(trade_moments_fn!, D, lastθ_st[], lastRcol_st[], gcol_st[], dRdθ_st[])
        x_fixed_full = collect(inner_x)   # length oci = D+3: (zeta, lambda_1..lambda_{D+2}) -- the REAL inner solution, unreduced

        method_grad_log = if method == :fixed_dual_fd_full
            fixed_dual_fd_gradient(θ0, γobj, Usub, frozen_moments, d, x_fixed_full, h_fd;
                l=length(θ0), Acol_offset=Acol_offset, find_smallest=true)
        elseif method == :boundary_full
            g, _ = boundary_envelope_gradient_full(θ0, γobj, Usub, D, (1 / θ_full[1]) / (θ_full[2] - 1),
                frozen_moments, x_fixed_full, true; Acol_offset=Acol_offset)
            g
        else
            error("unknown gradient_method $method")
        end

        # Convert from (signed via find_smallest=true, log-Acol, Q=delta*-convention) units directly
        # into g_free's native (-1e10-scaled, level-Acol, constr[1]-convention) units -- NO
        # baseline/AD subtraction: this IS the gradient for that block now, not a correction added to
        # something else. Derivation: constr[1]=-f_raw*1e10, Q(find_smallest=true)=-f_raw =>
        # constr[1]=Q*1e10 => d(constr[1])/dAcol = 1e10*dQ/dAcol = 1e10*method_grad_log/Acol0 -- a
        # CONSTANT +1e10 factor, independent of the outer problem's own find_smallest (see module
        # docstring for why threading obj.find_smallest through here was the earlier bug).
        @views g_free[2:D+1] .= 1e10 .* method_grad_log ./ Acol0
        return g_free
    end
end
