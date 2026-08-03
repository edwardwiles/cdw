# Exact reduced (Schur-complement) Hessian of the fixed-q A-middle loop (2026-07-31 task).
#
# ============================================================================
# DERIVATION AND SIGN-CONVENTION MAPPING (verified against source, not assumed)
# ============================================================================
#
# Production's ACTUAL inner problem (`cc_bundle.jl`'s `MelitzCCBundle` functor, `mul_G!`/
# `mul_Gt!`/`melitz_full_weighted_gram!`, `moment_operator.jl`):
#
#   x = (zeta, mu) in R^{1+K}, K = op.layout.num_moments (trade cells D^2 + 1 focal link).
#   u_s(a,x) = -zeta - m_s(a)'*mu = -(G_full(a)*x)_s,  G_full = [ones(W) G(a)].
#   f(x;a) = zeta + (1/M) sum_s Psi(u_s),  M = W (number of QMC draws) -- MINIMIZED over x by
#            KNITRO (the inner CC dual solve).
#   x*(a) = argmin_x f(x;a);  DeltaStar(a) = -f(x*(a);a).
#
# Mapping to this task's abstract (eta, m_s(a), rho_s, Psi, w_s^0) notation: eta === x (BOTH
# zeta and mu -- production's own normalization multiplier zeta is genuinely part of the dual,
# not an extra parameter outside it), m_s(a) === G_full[s,:](a) (an augmented (K+1)-row whose
# FIRST entry is the constant 1 -- zero a-derivative), rho_s === 0 identically (t_s=0, exactly
# the task's own "likely production case"), w_s^0 === 1/M for every draw. Writing
# phi(a,x) := -f(x;a) gives r_s = -x'*m_s(a) = u_s EXACTLY the task's own sign-normalized
# convention (r_s = rho_s - eta'*m_s(a), rho_s=0) -- production does NOT use the alternative
# "r_s = rho_s + eta'*m_s(a)" convention. V(a) = max_x phi(a,x) = -min_x f(x;a) = DeltaStar(a)
# ("ext" = max here, not min) -- confirmed by the functor's own doc comment ("minimized over
# x") and the `DeltaStar = -f(x*)` relation used throughout `exact_a_gradient.jl`.
#
# The Schur-complement formula `H_V = phi_aa - B'*H_eta_eta^{-1}*B` (task Section 2) is the
# ordinary implicit-function-theorem Hessian of a profiled stationary point and holds
# regardless of whether x*(a) is a max or min of phi in x (it only requires phi_xx(a,x*(a))
# invertible) -- verified directly here rather than assumed, by re-deriving it from
# `g(a):=min_x f(x;a)` (an ordinary MINIMIZATION profile, standard textbook Schur complement
# `g_aa = f_aa - f_ax*f_xx^{-1}*f_xa`) and using `DeltaStar=-g`, giving the equivalent,
# implementation-friendly form actually used below:
#
#   H_V (= Hessian of DeltaStar w.r.t. full log-A) = f_ax*f_xx^{-1}*f_xa - f_aa
#
# `f_xx = (1/M)*G_full'*Diag(Psi''(u))*G_full = (1/M)*melitz_full_weighted_gram!(...)` --
# EXACTLY the matrix KNITRO's own Hessian callback already builds (`cc_bundle.jl:344-357`,
# confirmed by direct comparison, not assumed).
#
# `f_a` (existing, unchanged, production code) is `melitz_exact_a_gradient_full!`
# (`exact_a_gradient.jl`) NEGATED: that function computes `dDelta_da = phi_a = -f_a` (verified
# by direct differentiation below, matching its own envelope-theorem derivation exactly).
#
# `f_xa`/`f_ax` (mixed) and `f_aa` (explicit second A-derivative, x held fixed) were BOTH
# re-derived here from first principles (direct differentiation of the actual `f(x;a)` above,
# exploiting that every cell's a-dependence is a pure exponential `X(a_od) = Xbar*exp(k*a_od)`,
# `k=sigma-1` -- Section 3B/4/5 of the governing prompt, confirmed against `firm_quantities.jl`/
# `moment_operator.jl` exactly) -- NOT copied verbatim from the governing prompt's own Section
# 7-8 boxed `Bv`/`B'y` formulas. Cross-checking those boxed formulas against this direct
# derivation found an internal SIGN INCONSISTENCY between the prompt's own Section 2 general
# box and its Section 7-8 "simplified" restatement (substituting the prompt's own `u_s=-J_s'eta`
# into its own Section 2 `B'y` box produces the OPPOSITE overall sign from its Section 8 boxed
# restatement) -- a documented Decision-D finding (prompt Section 14/"Final report"), not a
# production-vs-task convention mismatch. The formulas implemented below were independently
# re-derived from `f(x;a)` directly, cross-checked against each other via the numerical
# adjoint identity `y'*(B*v) == v'*(B'*y)` and against finite differences (test suite), and
# are what is actually used here -- see docs/melitz_exact_reduced_hessian_2026-07-31.md for the
# full symbolic derivation.
#
# Key building blocks (all bilateral/focal second-derivative structure is CELL-DIAGONAL before
# the gravity map, Section 3B/4/5 of the governing prompt -- re-derived and confirmed, not
# assumed):
#
#   R_od,s(a) := coef_od(a)*z_power[s,o]*active_od(s)   (the "active contribution" term;
#                G[s,trade(o,d)] = R_od,s - lambda_od, lambda_od a-independent)
#   d(R_od,s)/d(a_od) = k*R_od,s,  d^2(R_od,s)/d(a_od)^2 = k^2*R_od,s   (k=sigma-1)
#   Rt_od(w) := sum_s w_s*R_od,s = mul_Gt!(w)[trade(o,d)] + lambda[o,d]*sum(w)   (recovers the
#               R-weighted sum from the G-weighted one mul_Gt! actually computes)
#
# Focal-link cell (j,d) (j=target_country) has an analogous exponential structure through its
# OWN profit term (factual, all d; autarky, d=j only) -- `Ld(w)[d] := link_term_d(w)/w_j -
# [d==j]*autarky_term(w)/w_prime` bundles both, matching `melitz_exact_a_gradient_full!`'s own
# Steps 2-3 exactly (re-derived here for a GENERAL weight vector `w` instead of the fixed
# `dpsi`, so the same machinery serves the gradient, the mixed block, and the explicit second
# derivative).
#
# `qvec_s := eta'*(J_s*v)` (task Section 7's own per-draw scalar, `eta`=the ACTUAL verified
# dual `(zeta,mu)`, `v`=an A-direction) is computed via `qvec = k*(mul_R!(mu.*v) +
# mu_link*ellv)`, `ellv` the v-weighted directional analogue of `moment_operator.jl`'s own
# `ell` construction (Section 4/5) -- both O(W*D)/O(W), no dense W x M or W x D^2 matrix ever
# materialized (governing prompt Section 11's own requirement).
#
# Verified final formulas (all cross-checked against finite differences, `test/melitz/
# exact_reduced_hessian_test.jl`):
#
#   phi_a                  = dDelta_da                              (EXISTING, unchanged)
#   H_eta_eta               = -(1/M)*melitz_full_weighted_gram!(op,ddpsi)     [(K+1)x(K+1)]
#   Bv   = (1/M)*(TermB' - TermA)
#            TermB'[trade(od)] = k*v_od*Rt_od(dpsi);  TermB'[link] = k*(dot(v[j,:],
#                link_term_dpsi)/w_j - v[j,j]*autarky_term_dpsi/w_prime);  TermB'[zeta]=0
#            TermA = G_full' * (ddpsi.*qvec)     [zeta-row=sum(ddpsi.*qvec), mu-rows=mul_Gt!]
#   B'y[od] = melitz_dual_weighted_a_gradient!(w=dpsi/M,        dual=y_mu)[od]
#           - melitz_dual_weighted_a_gradient!(w=ddpsi.*(m'y)/M, dual=mu)[od]
#            (m_s'*y = mul_G!(op,-y[1],-y_mu)[s], reusing mul_G! with a negated fake point)
#   phi_aa*v[od] = k*dDelta_da[od]*v[od]
#                - melitz_dual_weighted_a_gradient!(w=ddpsi.*qvec/M, dual=mu)[od]
#
# `melitz_dual_weighted_a_gradient!(out,op,ctx,state,w,dual)` generalizes
# `melitz_exact_a_gradient_full!`'s own Steps 1-3 to an ARBITRARY per-draw weight vector `w`
# and an ARBITRARY dual-like vector `dual` (length K, trade+link, matching `mu`'s own layout)
# -- a cross-check test confirms it reproduces `dDelta_da` EXACTLY when called with
# `w=dpsi./M, dual=mu` (the bundle's own actual dual), before it is trusted for the harder
# (`w=ddpsi.*qvec`, `w=ddpsi.*(m'y)`) cases the mixed/second-derivative operators need.
#
# Reduced (Schur-complement) full-A-space Hessian-vector product:
#
#   Hv (full log-A) = phi_aa*v - B'*(H_eta_eta^{-1}*(B*v))
#
# `H_eta_eta` is NEGATIVE SEMIDEFINITE (`= -(1/M)*Hfull`, `Hfull` a PSD Gram matrix); solved via
# a cached Cholesky factorization of the PSD `P := (1/M)*Hfull` (`H_eta_eta^{-1}*b = -(P\b)`),
# regularized only if Cholesky fails, with the regularization reported (governing prompt
# Section 13).
#
# Free-coordinate chain rule: `Hx = R'*Hv*R` (`R = M_A`, the existing exact full-from-free
# log-A linear map, `melitz_A_free_linear_map`, `fixed_q_a_middle_loop.jl`) -- reused verbatim,
# never re-derived.

using LinearAlgebra: dot, cholesky, Symmetric, Cholesky, PosDefException, eigmin, eigmax

# ----------------------------------------------------------------------------------------
# Primitive 1: mul_R! -- trade-only forward "R-weighted" sum (mul_G!'s own trade loop MINUS
# the lambda/zeta correction), the "no lambda, no zeta" moment forward-apply Section 3B needs.
# ----------------------------------------------------------------------------------------

"""
    mul_R!(out, op, w_trade) -> out

`out[s] = sum_o z_power[s,o]*cum_o(bin[s,o])`, `cum_o` the ascending-cutoff-rank prefix sum of
`w_trade[o,d]*coef[o,d]` -- i.e. `out[s] = sum_od w_trade[o,d]*R_od,s` (NO `lambda`
subtraction, NO `zeta`), unlike `mul_G!`. `w_trade` is a `D x D` matrix (`[o,d]` indexing,
matching `dDelta_da`'s own convention -- NOT `op.layout.trade_index`'s moment-column order).
Zero-allocation given preallocated `out`/`op`'s own `cum` scratch.
"""
function mul_R!(out::AbstractVector{Float64}, op::MelitzMomentOperator, w_trade::AbstractMatrix{Float64})
    D = op.D
    W = op.W
    order = op.order
    bin = op.bin
    coef = op.coef
    cum = op.cum
    z_power = op.sorted_ctx.z_power_original
    @inbounds for s in 1:W
        out[s] = 0.0
    end
    @inbounds for o in 1:D
        cum[1] = 0.0
        for m in 1:D
            d = order[m, o]
            cum[m+1] = cum[m] + w_trade[o, d] * coef[o, d]
        end
        for s in 1:W
            b = bin[s, o]
            b == 0 && continue
            out[s] += z_power[s, o] * cum[b+1]
        end
    end
    return out
end

# ----------------------------------------------------------------------------------------
# Primitive 2: melitz_ell_directional! -- v-weighted directional analogue of
# `melitz_update_moment_operator!`'s own `ell` construction (Section 4/5). Returns `ellv` such
# that `(J_s v)_link = k*ellv[s]` (`k=sigma-1`), using ONLY the focal row `v[j,:]`.
# ----------------------------------------------------------------------------------------

"""
    melitz_ell_directional!(out, op, ctx, state, v_full_A) -> out

`out[s] = d(ell_s)/d(direction v)`, DIVIDED by `k=sigma-1` (i.e. `(J_s v)_link = k*out[s]`) --
mirrors `melitz_update_moment_operator!`'s own factual/autarky `ell` construction exactly,
replacing each per-cell level (`C_jd`, `f_jd`, `f_jj`) with itself times `v[j,d]`/`v[j,j]`.
Only reads `v_full_A[j,:]` (the focal origin's own row) -- every other row is structurally
irrelevant to the focal link column, matching `MelitzCompactColumns.touches_link`.
"""
function melitz_ell_directional!(out::AbstractVector{Float64}, op::MelitzMomentOperator, ctx,
                                  state::MelitzExpandedState, v_full_A::AbstractMatrix{Float64})
    D = op.D
    W = op.W
    j = ctx.target_country
    sigma = ctx.sigma
    w_j = ctx.w[j]
    f = state.f

    prefixCv = op.prefixC   # reuse D+1 scratch (safe: melitz_update_moment_operator! only
    prefixFv = op.prefixF   # touches these once per outer point, not concurrently with an HVP)
    order_j = @view op.order[:, j]
    prefixCv[1] = 0.0
    prefixFv[1] = 0.0
    @inbounds for rank in 1:D
        d = order_j[rank]
        C_jd = melitz_C(w_j, ctx.tau[j, d], state.A[j, d], sigma, ctx.expenditure[d])
        prefixCv[rank+1] = prefixCv[rank] + C_jd * v_full_A[j, d]
        prefixFv[rank+1] = prefixFv[rank] + f[j, d] * v_full_A[j, d]
    end
    bin_j = @view op.bin[:, j]
    z_orig = op.sorted_ctx.z_original
    @inbounds for s in 1:W
        z = z_orig[s, j]
        b = Int(bin_j[s]) + 1
        out[s] = (z^(sigma - 1) * prefixCv[b] / sigma - w_j * prefixFv[b]) / w_j
    end

    f_jj = state.f_jj
    vjj = v_full_A[j, j]
    if vjj != 0.0
        z_orig_j = @view op.sorted_ctx.z_original[:, j]
        z_power_j = @view op.sorted_ctx.z_power_original[:, j]
        @inbounds for s in 1:W
            if z_orig_j[s] >= 1.0
                out[s] -= f_jj * vjj * (z_power_j[s] - 1.0)
            end
        end
    end
    return out
end

# ----------------------------------------------------------------------------------------
# Primitive 3: generalized weighted cell sums (Rt/link_term_d/autarky_term for an ARBITRARY
# per-draw weight vector `w`) -- generalizes `melitz_exact_a_gradient_full!`'s own inline
# Steps 1-3 machinery so the SAME code serves phi_a (cross-checked), B'y, and phi_aa*v.
# ----------------------------------------------------------------------------------------

"""
    melitz_weighted_Rt!(Rt, gtmp, op, w) -> sumw

Fills `Rt[o,d] = sum_s w_s*R_od,s = mul_Gt!(w)[trade(o,d)] + lambda[o,d]*sum(w)` for every
cell (`gtmp` a length-`num_moments` scratch for the raw `mul_Gt!(w)` call); returns `sum(w)`.
"""
function melitz_weighted_Rt!(Rt::AbstractMatrix{Float64}, gtmp::AbstractVector{Float64},
                              op::MelitzMomentOperator, w::AbstractVector{Float64})
    D = op.D
    mul_Gt!(gtmp, op, w)
    sumw = 0.0
    @inbounds for s in 1:op.W
        sumw += w[s]
    end
    trade_index = op.layout.trade_index
    lambda = op.lambda
    @inbounds for o in 1:D, d in 1:D
        Rt[o, d] = gtmp[trade_index[o, d]] + lambda[o, d] * sumw
    end
    return sumw
end

"""
    melitz_weighted_focal_terms!(link_term_d, op, ctx, state, w, Rt) -> autarky_term

Fills `link_term_d[d] = sum_{s active at (j,d)} w_s*pi_jd+_s = (expenditure[d]/sigma)*Rt[j,d]
- w_j*f[j,d]*tail1_d(w)` for every `d` (Section 4); returns `autarky_term =
sum_{s active autarky} w_s*pi_jj^{A,+}_s = w_prime*f_jj*(Sya(w)-S1a(w))` (Section 5). `Rt` must
already be filled (`melitz_weighted_Rt!`) at the SAME `w`.
"""
function melitz_weighted_focal_terms!(link_term_d::AbstractVector{Float64}, op::MelitzMomentOperator, ctx,
                                       state::MelitzExpandedState, w::AbstractVector{Float64},
                                       Rt::AbstractMatrix{Float64})
    D = op.D
    W = op.W
    j = ctx.target_country
    binsum1 = op.binsum   # reuse D+1 scratch (mul_Gt!'s own -- safe, not concurrently active)
    tail1 = op.tail
    @inbounds for k in 1:D+1
        binsum1[k] = 0.0
    end
    bin_j = @view op.bin[:, j]
    @inbounds for s in 1:W
        binsum1[bin_j[s]+1] += w[s]
    end
    tail1[D+1] = binsum1[D+1]
    @inbounds for k in D:-1:1
        tail1[k] = tail1[k+1] + binsum1[k]
    end
    rank_j = @view op.rank[:, j]
    expenditure = ctx.expenditure
    w_j = ctx.w[j]
    f = state.f
    sigma = ctx.sigma
    @inbounds for d in 1:D
        tail1_d = tail1[rank_j[d]+1]
        link_term_d[d] = (expenditure[d] / sigma) * Rt[j, d] - w_j * f[j, d] * tail1_d
    end

    z_orig_j = @view op.sorted_ctx.z_original[:, j]
    z_power_j = @view op.sorted_ctx.z_power_original[:, j]
    Sya = 0.0
    S1a = 0.0
    @inbounds for s in 1:W
        if z_orig_j[s] >= 1.0
            Sya += w[s] * z_power_j[s]
            S1a += w[s]
        end
    end
    w_prime = ctx.w_prime
    f_jj = state.f_jj
    return w_prime * f_jj * (Sya - S1a)
end

"""
    melitz_dual_weighted_a_gradient!(out_full_A, op, ctx, state, w, dual) -> out_full_A

Generalizes `melitz_exact_a_gradient_full!`'s own Steps 1-3 (`exact_a_gradient.jl`) to an
ARBITRARY per-draw weight vector `w` (length `W`) and an ARBITRARY dual-like vector `dual`
(length `op.layout.num_moments`, SAME layout as `mu`): `out[o,d] = dual[trade(o,d)]*k*Rt_od(w)`
for every cell, plus `out[j,d] += dual[link]*k/w_j*link_term_d(w)` and `out[j,j] -=
dual[link]*k/w_prime*autarky_term(w)` (`k=sigma-1`). Reproduces `dDelta_da` (`phi_a`) EXACTLY
when called with `w=dpsi./M, dual=mu` -- the cross-check this module's own test suite verifies
before trusting it for the mixed-block/second-derivative cases (`w=ddpsi.*qvec`,
`w=ddpsi.*(m'y)`) that reuse this SAME function.
"""
function melitz_dual_weighted_a_gradient!(out_full_A::AbstractMatrix{Float64}, op::MelitzMomentOperator, ctx,
                                           state::MelitzExpandedState, w::AbstractVector{Float64},
                                           dual::AbstractVector{Float64})
    D = op.D
    sigma = ctx.sigma
    k_exp = sigma - 1.0
    j = ctx.target_country
    trade_index = op.layout.trade_index

    Rt = zeros(D, D)
    gtmp = zeros(op.layout.num_moments)
    melitz_weighted_Rt!(Rt, gtmp, op, w)

    @inbounds for o in 1:D, d in 1:D
        out_full_A[o, d] = dual[trade_index[o, d]] * k_exp * Rt[o, d]
    end

    dual_link = dual[op.layout.focal_link_index]
    if dual_link != 0.0
        link_term_d = zeros(D)
        autarky_term = melitz_weighted_focal_terms!(link_term_d, op, ctx, state, w, Rt)
        w_j = ctx.w[j]
        w_prime = ctx.w_prime
        @inbounds for d in 1:D
            out_full_A[j, d] += dual_link * k_exp / w_j * link_term_d[d]
        end
        out_full_A[j, j] -= dual_link * k_exp / w_prime * autarky_term
    end
    return out_full_A
end

# ----------------------------------------------------------------------------------------
# Per-point cached state: everything depending on (a, x=(zeta,mu)) but NOT on the HVP
# direction, built once per verified point (governing prompt Section 13).
# ----------------------------------------------------------------------------------------

"""
    MelitzExactHessianPointState

Cached at ONE verified optimal dual `x*=(zeta,mu)` (post `solve_melitz_delta!`), reused for
every subsequent `melitz_mixed_B_mul!`/`_Bt_mul!`/`melitz_explicit_phi_aa_mul!`/
`melitz_profiled_A_hvp!` call at that SAME point -- construct/factorize once, per governing
prompt Section 13.
"""
mutable struct MelitzExactHessianPointState
    op::MelitzMomentOperator
    state::MelitzExpandedState
    M::Int
    K::Int
    mu::Vector{Float64}          # length K (trade+link), = x[2:end]
    zeta::Float64
    dpsi::Vector{Float64}        # W, Psi'(u)
    ddpsi::Vector{Float64}       # W, Psi''(u)
    Hfull::Matrix{Float64}       # (K+1)x(K+1), G_full' Diag(ddpsi) G_full
    Pfact::Union{Cholesky{Float64,Matrix{Float64}},Nothing}  # Cholesky of P=(1/M)*Hfull(+reg)
    reg_used::Float64
    grad_full_A::Matrix{Float64} # D x D, phi_a = dDelta_da (existing exact gradient)
    Rt_dpsi::Matrix{Float64}     # D x D, Rt_od(dpsi)
    link_term_dpsi::Vector{Float64}  # D
    autarky_term_dpsi::Float64
end

"""
    build_melitz_exact_hessian_point_state(obj::MelitzCCBundle, x, ctx, state; hessian_backend=:structured_serial,
        reg0=1e-10, reg_growth=100.0, max_reg_tries=12) -> MelitzExactHessianPointState

Builds the per-point cache at the verified optimal dual `x=(zeta,mu)`. `obj.op` must already
reflect the theta at which `x` is optimal (same precondition as
`melitz_exact_a_gradient_full!`). Cholesky-factorizes `P=(1/M)*Hfull` (PSD in exact
arithmetic, a Gram matrix); if it fails numerically, adds a growing diagonal regularization
(`reg0*2^k`) until it succeeds, reports the regularization actually used (`reg_used`, `0.0` if
none was needed) -- governing prompt Section 13's "regularize only if necessary and report".
"""
function build_melitz_exact_hessian_point_state(obj::MelitzCCBundle, x::AbstractVector{Float64}, ctx,
                                                 state::MelitzExpandedState;
                                                 hessian_backend::Symbol=obj.hessian_backend,
                                                 reg0::Float64=1e-10, reg_growth::Float64=100.0,
                                                 max_reg_tries::Int=12)
    op = obj.op
    D = op.D
    W = op.W
    K = op.layout.num_moments
    M = obj.M

    zeta = x[1]
    mu = Vector{Float64}(x[2:end])
    u = zeros(W)
    mul_G!(u, op, zeta, mu)
    dpsi = zeros(W)
    ddpsi = zeros(W)
    melitz_cc_dPsi!(dpsi, u)
    melitz_cc_ddPsi!(ddpsi, u)

    Hfull = zeros(K + 1, K + 1)
    if hessian_backend == :structured_parallel
        melitz_full_weighted_gram_parallel!(Hfull, op, ddpsi)
    else
        melitz_full_weighted_gram!(Hfull, op, ddpsi)
    end
    # melitz_full_weighted_gram! fills the UPPER TRIANGLE ONLY -- mirror it before factorizing.
    @inbounds for i in 1:K+1, jc in i+1:K+1
        Hfull[jc, i] = Hfull[i, jc]
    end

    P = Hfull ./ M
    Pfact = nothing
    reg_used = 0.0
    reg = reg0
    for _try in 1:max_reg_tries
        try
            Pfact = cholesky(Symmetric(P .+ (reg_used == 0.0 && _try == 1 ? 0.0 : reg) .* I(K + 1)))
            break
        catch e
            e isa PosDefException || rethrow()
            reg_used = reg
            reg *= reg_growth
        end
    end
    Pfact === nothing && error("build_melitz_exact_hessian_point_state: Cholesky of the inner " *
        "dual Hessian failed even after $max_reg_tries regularization attempts (largest tried " *
        "= $(reg/reg_growth)) -- genuinely ill-conditioned/singular H_eta_eta at this point.")

    ws_grad = MelitzExactAGradientWorkspace(op)
    grad_full_A = zeros(D, D)
    melitz_exact_a_gradient_full!(grad_full_A, obj, x, state, ctx, ws_grad)

    Rt_dpsi = zeros(D, D)
    gtmp = zeros(K)
    melitz_weighted_Rt!(Rt_dpsi, gtmp, op, dpsi)
    link_term_dpsi = zeros(D)
    autarky_term_dpsi = melitz_weighted_focal_terms!(link_term_dpsi, op, ctx, state, dpsi, Rt_dpsi)

    return MelitzExactHessianPointState(op, state, M, K, mu, zeta, dpsi, ddpsi, Hfull, Pfact, reg_used,
        grad_full_A, Rt_dpsi, link_term_dpsi, autarky_term_dpsi)
end

"""
    melitz_hxx_report(pt::MelitzExactHessianPointState) -> (cond, eigmin_P, eigmax_P, reg_used)

Governing prompt Section 13's own "report condition estimate and inertia" -- `P=(1/M)*Hfull`
is PSD (a Gram matrix, `H_eta_eta=-P` is NSD); `eigmin_P/eigmax_P` from a full symmetric
eigendecomposition (cheap at `K+1 <= D^2+2 <= 402`), `cond=eigmax_P/eigmin_P`.
"""
function melitz_hxx_report(pt::MelitzExactHessianPointState)
    P = Symmetric(pt.Hfull ./ pt.M)
    ev_min = eigmin(P)
    ev_max = eigmax(P)
    return (cond=ev_max / max(ev_min, eps()), eigmin_P=ev_min, eigmax_P=ev_max, reg_used=pt.reg_used)
end

"""
    melitz_hxx_solve(pt, b) -> z

Solves `H_eta_eta*z = b` (`H_eta_eta = -(1/M)*Hfull`, `pt.Pfact` a cached Cholesky of
`(1/M)*Hfull` (+regularization)) via `z = -(Pfact \\ b)`. Never inverts `H_eta_eta` explicitly.
"""
melitz_hxx_solve(pt::MelitzExactHessianPointState, b::AbstractVector{Float64}) = -(pt.Pfact \ b)

# ----------------------------------------------------------------------------------------
# qvec: eta'*(J_s v), the per-draw scalar Bv/phi_aa*v both need (task Section 7/9).
# ----------------------------------------------------------------------------------------

"""
    melitz_qvec!(qvec, pt, ctx, v_full_A) -> qvec

`qvec[s] = eta'*(J_s*v) = k*(mul_R!(mu.*v)[s] + mu_link*ellv[s])` (`k=sigma-1`, `eta=(zeta,mu)`
the ACTUAL verified dual cached in `pt`) -- O(W*D) (one `mul_R!`) + O(W) (one
`melitz_ell_directional!`), no dense W x D^2 tensor.
"""
function melitz_qvec!(qvec::AbstractVector{Float64}, pt::MelitzExactHessianPointState, ctx,
                       v_full_A::AbstractMatrix{Float64})
    op = pt.op
    D = op.D
    k_exp = ctx.sigma - 1.0
    trade_index = op.layout.trade_index
    muv = zeros(D, D)
    @inbounds for o in 1:D, d in 1:D
        muv[o, d] = pt.mu[trade_index[o, d]] * v_full_A[o, d]
    end
    mul_R!(qvec, op, muv)
    mu_link = pt.mu[op.layout.focal_link_index]
    ellv = zeros(op.W)
    melitz_ell_directional!(ellv, op, ctx, pt.state, v_full_A)
    @inbounds for s in 1:op.W
        qvec[s] = k_exp * (qvec[s] + mu_link * ellv[s])
    end
    return qvec
end

# ----------------------------------------------------------------------------------------
# Bv, B'y, phi_aa*v
# ----------------------------------------------------------------------------------------

"""
    melitz_mixed_B_mul!(out_dual, pt, ctx, v_full_A) -> out_dual

`out_dual = B*v = phi_(eta,a)*v`, length `K+1` (`out_dual[1]`=zeta-row, `out_dual[2:end]`=
mu-rows, matching `x`'s own layout). See module header for the verified formula
(`(1/M)*(TermB' - TermA)`).
"""
function melitz_mixed_B_mul!(out_dual::AbstractVector{Float64}, pt::MelitzExactHessianPointState, ctx,
                              v_full_A::AbstractMatrix{Float64})
    op = pt.op
    D = op.D
    K = pt.K
    M = pt.M
    j = ctx.target_country
    k_exp = ctx.sigma - 1.0
    trade_index = op.layout.trade_index
    link_idx = op.layout.focal_link_index

    qvec = zeros(op.W)
    melitz_qvec!(qvec, pt, ctx, v_full_A)
    ddpsi_qvec = pt.ddpsi .* qvec
    gA = zeros(K)
    mul_Gt!(gA, op, ddpsi_qvec)
    zetaA = 0.0
    @inbounds for s in 1:op.W
        zetaA += ddpsi_qvec[s]
    end

    out_dual[1] = -zetaA / M
    @inbounds for o in 1:D, d in 1:D
        tb = k_exp * v_full_A[o, d] * pt.Rt_dpsi[o, d]
        out_dual[1+trade_index[o, d]] = (tb - gA[trade_index[o, d]]) / M
    end
    w_j = ctx.w[j]
    w_prime = ctx.w_prime
    tb_link = k_exp * (dot(view(v_full_A, j, :), pt.link_term_dpsi) / w_j -
                        v_full_A[j, j] * pt.autarky_term_dpsi / w_prime)
    out_dual[1+link_idx] = (tb_link - gA[link_idx]) / M
    return out_dual
end

"""
    melitz_mixed_Bt_mul!(out_full_A, pt, ctx, y_dual) -> out_full_A

`out_full_A = B'*y`, a `D x D` full log-A gradient (`y_dual` length `K+1`, SAME layout as
`x`). See module header for the verified formula (two `melitz_dual_weighted_a_gradient!`
calls, subtracted).
"""
function melitz_mixed_Bt_mul!(out_full_A::AbstractMatrix{Float64}, pt::MelitzExactHessianPointState, ctx,
                               y_dual::AbstractVector{Float64})
    op = pt.op
    D = op.D
    M = pt.M
    y_zeta = y_dual[1]
    y_mu = @view y_dual[2:end]

    term1 = zeros(D, D)
    melitz_dual_weighted_a_gradient!(term1, op, ctx, pt.state, pt.dpsi ./ M, y_mu)

    mTy = zeros(op.W)
    mul_G!(mTy, op, -y_zeta, .-y_mu)
    w2 = (pt.ddpsi .* mTy) ./ M
    term2 = zeros(D, D)
    melitz_dual_weighted_a_gradient!(term2, op, ctx, pt.state, w2, pt.mu)

    @. out_full_A = term1 - term2
    return out_full_A
end

"""
    melitz_explicit_phi_aa_mul!(out_full_A, pt, ctx, v_full_A) -> out_full_A

`out_full_A = phi_aa*v` (x HELD FIXED at the cached `pt.mu`/`pt.zeta` -- NOT the profiled
Hessian), a `D x D` full log-A vector. See module header (`k*dDelta_da.*v` MINUS a
`melitz_dual_weighted_a_gradient!(w=ddpsi.*qvec/M, dual=mu)` term).
"""
function melitz_explicit_phi_aa_mul!(out_full_A::AbstractMatrix{Float64}, pt::MelitzExactHessianPointState, ctx,
                                      v_full_A::AbstractMatrix{Float64})
    op = pt.op
    D = op.D
    M = pt.M
    k_exp = ctx.sigma - 1.0

    qvec = zeros(op.W)
    melitz_qvec!(qvec, pt, ctx, v_full_A)
    w2 = (pt.ddpsi .* qvec) ./ M
    term2 = zeros(D, D)
    melitz_dual_weighted_a_gradient!(term2, op, ctx, pt.state, w2, pt.mu)

    @inbounds for o in 1:D, d in 1:D
        out_full_A[o, d] = k_exp * pt.grad_full_A[o, d] * v_full_A[o, d] - term2[o, d]
    end
    return out_full_A
end

# ----------------------------------------------------------------------------------------
# Complete profiled reduced Hessian-vector product (full A-space and free-coordinate).
# ----------------------------------------------------------------------------------------

"""
    melitz_profiled_A_hvp_full!(out_full_A, pt, ctx, v_full_A) -> out_full_A

`out_full_A = H_V*v = phi_aa*v - B'*(H_eta_eta^{-1}*(B*v))`, the exact profiled DeltaStar
Hessian-vector product in FULL (`D x D`) log-A coordinates (module header). One `melitz_mixed_B_mul!`
call, one cached-Cholesky solve, one `melitz_mixed_Bt_mul!` call, one `melitz_explicit_phi_aa_mul!`
call -- no dense `W x M`/`W x D^2` matrix, `H_eta_eta` never inverted explicitly.
"""
function melitz_profiled_A_hvp_full!(out_full_A::AbstractMatrix{Float64}, pt::MelitzExactHessianPointState, ctx,
                                      v_full_A::AbstractMatrix{Float64})
    K = pt.K
    b = zeros(K + 1)
    melitz_mixed_B_mul!(b, pt, ctx, v_full_A)
    z = melitz_hxx_solve(pt, b)
    Btz = zeros(size(v_full_A))
    melitz_mixed_Bt_mul!(Btz, pt, ctx, z)
    phiaa = zeros(size(v_full_A))
    melitz_explicit_phi_aa_mul!(phiaa, pt, ctx, v_full_A)
    @. out_full_A = phiaa - Btz
    return out_full_A
end

"""
    melitz_profiled_A_hvp!(out_free_A, pt, ctx, v_free_A) -> out_free_A

Free-coordinate Hessian-vector product: `Hx*v_free = R'*(H_V*(R*v_free))`, `R = M_A`
(`melitz_A_free_linear_map(ctx)`, `fixed_q_a_middle_loop.jl`, reused verbatim -- the exact
full-from-free log-A linear map, an algebraic identity of `ctx.A_pivot`). `M_A` is
materialized densely (`D^2 x (D^2-1)`, trivial at `D<=20`, matching
`melitz_exact_a_gradient_free`'s own convention -- the pivot coordinate is never
differentiated by hand anywhere in this file, per governing prompt Section 2).
"""
function melitz_profiled_A_hvp!(out_free_A::AbstractVector{Float64}, pt::MelitzExactHessianPointState, ctx,
                                 v_free_A::AbstractVector{Float64}, M_A::AbstractMatrix{Float64}=melitz_A_free_linear_map(ctx))
    D = ctx.D
    v_full_vec = M_A * v_free_A
    v_full = reshape(v_full_vec, D, D)
    h_full = zeros(D, D)
    melitz_profiled_A_hvp_full!(h_full, pt, ctx, v_full)
    out_free_A .= M_A' * vec(h_full)
    return out_free_A
end
