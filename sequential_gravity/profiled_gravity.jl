# Sequentially-linearized profiled full-gravity — Phase-1 core (KNITRO-free).
#
# Implements the standalone, mathematically-risky pieces of the ChatGPT spec, translated into
# this codebase's objects (see SEQUENTIAL_GRAVITY_DESIGN.md for the derivation):
#   • destination trade-share inversion  I_d(F)                        (spec §6)
#   • exact origin+destination-FE gravity residual  R                  (spec §7)
#   • draw-level share objects  r, ξ, M_d                              (spec §8)
#   • share Jacobian  H_d  (closed form + softmax-consistent + AD)     (spec §§9-10)
#   • adjoint solve  H_d a_d = c_d  and influence function  ψ_R, ψ̄     (spec §§11-13)
#
# Model↔spec map (all indices: draw s = row, origin o = column):
#   log_x[s,o] = μ·log(Uσ[s,o]) = μ(1-σ)·log(U[s,o])         (destination-independent)
#   u[o,d]     = (σ-1)(log A_od − log w_o − log τ_od)        (A_od = 1/AodPow structural)
#   winner[s]  = argmax_o ( u[o,d] + log_x[s,o] )            (= argmin level price)
#   share_d[o] = Σ_s p[s]·W[s,o]·exp(V[s]) / Σ_s p[s]·exp(V[s])
# where V[s]=max_o(·) and W[s,o]=1{winner} for HARD max (ρ=0), or the softmax over origins at
# temperature ρ>0 with V[s]=ρ·logsumexp_o(·/ρ).  ρ>0 makes share(u,p) C^∞ (spec §10): the
# inversion converges tightly through the dense finite-sample winner-switch kinks, R varies
# smoothly in the distribution, and the influence function is the EXACT derivative of the
# smoothed residual.  ρ=0 is the hard-max model used for the exact residual / convergence check.
#
# Gauge: reference origin `ref` (default 1) pinned to u[ref,d]=0 (matches the code's A[1,d]=1).
# Dependencies: LinearAlgebra, ForwardDiff only.  No KNITRO, no pipeline globals.

module ProfiledGravity

using LinearAlgebra
using ForwardDiff

export build_log_x, build_log_x_fromU, invert_destination, DestInversion,
       dest_stats, dest_share, potential_free, share_jacobian_closed, share_jacobian_smoothed,
       share_jacobian_ad, free_hessian, two_way_demean, gravity_residual, GravityResidual,
       draw_level_objects, influence_function, InfluenceResult, logsumexp, free_idx

# ----------------------------------------------------------------------------------------------
# draw objects
# ----------------------------------------------------------------------------------------------

"log_x[s,o] = μ·log(Uσ[s,o]).  Uσ is the pipeline's `Uσ = U.^(1-σ)` matrix."
build_log_x(Uσ::AbstractMatrix, μ::Real) = μ .* log.(Uσ)

"log_x[s,o] = μ(1-σ)·log(U[s,o]) built straight from the raw Exp(1) draws U."
build_log_x_fromU(U::AbstractMatrix, μ::Real, σ::Real) = (μ * (1 - σ)) .* log.(U)

@inline function logsumexp(a::AbstractVector)
    m = maximum(a)
    isfinite(m) || return m
    s = zero(eltype(a))
    @inbounds for x in a
        s += exp(x - m)
    end
    return m + log(s)
end

free_idx(ref::Int, D::Int) = [o for o in 1:D if o != ref]

# ----------------------------------------------------------------------------------------------
# share potential, per-draw assignment weights, and model shares
# ----------------------------------------------------------------------------------------------

"""
    dest_stats(log_x, logp, u; ρ=0.0)

For one destination with full competitiveness vector `u` (length D, incl. ref) and log-weights
`logp` (length S): per-draw log-value `V`, log-normalizer `logdenom = φ`, normalized draw weights
`rweight` (Σ=1), origin `share` (Σ=1, = ∂φ/∂u), and per-draw origin-assignment weights `W` (S×D:
0/1 winner indicator for ρ=0; softmax at temperature ρ for ρ>0).
"""
function dest_stats(log_x::AbstractMatrix, logp::AbstractVector, u::AbstractVector; ρ::Real = 0.0)
    S, D = size(log_x)
    T = promote_type(eltype(log_x), eltype(u), typeof(float(ρ)))
    V = Vector{T}(undef, S)
    W = Matrix{T}(undef, S, D)
    @inbounds for s in 1:S
        m = T(-Inf)
        for o in 1:D
            v = u[o] + log_x[s, o]
            if v > m; m = v; end
        end
        if ρ <= 0
            for o in 1:D
                W[s, o] = (u[o] + log_x[s, o] == m) ? one(T) : zero(T)
            end
            V[s] = m
        else
            se = zero(T)
            for o in 1:D
                e = exp((u[o] + log_x[s, o] - m) / ρ)
                W[s, o] = e; se += e
            end
            for o in 1:D; W[s, o] /= se; end
            V[s] = m + ρ * log(se)
        end
    end
    a = logp .+ V
    logdenom = logsumexp(a)
    rweight = exp.(a .- logdenom)
    share = zeros(T, D)
    @inbounds for s in 1:S, o in 1:D
        share[o] += rweight[s] * W[s, o]
    end
    return (V = V, logdenom = logdenom, rweight = rweight, share = share, W = W)
end

"""
    dest_share(log_x, logp, u; ρ=0.0)

Non-allocating model shares + logdenom (φ) for the inversion line search — same values as
`dest_stats` but without materializing the S×D assignment-weight matrix (two O(S·D) passes).
"""
function dest_share(log_x::AbstractMatrix, logp::AbstractVector, u::AbstractVector; ρ::Real = 0.0)
    S, D = size(log_x)
    T = promote_type(eltype(log_x), eltype(u), typeof(float(ρ)))
    a = Vector{T}(undef, S)
    @inbounds for s in 1:S
        m = T(-Inf)
        for o in 1:D
            v = u[o] + log_x[s, o]; v > m && (m = v)
        end
        if ρ <= 0
            a[s] = logp[s] + m
        else
            se = zero(T)
            for o in 1:D; se += exp((u[o] + log_x[s, o] - m) / ρ); end
            a[s] = logp[s] + m + ρ * log(se)
        end
    end
    logdenom = logsumexp(a)
    share = zeros(T, D)
    @inbounds for s in 1:S
        rw = exp(a[s] - logdenom)
        m = T(-Inf)
        for o in 1:D
            v = u[o] + log_x[s, o]; v > m && (m = v)
        end
        if ρ <= 0
            for o in 1:D
                if u[o] + log_x[s, o] == m; share[o] += rw; break; end
            end
        else
            se = zero(T)
            for o in 1:D; se += exp((u[o] + log_x[s, o] - m) / ρ); end
            for o in 1:D; share[o] += rw * exp((u[o] + log_x[s, o] - m) / ρ) / se; end
        end
    end
    return share, logdenom
end

@inline function _insert_ref(u_free::AbstractVector{T}, ref::Int, D::Int) where {T}
    u = Vector{T}(undef, D)
    j = 1
    @inbounds for o in 1:D
        if o == ref
            u[o] = zero(T)
        else
            u[o] = u_free[j]; j += 1
        end
    end
    return u
end

"Convex potential φ_d as a function of the D−1 free coordinates (u[ref]=0). Type-generic for AD."
function potential_free(u_free::AbstractVector, log_x::AbstractMatrix, logp::AbstractVector;
                        ref::Int = 1, ρ::Real = 0.0)
    D = size(log_x, 2)
    u = _insert_ref(u_free, ref, D)
    S = size(log_x, 1)
    T = promote_type(eltype(log_x), eltype(u_free), typeof(float(ρ)))
    a = Vector{T}(undef, S)
    @inbounds for s in 1:S
        m = T(-Inf)
        for o in 1:D
            v = u[o] + log_x[s, o]
            if v > m; m = v; end
        end
        if ρ <= 0
            a[s] = logp[s] + m
        else
            se = zero(T)
            for o in 1:D
                se += exp((u[o] + log_x[s, o] - m) / ρ)
            end
            a[s] = logp[s] + m + ρ * log(se)
        end
    end
    return logsumexp(a)
end

# ----------------------------------------------------------------------------------------------
# share Jacobian H_d = ∂share/∂u (free coords)
# ----------------------------------------------------------------------------------------------

"Hard-max (ρ=0) Hessian of φ on free coords: (diag(share) − share·share')[free,free]."
function share_jacobian_closed(share::AbstractVector; ref::Int = 1)
    D = length(share)
    fi = free_idx(ref, D)
    H = Matrix{eltype(share)}(undef, D - 1, D - 1)
    @inbounds for (a, o) in enumerate(fi), (b, p) in enumerate(fi)
        H[a, b] = (o == p ? share[o] : zero(eltype(share))) - share[o] * share[p]
    end
    return H
end

"""
    share_jacobian_smoothed(rweight, W, share, ρ; ref=1)

Exact free-coord Hessian of the softmax-smoothed potential (spec §10):
  H = A − ss' + (1/ρ)(diag(share) − A),   A_oo' = Σ_s rweight_s·W[s,o]·W[s,o'].
As ρ→0 (W→indicator, A→diag(share)) this reduces to `share_jacobian_closed`.
"""
function share_jacobian_smoothed(rweight::AbstractVector, W::AbstractMatrix, share::AbstractVector,
                                 ρ::Real; ref::Int = 1)
    S, D = size(W)
    fi = free_idx(ref, D)
    # A = W' diag(rweight) W restricted to free origins
    nf = length(fi)
    A = zeros(eltype(share), nf, nf)
    @inbounds for s in 1:S
        rs = rweight[s]
        for (a, o) in enumerate(fi)
            wso = rs * W[s, o]
            for (b, q) in enumerate(fi)
                A[a, b] += wso * W[s, q]
            end
        end
    end
    H = Matrix{eltype(share)}(undef, nf, nf)
    @inbounds for (a, o) in enumerate(fi), (b, q) in enumerate(fi)
        diag = (o == q) ? share[o] : zero(eltype(share))
        H[a, b] = A[a, b] - share[o] * share[q] + (diag - A[a, b]) / ρ
    end
    return H
end

"Hard-max/soft AD Hessian of the free potential (spec §9). Cross-checks the closed forms."
function share_jacobian_ad(log_x::AbstractMatrix, logp::AbstractVector, u_full::AbstractVector;
                           ref::Int = 1, ρ::Real = 0.0)
    D = size(log_x, 2)
    u_free = collect(u_full[free_idx(ref, D)])
    f = uf -> potential_free(uf, log_x, logp; ref = ref, ρ = ρ)
    return ForwardDiff.hessian(f, u_free)
end

"Free-coord Hessian from a dest_stats result: smoothed closed form for ρ>0, hard closed form for ρ=0."
function free_hessian(st, ρ::Real; ref::Int = 1)
    ρ > 0 ? share_jacobian_smoothed(st.rweight, st.W, st.share, ρ; ref = ref) :
            share_jacobian_closed(st.share; ref = ref)
end

# ----------------------------------------------------------------------------------------------
# destination inversion  I_d(F)  (spec §6)
# ----------------------------------------------------------------------------------------------

struct DestInversion{T}
    u_full::Vector{T}
    model_shares::Vector{T}
    max_abs_share_error::T
    objective_value::T
    gradient_norm::T
    iterations::Int
    converged::Bool
    ref::Int
    ρ::T
end

"""
    invert_destination(log_x, p, λ̂; ref=1, ρ=0.0, tol=1e-10, maxit=100, u_init=nothing)

Recover `u[·,d]` (gauge u[ref]=0) so model shares match the observed column `λ̂` (Σ=1, all >0)
under LFD weights `p` (Σ=1). Damped Newton on the convex potential φ−λ̂·u (gradient share−λ̂,
Hessian `free_hessian`), with an EXACT line search along the Newton direction (the objective is
convex and C¹ in u; g(α)=obj(u+α·step) is convex with g'(α)=⟨step, share−λ̂⟩, so we bracket the
sign change of g' and bisect — this takes the maximal safe step through the dense winner-switch
kinks instead of chattering).  ρ>0 smooths the model so this converges to machine-ish tolerance.
"""
function invert_destination(log_x::AbstractMatrix, p::AbstractVector, λ̂::AbstractVector;
                            ref::Int = 1, ρ::Real = 0.0, tol::Real = 1e-10, maxit::Int = 100,
                            ls_iters::Int = 80,
                            u_init::Union{Nothing,AbstractVector} = nothing, verbose::Bool = false)
    S, D = size(log_x)
    @assert length(p) == S && length(λ̂) == D
    logp = log.(p)
    fi = free_idx(ref, D)
    T = float(promote_type(eltype(log_x), eltype(p)))
    u = u_init === nothing ? zeros(T, D) : T.(collect(u_init))
    u[ref] = 0.0
    st = dest_stats(log_x, logp, u; ρ = ρ)
    local iter = 0
    converged = false
    for it in 1:maxit
        iter = it
        share = st.share
        gerr = maximum(abs.(share .- λ̂))
        verbose && println("  inv it=$it  share_err=$(gerr)")
        if gerr < tol
            converged = true; break
        end
        grad_free = [share[o] - λ̂[o] for o in fi]
        H = free_hessian(st, ρ; ref = ref)
        step = try
            -(H \ grad_free)
        catch
            -((H + 1e-12 * I) \ grad_free)
        end
        unew = copy(u)
        if ρ > 0
            # SMOOTH regime (ρ>0): φ−λ̂·u is C² convex ⇒ Newton + cheap Armijo backtrack converges
            # quadratically (~1–3 evals/step). The exact line search below is only needed to fight
            # the hard-max winner-switch kinks; unnecessary (and ~ls_iters× slower) when ρ>0.
            # Accept the full Newton step on monotone decrease with a tiny slack (NOT Armijo
            # sufficient-decrease: near the flat minimum the sufficient-decrease target sinks below
            # float noise in the objective and backtracking would stall short of machine precision).
            # Near the solution α=1 always passes ⇒ Newton drives the gradient to ~1e-13; far away a
            # genuine overshoot still increases the objective and is backtracked.
            obj_u = st.logdenom - dot(λ̂, u)
            slack = 1e-12 * (1 + abs(obj_u))
            αstar = 1.0
            for _ in 1:40
                for (k, o) in enumerate(fi); unew[o] = u[o] + αstar * step[k]; end
                unew[ref] = 0.0
                _, ld = dest_share(log_x, logp, unew; ρ = ρ)
                (ld - dot(λ̂, unew) <= obj_u + slack) && break
                αstar *= 0.5
            end
        else
            # HARD-MAX regime (ρ=0): exact line search — bracket the sign change of the directional
            # derivative g'(α)=⟨step, share−λ̂⟩ and bisect (share is only C⁰; kinks are dense).
            dprime = function (α)
                for (k, o) in enumerate(fi); unew[o] = u[o] + α * step[k]; end
                unew[ref] = 0.0
                s, _ = dest_share(log_x, logp, unew; ρ = ρ)
                acc = zero(T)
                for (k, o) in enumerate(fi); acc += step[k] * (s[o] - λ̂[o]); end
                return acc
            end
            αlo = 0.0; αhi = 1.0; dhi = dprime(αhi); nexp = 0
            while dhi < 0 && αhi < 1e8 && nexp < 60
                αlo = αhi; αhi *= 2.0; dhi = dprime(αhi); nexp += 1
            end
            αstar = αhi
            if dhi > 0
                for _ in 1:ls_iters
                    αm = 0.5 * (αlo + αhi)
                    dprime(αm) > 0 ? (αhi = αm) : (αlo = αm)
                end
                αstar = 0.5 * (αlo + αhi)
            end
        end
        for (k, o) in enumerate(fi); u[o] = u[o] + αstar * step[k]; end
        u[ref] = 0.0
        st = dest_stats(log_x, logp, u; ρ = ρ)
    end
    share = st.share
    gerr = maximum(abs.(share .- λ̂))
    gnorm = norm([share[o] - λ̂[o] for o in fi])
    fval = st.logdenom - dot(λ̂, u)
    return DestInversion(u, share, gerr, fval, gnorm, iter, converged && gerr < tol, ref, T(ρ))
end

# ----------------------------------------------------------------------------------------------
# gravity residual  R  (spec §7)
# ----------------------------------------------------------------------------------------------

"Two-way (origin=row, dest=col) fixed-effects demean of a matrix M (M already in logs)."
function two_way_demean(M::AbstractMatrix)
    D1, D2 = size(M)
    ro = sum(M, dims = 2) ./ D2
    co = sum(M, dims = 1) ./ D1
    g = sum(M) / (D1 * D2)
    return M .- ro .- co .+ g
end

struct GravityResidual{T}
    R_sum::T
    R_beta::T
    S_Q::T
    Qt::Matrix{T}
    logAt::Matrix{T}
    ut::Matrix{T}
end

"""
    gravity_residual(u_mat, logτ, logw, σ)

`u_mat[o,d]` = full D×D competitiveness matrix. logA[o,d]=logw_o+logτ_od+u/(σ-1); two-way demean
Q=logτ and logA; R_sum=ΣQ̃·logÃ, R_beta=R_sum/ΣQ̃². Identity: logÃ=Q̃+ũ/(σ-1) (logw is an origin FE).
"""
function gravity_residual(u_mat::AbstractMatrix, logτ::AbstractMatrix, logw::AbstractVector, σ::Real)
    D = size(u_mat, 1)
    Tt = float(promote_type(eltype(u_mat), eltype(logτ), eltype(logw)))
    logA = Matrix{Tt}(undef, D, D)
    @inbounds for o in 1:D, d in 1:D
        logA[o, d] = logw[o] + logτ[o, d] + u_mat[o, d] / (σ - 1)
    end
    Qt = two_way_demean(logτ)
    logAt = two_way_demean(logA)
    ut = two_way_demean(u_mat)
    R_sum = sum(Qt .* logAt)
    S_Q = sum(Qt .* Qt)
    return GravityResidual(R_sum, R_sum / S_Q, S_Q, Matrix{Tt}(Qt), Matrix{Tt}(logAt), Matrix{Tt}(ut))
end

# ----------------------------------------------------------------------------------------------
# draw-level objects and influence function (spec §§8, 11-13)
# ----------------------------------------------------------------------------------------------

"""
    draw_level_objects(log_x, p, u, λ̂; ref=1, ρ=0.0)

At a converged inverted column `u`: `Vexp[s]=exp(V[s])`, `M_d=Σ p·Vexp`, and the free-coord share
perturbation `ξ_free[s,k]=Vexp[s]·(W[s,o]−λ̂[o])` (k indexes free origins o). E_F[ξ_d]=0 at a
converged inversion.
"""
function draw_level_objects(log_x::AbstractMatrix, p::AbstractVector, u::AbstractVector,
                            λ̂::AbstractVector; ref::Int = 1, ρ::Real = 0.0)
    S, D = size(log_x)
    logp = log.(p)
    st = dest_stats(log_x, logp, u; ρ = ρ)
    Vexp = exp.(st.V)
    M_d = dot(p, Vexp)
    fi = free_idx(ref, D)
    ξ_free = Matrix{float(eltype(log_x))}(undef, S, length(fi))
    @inbounds for s in 1:S, (k, o) in enumerate(fi)
        ξ_free[s, k] = Vexp[s] * (st.W[s, o] - λ̂[o])
    end
    return (Vexp = Vexp, M_d = M_d, ξ_free = ξ_free, share = st.share, st = st, fi = fi)
end

struct InfluenceResult{T}
    ψ_R::Vector{T}
    ψ_bar::Vector{T}
    R_sum::T
    R_beta::T
    Eψ::T
    per_dest::Vector{NamedTuple}
end

"""
    influence_function(log_x, p, u_mat, λ̂_mat, omitted, logτ, logw, σ;
                       ref=1, ρ=0.0, scale=:R_beta, ridge=0.0)

Centered gravity influence function ψ̄ (spec §13) for the profiled residual, summing over the
`omitted` destinations. For each omitted d: c_d=Q̃[·,d]/(σ-1) (÷S_Q if scale=:R_beta); solve
H_d a_d = c_d; ψ_R[s] += −⟨a_d, ξ_free[s,·,d]⟩/M_d. Returns ψ_R, ψ̄, the exact residual, E_F[ψ_R],
and per-destination diagnostics (M_d, cond(H_d), σ_min, adjoint residual, share error).
"""
function influence_function(log_x::AbstractMatrix, p::AbstractVector, u_mat::AbstractMatrix,
                            λ̂_mat::AbstractMatrix, omitted::AbstractVector{<:Integer},
                            logτ::AbstractMatrix, logw::AbstractVector, σ::Real;
                            ref::Int = 1, ρ::Real = 0.0, scale::Symbol = :R_beta, ridge::Real = 0.0)
    S, D = size(log_x)
    gr = gravity_residual(u_mat, logτ, logw, σ)
    ψ_R = zeros(float(eltype(log_x)), S)
    per_dest = NamedTuple[]
    scale_div = scale == :R_beta ? gr.S_Q : one(gr.S_Q)
    for d in omitted
        dlo = draw_level_objects(log_x, p, view(u_mat, :, d), view(λ̂_mat, :, d); ref = ref, ρ = ρ)
        fi = dlo.fi
        H = free_hessian(dlo.st, ρ; ref = ref)
        Hr = ridge > 0 ? H + ridge * I : H
        c_d = [gr.Qt[o, d] / (σ - 1) / scale_div for o in fi]
        a_d = try
            Hr \ c_d
        catch
            pinv(Matrix(Hr)) * c_d
        end
        solve_resid = norm(Hr * a_d - c_d)
        @inbounds for s in 1:S
            acc = zero(eltype(ψ_R))
            for k in 1:length(fi); acc += a_d[k] * dlo.ξ_free[s, k]; end
            ψ_R[s] += -acc / dlo.M_d
        end
        push!(per_dest, (d = d, M_d = dlo.M_d, condH = cond(Matrix(H)),
                         smin_H = minimum(svdvals(Matrix(H))), adjoint_resid = solve_resid,
                         max_share_err = maximum(abs.(dlo.share .- λ̂_mat[:, d]))))
    end
    eψ = dot(p, ψ_R)
    return InfluenceResult(ψ_R, ψ_R .- eψ, gr.R_sum, gr.R_beta, eψ, per_dest)
end

end # module
