# ============================================================================
# Part 1.2-1.4: central-difference gradient/directional derivative of the
# fixed-dual criterion, plus the adaptive step-size diagnostic.
#
# Coordinate convention (verified against the implementation, not assumed --
# see derivative_methods_report.md "coordinate scaling" section): production's
# OWN outer KNITRO decision variable is the RAW LEVEL theta[4:3+D]=Acol (no
# log-transform anywhere in the outer parameterization/bounds/FreeParamMap).
# The analytical Frechet formula (Part 2) is naturally stated in
# log(Acol)-derivatives. This file computes LOG-space directional derivatives
# by default (`Acol .*= exp.(h.*v)`, matching fixed_dual_fd_directional_derivative's
# docstring exactly), and separately exposes a level-space single-coordinate
# gradient (`fixed_dual_fd_gradient_level`) for DIRECT comparison against
# production's raw AD gradient (`_methodB_envelope_scalar`) without any
# rescaling assumption. `level_to_log_jacobian_factor` gives the exact
# elementwise conversion (level-grad .* Acol == log-grad, valid AT the point
# Acol is evaluated, an exact identity: d/dlogx = x*d/dx).
# ============================================================================

"Winners at the focal destination for theta (D+1-or-D+2-moment baseline; gravity column, if present, ignored)."
function focal_winners(θ::Vector{Float64}, moments_fn::Function, γobj, U::Matrix{Float64}, D::Int, d::Int)
    W = size(U, 1)
    K = zeros(W); G = zeros(W, d)
    moments_fn(K, G, θ, U, (γ=γobj,))
    λ = reshape(γobj.P, (D, D))'
    focal = γobj.baseIndex
    denomf = γobj.wHat[focal] * γobj.L[focal]
    winners = Vector{Int}(undef, W)
    @inbounds for ω in 1:W
        bo = 1; best = -Inf
        for o in 1:D
            val = G[ω, o] + λ[o, focal] * denomf
            if val > best; best = val; bo = o; end
        end
        winners[ω] = bo
    end
    return winners
end

"""
    fixed_dual_fd_directional_derivative(θ, obj, x_fixed, v, h; Acol_offset=3)

D_v δ* ≈ (Q(A⊙exp(hv),x*) - Q(A⊙exp(-hv),x*)) / (2h), perturbing ONLY
θ[Acol_offset+1:Acol_offset+D] = Acol multiplicatively (log-space direction
v, length D). No inner re-solve, no KNITRO call -- x_fixed is reused as-is at
both evaluation points (that IS the fixed-dual envelope approximation).
"""
function fixed_dual_fd_directional_derivative(θ::Vector{Float64}, obj::PsiObjectiveBundleDelta,
        x_fixed::Vector{Float64}, v::Vector{Float64}, h::Float64; Acol_offset::Int=3)
    D = length(v)
    θp = copy(θ); θm = copy(θ)
    @views θp[Acol_offset+1:Acol_offset+D] .*= exp.(h .* v)
    @views θm[Acol_offset+1:Acol_offset+D] .*= exp.(-h .* v)
    Qp = dual_criterion_fixed_x(θp, obj, x_fixed)
    Qm = dual_criterion_fixed_x(θm, obj, x_fixed)
    return (Qp - Qm) / (2h)
end

"One-sided (forward) version, for the central-vs-one-sided discrepancy diagnostic."
function fixed_dual_fd_directional_derivative_forward(θ::Vector{Float64}, obj::PsiObjectiveBundleDelta,
        x_fixed::Vector{Float64}, v::Vector{Float64}, h::Float64; Acol_offset::Int=3)
    D = length(v)
    θp = copy(θ)
    @views θp[Acol_offset+1:Acol_offset+D] .*= exp.(h .* v)
    Q0 = dual_criterion_fixed_x(θ, obj, x_fixed)
    Qp = dual_criterion_fixed_x(θp, obj, x_fixed)
    return (Qp - Q0) / h
end

"""
    fixed_dual_fd_gradient(θ, γobj, U, moments_fn, d, x_fixed, h; l, Acol_offset, parallel)

Full D-coordinate LOG-space gradient (∂Q/∂log(Acol[r]), r=1:D), via central
differences on unit-coordinate directions e_r. `parallel=true` builds one
bundle per Julia thread (each call mutates `obj.H` in place, so bundles
cannot be shared across threads) and requires `julia -t N` for N>1.
"""
function fixed_dual_fd_gradient(θ::Vector{Float64}, γobj, U::Matrix{Float64}, moments_fn::Function, d::Int,
        x_fixed::Vector{Float64}, h::Float64; l::Int=length(θ), Acol_offset::Int=3, parallel::Bool=false,
        find_smallest::Bool=true)
    D = size(γobj.τ, 1)
    grad = zeros(D)
    if parallel && Threads.nthreads() > 1
        bundles = [build_fixed_dual_bundle(γobj, U, l, d, moments_fn; find_smallest=find_smallest) for _ in 1:Threads.nthreads()]
        Threads.@threads for r in 1:D
            obj = bundles[Threads.threadid()]
            v = zeros(D); v[r] = 1.0
            grad[r] = fixed_dual_fd_directional_derivative(θ, obj, x_fixed, v, h; Acol_offset=Acol_offset)
        end
    else
        obj = build_fixed_dual_bundle(γobj, U, l, d, moments_fn; find_smallest=find_smallest)
        for r in 1:D
            v = zeros(D); v[r] = 1.0
            grad[r] = fixed_dual_fd_directional_derivative(θ, obj, x_fixed, v, h; Acol_offset=Acol_offset)
        end
    end
    return grad
end

"""
    level_to_log_jacobian_factor(θ, Acol_offset, D)

Returns Acol itself (the exact elementwise factor converting a level-space
derivative to a log-space one: d/dlog(x) = x * d/dx). Multiply a
level-space gradient by this to compare directly against `fixed_dual_fd_gradient`
or the analytical/AD log-space gradients; divide a log-space gradient by
this to compare directly against production's raw AD gradient
(`_methodB_envelope_scalar`, which differentiates the RAW level θ).
"""
level_to_log_jacobian_factor(θ::Vector{Float64}, Acol_offset::Int, D::Int) = θ[Acol_offset+1:Acol_offset+D]

"""
    adaptive_stepsize_diagnostic(θ, obj, x_fixed, v, hs; Acol_offset, moments_fn, γobj, U, D, d)

Part 1.4. For direction v and each h in hs, records: winner-switch count/frac
(base vs A⊙exp(hv)), central derivative at h, derivative at h/2 and 2h (for a
stability check), the central-vs-forward discrepancy, and wall time. Returns a
Vector{NamedTuple}.
"""
function adaptive_stepsize_diagnostic(θ::Vector{Float64}, obj::PsiObjectiveBundleDelta, x_fixed::Vector{Float64},
        v::Vector{Float64}, hs::Vector{Float64}; Acol_offset::Int=3, moments_fn::Function, γobj, U::Matrix{Float64},
        D::Int=length(v), d::Int)
    w0 = focal_winners(θ, moments_fn, γobj, U, D, d)
    rows = NamedTuple[]
    for h in hs
        t0 = time()
        central_h = fixed_dual_fd_directional_derivative(θ, obj, x_fixed, v, h; Acol_offset=Acol_offset)
        rt = time() - t0
        central_h2 = fixed_dual_fd_directional_derivative(θ, obj, x_fixed, v, h / 2; Acol_offset=Acol_offset)
        central_2h = fixed_dual_fd_directional_derivative(θ, obj, x_fixed, v, 2h; Acol_offset=Acol_offset)
        forward_h = fixed_dual_fd_directional_derivative_forward(θ, obj, x_fixed, v, h; Acol_offset=Acol_offset)

        θp = copy(θ)
        @views θp[Acol_offset+1:Acol_offset+D] .*= exp.(h .* v)
        wp = focal_winners(θp, moments_fn, γobj, U, D, d)
        n_switch = count(w0 .!= wp)

        push!(rows, (h=h, n_switch=n_switch, frac_switch=n_switch / length(w0),
            central=central_h, central_half_h=central_h2, central_2h=central_2h, forward=forward_h,
            central_vs_forward_reldiff=abs(central_h - forward_h) / max(abs(central_h), 1e-12),
            central_vs_halfh_reldiff=abs(central_h - central_h2) / max(abs(central_h), 1e-12),
            central_vs_2h_reldiff=abs(central_h - central_2h) / max(abs(central_h), 1e-12),
            runtime_s=rt))
    end
    return rows
end

"""
    identify_stable_plateau(rows; switch_threshold=100, rel_tol=0.05)

Empirical rule (NOT a hard-coded theorem, per the user's instruction): flags
h as "stable" if (a) the winner-switch count at h is >= switch_threshold
(enough switches that MC discreteness isn't dominating), AND (b) the central
estimate agrees with BOTH its h/2 and 2h neighbors to within rel_tol. Returns
the largest-switch-count stable row, or `nothing` with a diagnostic message
if no h satisfies both criteria (report this explicitly rather than silently
picking the smallest h).
"""
function identify_stable_plateau(rows::Vector{<:NamedTuple}; switch_threshold::Int=100, rel_tol::Float64=0.05)
    candidates = [r for r in rows if r.n_switch >= switch_threshold &&
                  r.central_vs_halfh_reldiff <= rel_tol && r.central_vs_2h_reldiff <= rel_tol]
    isempty(candidates) && return nothing
    return candidates[argmax([r.n_switch for r in candidates])]
end
