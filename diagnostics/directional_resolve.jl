# ============================================================================
# Diagnostic 4-lite: fully re-solved directional profile test at A=A*, for
# gamma'_focal targets away from the Frechet benchmark. Tests directly whether
# moving A_od away from A* lowers the profiled minimum divergence delta*(A,
# gamma'), by (a) fully re-solving the CC inner divergence problem (the SAME
# nonlinear KNITRO solve production uses, no linearization) at perturbed A,
# and (b) comparing the realized finite-difference slope to the
# envelope-theorem slope from production's own gradient construction.
#
# Uses a SEPARATE gravMoment=0 context (ctx_nograv) so the D^2+1-moment
# PsiObjectiveBundleDelta's G columns are exactly [D^2 trade shares, 1
# price-index-consistency moment] with no gravity-column collision (see
# A_profile_optimality_report.md). Gravity feasibility of a perturbation
# direction is instead guaranteed EXACTLY (not just to first order) by
# projecting the direction to be orthogonal to vec(q_tilde) in log(Aod_theta)
# space -- valid because the gravity moment is EXACTLY LINEAR in log(Aod_theta)
# (gravity_tariff.jl), so this is a genuine (not linearized) feasibility
# guarantee for ALL h, not just small h.
# ============================================================================

"Build the D^2+1-moment 'exact profiled delta*' bundle for the full-A model (no gravity column)."
function build_deltastar_bundle(ctx_nograv; δ::Float64=1.0, find_smallest::Bool=true)
    D = ctx_nograv.D
    d = D^2 + 1
    return PsiObjectiveBundleDelta(γ=ctx_nograv.γobj, (moments!) = EK_moments_gammanorm_directgp!,
        moments_jacobian! = error, d=d, outer_constr_index=d + 1,
        inequality_index=Int64[], complement_index=[0 0],
        l=length(ctx_nograv.θ0_up), U=ctx_nograv.U, N=ctx_nograv.params.Jac_W,
        lower_limit=-5000.0, use_cached_x=true, find_smallest=find_smallest,
        outer_loop_opt="ek_outer_loop_options.opt", inner_loop_opt="ek_inner_loop_options.opt")
end

"Sign convention matching inner_loop()'s own (val *= -1 iff find_smallest)."
deltastar_sign(bundle) = bundle.find_smallest ? -1.0 : 1.0

"Solve delta*(theta) exactly (fully re-solving the CC inner divergence problem). Returns (deltastar, x, nStatus)."
function solve_deltastar(bundle, θ::Vector{Float64})
    val, x, nStatus = inner_loop(bundle, θ)
    return val, x, nStatus
end

"""
    envelope_scalar_deltastar(θ, moments!, γobj, U, λ, arg1, d)

dδ*/dθ envelope scalar (NOT the 1e10-scaled constraint gradient production
uses internally -- this is the plain derivative of the raw solved objective
`f`), holding (λ,arg1) fixed at their value from the real inner solve at θ.
Mirrors PsiObjectiveBundleImplicitMethodB_fullA.jl::_methodB_fullA_envelope_scalar's
derivation exactly, adapted to PsiObjectiveBundleDelta's (ζ,λ_1..λ_d) layout
(outer_constr_index=d+1, i.e. ALL d moments weighted, zero left over).
"""
function envelope_scalar_deltastar(θ::AbstractVector, moments!::Function, γobj, U::AbstractMatrix,
        λ::AbstractVector, arg1::Vector{Float64}, d::Int)
    N = size(U, 1)
    T = eltype(θ)
    K = zeros(T, N)
    G = zeros(T, N, d)
    moments!(K, G, θ, U, (γ=γobj,))
    s = zero(T)
    @inbounds for draw in 1:N
        acc = zero(T)
        for k in 1:d
            acc += λ[k] * G[draw, k]
        end
        s += arg1[draw] * acc
    end
    return -s / N
end

"""
    envelope_gradient_at(bundle, ctx_nograv, θ)

Solves delta*(θ) (if not already solved / to refresh bundle.x, bundle.arg1),
then returns (deltastar_signed, full_gradient_signed) where full_gradient
is length l = length(θ), in the SAME sign convention `inner_loop` reports
(deltastar_sign(bundle) applied to both).
"""
function envelope_gradient_at(bundle, ctx_nograv, θ::Vector{Float64})
    val, xopt, nStatus = solve_deltastar(bundle, θ)
    nStatus ∈ (0, -100, -101, -103) || return (val, nothing, nStatus)
    # refresh bundle.arg1 at the solved x (constr nonempty triggers dPsi! internally; use g nonempty, theta empty)
    bundle(xopt, zeros(length(xopt)))
    λ = collect(@view xopt[2:end])
    S = deltastar_sign(bundle)
    g = ForwardDiff.gradient(θθ -> envelope_scalar_deltastar(θθ, bundle.moments!, ctx_nograv.γobj,
            bundle.U[1:bundle.N, :], λ, bundle.arg1, bundle.d), θ)
    return (val, S .* g, nStatus)
end

"Flatten a D x D matrix (indexed [j,d]) into the idx_in(j,d)=j+(d-1)*D vector ordering -- this IS Julia's column-major vec()."
flatten_jd(M::AbstractMatrix) = vec(M)

"Project a length-D^2 direction (idx_in ordering) to be EXACTLY orthogonal to vec(q_tilde) (exact gravity feasibility for all h, since gravity is linear)."
function project_gravity_feasible(v::Vector{Float64}, qvec::Vector{Float64})
    return v .- (dot(v, qvec) / dot(qvec, qvec)) .* qvec
end

"""
    directional_resolve_report(ctx, ctx_nograv, γp_target, directions; hs, δ_budget)

For a fixed gamma'_focal target (θ0_up[3+D] overridden to γp_target) and A=A*
(Aod_theta≡1) as the base point, for each named direction (already gravity-
feasible, in idx_in ordering, length D^2), evaluates delta*(A(h),γp_target)
at each h in hs (both signs), warm-starting the bundle's cached x from the
PREVIOUS h in the same direction, plus one COLD start (fresh bundle) at the
largest |h| as a cross-check. Returns a Vector of NamedTuple rows.
"""
function directional_resolve_report(ctx, ctx_nograv, γp_target::Float64,
        directions::Vector{Tuple{String,Vector{Float64}}}; hs::Vector{Float64}, δ_budget::Float64=1.0)
    D = ctx.D
    Aod_offset = ctx.Aod_offset
    # true Frechet benchmark A_od_theta block (NOT all-ones under the gammanorm gauge -- see context.jl note)
    Aod0 = ctx_nograv.θ0_up[Aod_offset+1:Aod_offset+D^2]

    θbase = copy(ctx_nograv.θ0_up)
    θbase[3+D] = γp_target
    θ0 = make_full_theta(θbase, Aod0, Aod_offset, D)

    bundle0 = build_deltastar_bundle(ctx_nograv; δ=δ_budget)
    val0, gfull0, status0 = envelope_gradient_at(bundle0, ctx_nograv, θ0)
    g_prod_Aod = gfull0 === nothing ? nothing : gfull0[Aod_offset+1:Aod_offset+D^2]

    rows = NamedTuple[]
    push!(rows, (direction="baseline(h=0)", h=0.0, γp_target=γp_target, deltastar=val0, nStatus=status0,
        fd_slope=NaN, envelope_slope=NaN, warm=false))

    for (name, v) in directions
        # fresh warm-startable bundle per direction, seeded from the h=0 solution
        bundle = build_deltastar_bundle(ctx_nograv; δ=δ_budget)
        bundle.x .= bundle0.x
        prev_val = val0
        for h in hs
            Aod_h = Aod0 .* exp.(h .* v)
            θh = make_full_theta(θbase, Aod_h, Aod_offset, D)
            val, xopt, nStatus = solve_deltastar(bundle, θh)
            ok = nStatus ∈ (0, -100, -101, -103)
            fd_slope = ok ? (val - val0) / h : NaN
            envelope_slope = (g_prod_Aod === nothing) ? NaN : dot(g_prod_Aod, v)
            push!(rows, (direction=name, h=h, γp_target=γp_target, deltastar=ok ? val : NaN, nStatus=nStatus,
                fd_slope=fd_slope, envelope_slope=envelope_slope, warm=true))
        end
        # one COLD start at the largest |h| tested, as a cross-check
        hmax = hs[argmax(abs.(hs))]
        Aod_cold = Aod0 .* exp.(hmax .* v)
        θcold = make_full_theta(θbase, Aod_cold, Aod_offset, D)
        cold_bundle = build_deltastar_bundle(ctx_nograv; δ=δ_budget)
        val_cold, _, status_cold = solve_deltastar(cold_bundle, θcold)
        push!(rows, (direction=name * "[COLD@hmax]", h=hmax, γp_target=γp_target,
            deltastar=status_cold ∈ (0, -100, -101, -103) ? val_cold : NaN, nStatus=status_cold,
            fd_slope=NaN, envelope_slope=NaN, warm=false))
    end
    return rows, g_prod_Aod
end
