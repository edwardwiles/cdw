# ============================================================================
# Task §4 / §9 / §12A-D: the three distinct scalar objects, for the TRUE
# full-A d-moment problem (d=18 at D=4, NOT the sequential method's reduced
# D+2 problem -- see docs/fullA_d4_code_audit.md sec 6 for why that method's
# existing fixed_dual/full_fixed_dual_criterion.jl machinery does not apply
# here as-is). All three are defined so that AT THE BASE POINT x0 they are
# EXACTLY equal to `evaluate_fullA(x0,ctx).Delta_dual` (same sign convention,
# verified in oracle.jl: Delta(theta) = -[mean(Psi(q)) + zeta]) -- this is a
# hard equality check test_three_way.jl enforces, not a "should be close"
# one, since at x=x0 all three formulas literally reduce to the same
# computation.
# ============================================================================

"BaseDualState: the frozen (theta0, zeta*, lambda*, m*, G0) state every base point needs."
struct BaseDualState
    x_free0::Vector{Float64}
    θ_full0::Vector{Float64}
    ζstar::Float64
    λstar::Vector{Float64}
    m_star::Vector{Float64}     # obj.arg1 at the converged base solve = dPsi(q_s*), un-normalized LFD weight
    inner_status::Int
end

function solve_base_state(x_free0::AbstractVector, ctx)
    obj = ctx.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    K, inner_x, nStatus = CS.inner_loop_internal(obj, θ_full0)
    nStatus in (0, -100, -101, -103) || error("solve_base_state: inner solve failed, nStatus=$nStatus")
    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj(inner_x, constr = @view(cbuf[1:ncon]))   # populates obj.arg1 = m(s) at this (theta0, zeta*, lambda*)
    m_star = copy(obj.arg1)
    return BaseDualState(collect(x_free0), θ_full0, inner_x[1], collect(inner_x[2:end]), m_star, nStatus)
end

"""
    frozen_adjoint_Q(x_free, ctx, base::BaseDualState) -> Float64

Q_adj(x;x0,y*) = mean_s[m_s* * lambda*'G_s(x)] - zeta*. m_s*, lambda*, zeta*
are FROZEN at the base solve; G_s(x) is FULLY recomputed (fresh moments!) at
the perturbed x. LINEAR in G_s(x) (does NOT re-evaluate Psi) -- the bilinear
term is exactly the existing Method-B envelope-scalar construction
(`full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl::_methodB_fullA_envelope_scalar`,
reused conceptually, not copy-pasted).

SIGN, verified not assumed: the task brief's schematic `Q_adj =
-mean_s m_s*lambda*'G_s(x)` (no zeta term), taken literally with this
codebase's Delta=-f convention, gives dQ_adj/dtheta|x0 EXACTLY EQUAL TO
-dL_fix/dtheta|x0 (a clean sign flip, derivable by hand from the chain rule
plus the KKT identities mean(m*)=1, mean(m*G_j*)=0 -- not a numerical
artifact). Caught empirically: the first version of this function (with the
brief's literal sign) gave FD slopes of OPPOSITE SIGN from L_fix/optimized_Delta
even at h=1e-4, which a genuine small-h envelope agreement should never do.
Fixed here by flipping the bilinear term's sign (keeping -zeta* as the
additive constant) so dQ_adj/dtheta|x0 == dL_fix/dtheta|x0 by construction;
re-verified by test_three_way.jl's h-sweep after the fix.
"""
function frozen_adjoint_Q(x_free::AbstractVector, ctx, base::BaseDualState)
    obj = ctx.obj
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    W = size(obj.U, 1); d = obj.d
    K = zeros(eltype(θ_full), W); G = zeros(eltype(θ_full), W, d)
    obj.moments!(K, G, θ_full, obj.U, obj)
    oci = obj.outer_constr_index
    acc = zero(eltype(θ_full))
    @inbounds for s in 1:W
        acc += base.m_star[s] * dot(@view(base.λstar[:]), @view(G[s, 1:oci-1]))
    end
    return (acc / W) - base.ζstar
end

"""
    fixed_dual_L(x_free, ctx, base::BaseDualState) -> Float64

L_fix(x;y*) = -[mean_s Psi(-zeta* - lambda*'G_s(x)) + zeta*]. zeta*,lambda*
FROZEN; G_s(x) fully recomputed; Psi FULLY RE-EVALUATED (nonlinear in G_s(x)
through Psi, unlike frozen_adjoint_Q which is linear). This is the object
task §4.B / §12.C describes -- distinct from frozen_adjoint_Q (which never
re-evaluates Psi) and from optimized_Delta (which re-solves the dual).
"""
function fixed_dual_L(x_free::AbstractVector, ctx, base::BaseDualState)
    obj = ctx.obj
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    W = size(obj.U, 1); d = obj.d
    K = zeros(eltype(θ_full), W); G = zeros(eltype(θ_full), W, d)
    obj.moments!(K, G, θ_full, obj.U, obj)
    oci = obj.outer_constr_index
    q = [-base.ζstar - dot(base.λstar, @view(G[s, 1:oci-1])) for s in 1:W]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return -(sum(Psi_q) / W + base.ζstar)
end

"""
    optimized_Delta(x_free, ctx; warm=true) -> Float64

The fully re-solved value Delta(x) = min_y L(x,y) -- re-solves the inner CC
dual from scratch (or warm-started) at every x. This is `evaluate_fullA`'s
`Delta_dual` field; exposed here under the task's own naming for clarity in
the three-way comparison scripts (§4.C / §12.D).
"""
function optimized_Delta(x_free::AbstractVector, ctx; warm::Bool = true)
    return evaluate_fullA(x_free, ctx; cache = nothing, warm = warm).Delta_dual
end
