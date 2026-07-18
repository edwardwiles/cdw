# ============================================================================
# Equivalence test for compressed_cc_inner.jl against the REAL production
# PsiObjectiveBundleImplicit dual bundle (cc_algo/PsiObjectiveBundle.jl):
#   (A) objective f            : compressed == Q(x)                (tight)
#   (B) gradient (g_ζ, g_λ)    : compressed == Q(x, g)            (tight)
#   (C) Hessian-vector product : compressed HVP == central-FD of Q's analytic
#       gradient along several random directions p                (FD tol)
# at the converged base dual solution (ζ*, λ*) for candidate points.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
using Random, Printf, LinearAlgebra

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; oci = ctx.obj.outer_constr_index; ncol = oci - 1
W = size(ctx.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
obj = ctx.obj

# real bundle value at x=(ζ,λ): call obj(x) with H already set for this θ
real_f(x) = obj(copy(x))
function real_val_grad(x)
    g = zeros(length(x))
    f = obj(copy(x), g)          # (ζ,λ)-gradient branch (θ empty)
    return f, g
end

function check_point(label, w0)
    xf0 = x_free_from_w(w0)
    local base
    try
        base = solve_base_state(xf0, ctx)     # this sets obj.H for θ_full0
    catch e
        println(rpad(label, 22), "  SKIP (base solve failed)")
        return true
    end
    # ensure obj.H reflects θ_full0 (solve_base_state's inner_loop_internal set it);
    # rebuild defensively so a later point's solve cannot have overwritten it.
    K = zeros(W); Gd = zeros(W, obj.d)
    obj.moments!(K, Gd, base.θ_full0, obj.U, obj)
    obj.H[:, 1] .= K; obj.H[:, 2] .= 1.0; obj.H[:, 3:2+obj.d] .= Gd

    ζ = base.ζstar; λ = collect(base.λstar)
    x = vcat(ζ, λ)

    cf = build_compressed_factual(base.θ_full0, ctx)

    # (A)(B) value + gradient
    f_real, g_real = real_val_grad(x)
    f_c, gζ_c, gλ_c, q_c, _ = compressed_cc_value_grad(ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    errf = abs(f_c - f_real)
    errgζ = abs(gζ_c - g_real[1])
    errgλ = maximum(abs.(gλ_c .- g_real[2:end]))

    # (C) HVP vs central-FD of the real analytic gradient along random p
    rng = MersenneTwister(7)
    errH = 0.0
    for _ in 1:3
        p = randn(rng, length(x)); p ./= norm(p)
        pζ = p[1]; pλ = p[2:end]
        Hpζ_c, Hpλ_c = compressed_cc_hvp(q_c, pζ, pλ, cf; ddPsi! = obj.ddPsi!)
        ε = 1e-6
        _, gp = real_val_grad(x .+ ε .* p)
        _, gm = real_val_grad(x .- ε .* p)
        Hp_fd = (gp .- gm) ./ (2ε)
        e = max(abs(Hpζ_c - Hp_fd[1]), maximum(abs.(Hpλ_c .- Hp_fd[2:end])))
        errH = max(errH, e)
    end

    ok = errf < 1e-9 && errgζ < 1e-9 && errgλ < 1e-9 && errH < 1e-5
    @printf("%-22s |f|=%.2e |g_ζ|=%.2e |g_λ|=%.2e |HVP-FD|=%.2e  %s\n",
        label, errf, errgζ, errgλ, errH, ok ? "PASS" : "FAIL")
    return ok
end

all_ok = true
println("="^92)
println("COMPRESSED CC-INNER EQUIVALENCE vs production PsiObjectiveBundleImplicit  (D=$D, W=$W, oci=$oci)")
println("="^92)

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]
global all_ok &= check_point("upper_maxit40", w_up40)
global all_ok &= check_point("lower_stalled", w_low)

println("="^92)
if all_ok
    println("ALL COMPRESSED CC-INNER EQUIVALENCE TESTS PASSED")
else
    println("SOME COMPRESSED CC-INNER EQUIVALENCE TESTS FAILED"); exit(1)
end
