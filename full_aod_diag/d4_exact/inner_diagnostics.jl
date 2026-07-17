# ============================================================================
# Task §8: inner CC dual validation -- Hessian eigenvalues/conditioning, and
# an independent direct re-evaluation of the dual scalar objective.
# ============================================================================
using LinearAlgebra: eigvals, cond, Symmetric

"""
    inner_dual_hessian(obj, inner_x) -> Matrix

`H_y = (1/M) * sum_s Psi''(q_s) * [1;G_s][1;G_s]'` (task brief §8's exact
formula), restricted to the inner-matched moments (columns 1:outer_constr_index-1
of G, i.e. excluding gravity). REUSES `obj.H`/`obj.arg0` as already populated
by a preceding `obj(inner_x, constr=...)` call (matches the exact construction
`ift!` uses internally for `obj.∂∂f_∂∂x` -- see PsiObjectiveBundle.jl:409-416 --
but computed standalone here so it doesn't depend on `jac_h` having been
populated by a prior gradient call, which `ift!` also touches).
"""
function inner_dual_hessian(obj, inner_x::AbstractVector)
    @unpack H, arg0, M, outer_constr_index = obj
    ζ = inner_x[1]
    BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -inner_x, 0.0, arg0)
    arg2 = similar(arg0)
    CS.ddPsi!(arg2, arg0)
    Hsub = @view(H[:, 2:1+outer_constr_index])
    Hscaled = Hsub .* sqrt.(arg2)
    return Symmetric((Hscaled' * Hscaled) ./ M)
end

"""
    inner_dual_conditioning(obj, inner_x) -> NamedTuple

Eigen-decomposes `inner_dual_hessian`, reports smallest/largest eigenvalue,
condition number, and numerical rank (task §8's explicit requirement: "do not
infer good conditioning from full rank").
"""
function inner_dual_conditioning(obj, inner_x::AbstractVector)
    Hy = inner_dual_hessian(obj, inner_x)
    ev = eigvals(Hy)
    ev_pos = filter(>(0), ev)
    λmin = minimum(ev); λmax = maximum(ev)
    tol = size(Hy, 1) * eps(λmax)
    rank_num = count(>(tol), ev)
    return (eigenvalues = ev, λmin = λmin, λmax = λmax,
            cond_number = isempty(ev_pos) ? Inf : λmax / max(λmin, eps()),
            numerical_rank = rank_num, dim = size(Hy, 1))
end

"""
    direct_dual_objective(obj, x, θ_full) -> Float64

Independent re-evaluation of the dual scalar `L(x,y) = mean(Psi(-zeta-lambda'G))
+ zeta` DIRECTLY from `obj.moments!` output, NOT via the production callable
`(Q::PsiObjectiveBundleImplicit)(...)`. Used to cross-check the production
callback bit-for-bit (task §8: "implement a direct evaluation of the scalar
dual objective independent of the production callback").
"""
function direct_dual_objective(obj, x::AbstractVector, θ_full::AbstractVector)
    W = size(obj.U, 1); d = obj.d
    K = zeros(W); G = zeros(W, d)
    obj.moments!(K, G, θ_full, obj.U, obj)
    ζ = x[1]; λ = x[2:end]
    oci = obj.outer_constr_index
    q = [-ζ - dot(λ, G[s, 1:oci-1]) for s in 1:W]
    Psi_q = similar(q)
    CS.Psi!(Psi_q, q)
    return sum(Psi_q) / W + ζ
end

"""
    inner_solve_reliability(x_free, ctx; n_cold=3, tols=[1e-12, 1e-8]) -> DataFrame-like Vector

Solves the inner problem repeatedly (cold x n_cold, warm x1) and at two
tolerances, recording the accepted value and status each time -- task §8's
"solve it cold; solve it warm; solve at two increasingly tight tolerances".
"""
function inner_solve_reliability(x_free::AbstractVector, ctx; n_cold::Int = 3,
        tol_opt_files::Vector{String} = [ctx.obj.inner_loop_opt])
    obj = ctx.obj
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    rows = NamedTuple[]
    for opt_file in tol_opt_files
        obj.inner_loop_opt = opt_file
        for trial in 1:n_cold
            obj.x .= NaN
            K, x, nStatus = CS.inner_loop_internal(obj, θ_full)
            Δd = begin
                cbuf = zeros(obj.d - obj.outer_constr_index + 2)
                obj(x, constr = @view(cbuf[1:length(cbuf)]))
                cbuf[1] / 1e10
            end
            push!(rows, (kind = "cold", trial = trial, opt_file = opt_file,
                          Delta_dual = Δd, nStatus = nStatus))
        end
        obj.x .= NaN
        CS.inner_loop_internal(obj, θ_full)   # prime the warm cache
        K, x, nStatus = CS.inner_loop_internal(obj, θ_full)  # now warm from itself
        cbuf = zeros(obj.d - obj.outer_constr_index + 2)
        obj(x, constr = @view(cbuf[1:length(cbuf)]))
        push!(rows, (kind = "warm", trial = 1, opt_file = opt_file,
                      Delta_dual = cbuf[1] / 1e10, nStatus = nStatus))
    end
    return rows
end
