# ============================================================================
# §5: the four benchmark derivative methods, all operating on the canonical
# moment_map / envelope_scalar_div_ctx from derivative_core.jl.
# ============================================================================
using SparseConnectivityTracer, SparseDiffTools, DifferentiationInterface, SparseMatrixColorings
import Enzyme
const DI = DifferentiationInterface

# ---------------------------------------------------------------------------
# METHOD A: dense ForwardDiff Jacobian of the FULL moment map, then contract
# with (λ, arg1) exactly as production's calculate_jac_θ_autodiff! + the
# PsiObjectiveBundle.jl:216-222 contraction do.
# ---------------------------------------------------------------------------
function method_A_dense_jacobian(θ, ctx)
    N = size(ctx.U, 1)
    f! = (Hvec, θ) -> begin
        H = reshape(Hvec, N, ctx.d + 2)
        moment_map!(H, θ, ctx.U, ctx.γobj)
    end
    Hvec0 = zeros(N * (ctx.d + 2))
    J = ForwardDiff.jacobian(f!, Hvec0, θ)   # (N*(d+2)) × l
    return J   # caller contracts
end

"Contract Method A's dense Jacobian exactly as production does for the divergence constraint."
function contract_dense_to_div_grad(J, ctx)
    N = size(ctx.U, 1)
    l = size(J, 2)
    Jr = reshape(J, N, ctx.d + 2, l)
    g = zeros(l)
    for i in 1:l
        acc = zeros(N)
        for j in 1:ctx.outer_constr_index-1
            acc .+= ctx.λ[j] .* Jr[:, 2+j, i]   # H col (2+j) = G col j
        end
        g[i] = (1e10 / N) * dot(ctx.arg1, acc)
    end
    return g
end

# ---------------------------------------------------------------------------
# METHOD B: ForwardDiff direct scalar gradient (no dense Jacobian ever built).
# ---------------------------------------------------------------------------
function method_B_forward_scalar(θ, ctx)
    return ForwardDiff.gradient(θ -> envelope_scalar_div_ctx(θ, ctx), θ)
end

# ---------------------------------------------------------------------------
# METHOD C: Enzyme reverse-mode gradient of the same scalar.
# ---------------------------------------------------------------------------
function method_C_enzyme_reverse(θ, ctx)
    f = θ -> envelope_scalar_div_ctx(θ, ctx)
    g = DI.gradient(f, DI.AutoEnzyme(mode=Enzyme.Reverse), θ)
    return g
end

# ---------------------------------------------------------------------------
# METHOD D: colored sparse ForwardDiff full Jacobian of the moment map.
# ---------------------------------------------------------------------------
function method_D_sparse_prep(θ0, ctx)
    N = size(ctx.U, 1)
    f! = (Hvec, θ) -> begin
        H = reshape(Hvec, N, ctx.d + 2)
        moment_map!(H, θ, ctx.U, ctx.γobj)
    end
    Hvec0 = zeros(N * (ctx.d + 2))
    backend = DI.AutoSparse(DI.AutoForwardDiff();
        sparsity_detector = DI.DenseSparsityDetector(DI.AutoForwardDiff(); atol=1e-12),
        coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
    prep = DI.prepare_jacobian(f!, Hvec0, backend, θ0)
    return prep, backend, f!, Hvec0
end

function method_D_sparse_jacobian(θ, ctx, prep, backend, f!, Hvec0)
    J = DI.jacobian(f!, copy(Hvec0), prep, backend, θ)
    return J
end
