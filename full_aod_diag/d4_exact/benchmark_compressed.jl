# ============================================================================
# Dense vs compressed benchmark for the factual-moment fixed-dual pipeline.
#   - dense component breakdown (score / winner / CES value / alloc+zero /
#     write+centering / counterfactual) at D=4/W=8000
#   - dense vs compressed across (D,W): build, forward contraction, transpose
#     (dual-gradient), full CC value+grad; time + allocations
# Uses the theta==1 calibration point (always constructible), random duals
# (contraction cost is dual-independent). All numbers warmed (JIT excluded),
# median of N reps, JULIA_NUM_THREADS from the environment.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))   # pulls context.jl; do NOT re-include context.jl
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
using Printf, Statistics, LinearAlgebra, SpecialFunctions

const NTHREADS = Threads.nthreads()
med(f, N) = median([(GC.gc(false); @elapsed f()) for _ in 1:N])

make_ctx(D, W) = (D == 4 && W == 8000) ? d4_exact_setup(find_smallest = true) :
                 d_exact_setup_scaled(D = D, W = W, find_smallest = true)

# theta==1 calibration-style point (Aod_theta = 1 everywhere), always constructible
calib_xfree(ctx) = vcat(ctx.θ0_up[3+ctx.D], ones(ctx.D^2))

function bench_row(D, W; N = 20)
    ctx = make_ctx(D, W)
    obj = ctx.obj; Wn = size(ctx.U, 1); d = obj.d; oci = obj.outer_constr_index; ncol = oci - 1
    θ_full = CS.reconstruct_full(calib_xfree(ctx), ctx.m)
    K = zeros(Wn); G = zeros(Wn, d)
    β = randn(ncol); w = randn(Wn)

    # ---- warm-up ----
    obj.moments!(K, G, θ_full, ctx.U, obj)
    cf = build_compressed_factual(θ_full, ctx; check_ties = false)
    compressed_dual_contraction(β, cf)
    compressed_transpose_contraction(w, cf)
    dense_contract(G, β) = [dot(@view(G[s, 1:ncol]), β) for s in 1:Wn]
    dense_transpose(G, w) = (@view(G[:, 1:ncol])' * w)
    dense_contract(G, β); dense_transpose(G, w)

    # ---- moment build ----
    t_dense_build = med(() -> obj.moments!(K, G, θ_full, ctx.U, obj), N)
    t_comp_build = med(() -> build_compressed_factual(θ_full, ctx; check_ties = false), N)
    a_dense_build = @allocated obj.moments!(K, G, θ_full, ctx.U, obj)
    a_comp_build = @allocated build_compressed_factual(θ_full, ctx; check_ties = false)

    # ---- forward contraction (λ'G_s for all s) ----
    t_dense_contract = med(() -> dense_contract(G, β), N)
    t_comp_contract = med(() -> compressed_dual_contraction(β, cf), N)

    # ---- transpose contraction (dual gradient G'w) ----
    t_dense_trans = med(() -> dense_transpose(G, w), N)
    t_comp_trans = med(() -> compressed_transpose_contraction(w, cf), N)

    # ---- full CC value+grad from scratch (build + contraction + Psi + grad) ----
    #   dense:   obj.moments! ; q=-ζ-G λ ; Psi ; dPsi ; g_λ = -(1/M)G'dPsi
    #   compressed: build_cf ; compressed_cc_value_grad (+ nothing else)
    ζ = 0.5
    function dense_cc_valuegrad()
        obj.moments!(K, G, θ_full, ctx.U, obj)
        q = [-ζ - dot(@view(G[s, 1:ncol]), β) for s in 1:Wn]
        Psq = similar(q); obj.Psi!(Psq, q)
        dPsq = similar(q); obj.dPsi!(dPsq, q)
        f = sum(Psq) / Wn + ζ
        gλ = -(1.0 / Wn) .* (@view(G[:, 1:ncol])' * dPsq)
        return f, gλ
    end
    function comp_cc_valuegrad()
        c = build_compressed_factual(θ_full, ctx; check_ties = false)
        return compressed_cc_value_grad(ζ, β, c; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    end
    dense_cc_valuegrad(); comp_cc_valuegrad()
    t_dense_cc = med(dense_cc_valuegrad, N)
    t_comp_cc = med(comp_cc_valuegrad, N)

    ms(x) = x * 1000
    @printf("D=%-2d W=%-6d | build: dense %7.3f / comp %7.3f ms (%.2fx) | contract: dense %7.4f / comp %7.4f ms (%.1fx) | transp: dense %7.4f / comp %7.4f ms (%.1fx) | CC v+g: dense %7.3f / comp %7.3f ms (%.2fx)\n",
        D, W, ms(t_dense_build), ms(t_comp_build), t_dense_build / t_comp_build,
        ms(t_dense_contract), ms(t_comp_contract), t_dense_contract / t_comp_contract,
        ms(t_dense_trans), ms(t_comp_trans), t_dense_trans / t_comp_trans,
        ms(t_dense_cc), ms(t_comp_cc), t_dense_cc / t_comp_cc)
    @printf("            allocs: dense build %8.1f KB / comp build %8.1f KB (%.1fx less)\n",
        a_dense_build / 1024, a_comp_build / 1024, a_dense_build / max(a_comp_build, 1))
    return nothing
end

# ---- dense component breakdown at D=4/W=8000 (maps to the task's requested phases) ----
function dense_component_profile(; N = 50)
    ctx = d4_exact_setup(find_smallest = true)
    D = ctx.D; W = size(ctx.U, 1); μ = ctx.θ0_up[1]; σ = ctx.σ; γo = ctx.γ
    θ_full = CS.reconstruct_full(calib_xfree(ctx), ctx.m)
    _, _, AodPow = factual_prices(θ_full, ctx)
    constCons = [γo.wHat[o] * AodPow[o, d] * γo.τ[o, d] for o in 1:D, d in 1:D]
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    U = ctx.U; Uσ = γo.Uσ
    println("-"^92); println("Dense factual-block component breakdown (D=4, W=$W), median of $N reps:")

    t_score = med(() -> begin
        cc = [γo.wHat[o] * AodPow[o, d] * γo.τ[o, d] for o in 1:D, d in 1:D]
        ccσ = [wPow[o] * (AodPow[o, d] * γo.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
        cc[1] + ccσ[1]
    end, N)
    UPow = U .^ (-μ); UσPow = Uσ .^ (-μ)
    t_pow = med(() -> (U .^ (-μ), Uσ .^ (-μ)), N)
    t_winner = med(() -> begin
        acc = 0
        @inbounds for d in 1:D, s in 1:W
            best = constCons[1, d] / UPow[s, 1]; bo = 1
            for o in 2:D
                p = constCons[o, d] / UPow[s, o]; (p < best) && (best = p; bo = o)
            end
            acc += bo
        end
        acc
    end, N)
    t_alloc = med(() -> zeros(W, ctx.obj.d), N)
    @printf("  score (constCons/constConsσ, O(D^2))        : %8.5f ms\n", t_score * 1000)
    @printf("  U^{-mu}/Uσ^{-mu} power transform (O(W·D))    : %8.5f ms\n", t_pow * 1000)
    @printf("  hard winner search over D origins (O(W·D^2)) : %8.5f ms\n", t_winner * 1000)
    @printf("  allocate+zero dense W×d output matrix        : %8.5f ms\n", t_alloc * 1000)
    println("  (dense also: CES value for ALL D origins/dest + write D cols/dest + centering,")
    println("   all fused in hFunction!'s loop; compressed evaluates the CES value for the")
    println("   WINNER ONLY and never writes/allocs the dense matrix.)")
    return nothing
end

println("="^92)
println("DENSE vs COMPRESSED benchmark  (NTHREADS=$NTHREADS)")
println("="^92)
dense_component_profile()
println("-"^92)
for (D, W) in [(4, 8000), (4, 80000), (6, 8000), (8, 8000), (10, 8000)]
    try
        bench_row(D, W)
    catch e
        @printf("D=%-2d W=%-6d | FAILED: %s\n", D, W, sprint(showerror, e))
    end
end
println("="^92)
