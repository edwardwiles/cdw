# diag/compressed-hessian-operator-audit-2026-07-25, Phase 5 validation at
# canonical production scale: D=20 real data, destination_sample=:exclude_row,
# W=80,000, pseudorandom seed 20260719 (task brief's canonical benchmark
# protocol config). Same comparison as validate_winner_pair_hessian_d4.jl,
# reused verbatim, just built on d20_real_setup_design instead of d4_exact_setup.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "winner_pair_hessian.jl"))
using Printf, LinearAlgebra, Random

function unpack_packed(h::AbstractVector, n::Int)
    Mx = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mx[i, j] = h[k]; Mx[j, i] = h[k]
        k += 1
    end
    return Mx
end

W = 80_000
println("Building D=20 real context: destination_sample=:exclude_row, W=$W, seed=20260719 ..."); flush(stdout)
t_ctx = @elapsed ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_seed = 20260719, destination_sample = :exclude_row)
@printf("context built in %.1fs\n", t_ctx); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
obj = ctx.obj
n = obj.outer_constr_index
println("n (outer_constr_index) = $n, D=$(ctx.D), D_dest=$(ctx.D_dest), W=$(size(obj.U,1))"); flush(stdout)

t_cf = @elapsed cf = build_compressed_factual(θ_full, ctx; check_ties = true)
@printf("CompressedFactual built in %.2fs: oci=%d, cf_col=%d\n", t_cf, cf.oci, cf.cf_col); flush(stdout)

function compare_at(x::AbstractVector, st, wctx; label = "")
    ζ = x[1]; λ = @view x[2:end]
    f, g_ζ, g_λ, q, _ = compressed_cc_value_grad(ζ, λ, st.cf; Psi! = st.obj.Psi!, dPsi! = st.obj.dPsi!)
    st.obj.arg0 .= q
    if !st.dense_materialized
        ncolI = st.cf.oci - 1
        materialize_dense_factual_structured!(@view(st.obj.H[:, 3:2+ncolI]), st.cf)
        st.dense_materialized = true
    end
    n_ = st.obj.outer_constr_index
    h_dense = Vector{Float64}(undef, n_*(n_+1)÷2)
    CS.hessian!(h_dense, st.obj)
    H_dense = unpack_packed(h_dense, n_)

    h_wp = Vector{Float64}(undef, n_*(n_+1)÷2)
    winner_pair_hessian!(h_wp, st.obj, wctx)
    H_wp = unpack_packed(h_wp, n_)

    errabs = maximum(abs.(H_wp .- H_dense))
    relerr = errabs / max(1.0, maximum(abs.(H_dense)))
    symerr = maximum(abs.(H_wp .- H_wp'))
    @printf("  %-28s |H_wp - H_dense|_inf = %.3e (rel %.3e)   symmetry resid = %.3e\n", label, errabs, relerr, symerr)
    return (label = label, errabs = errabs, relerr = relerr, symerr = symerr)
end

t_wctx = @elapsed wctx = build_winner_pair_ctx(cf)
@printf("winner-pair ctx built in %.2fs\n", t_wctx); flush(stdout)

results = NamedTuple[]
println("\n=== D=20 real (W=80,000, seed=20260719, :exclude_row): winner_pair_hessian! vs production hessian! ===")
st0 = CompressedCBState(obj, cf, 0.0, false)
push!(results, compare_at(zeros(n), st0, wctx; label = "x=0"))

Random.seed!(20260725)
for i in 1:3
    x = 0.02 .* randn(n)
    st = CompressedCBState(obj, cf, 0.0, false)
    push!(results, compare_at(x, st, wctx; label = "random x[$i] (0.02 scale)"))
end

println("\n--- real solved (KNITRO) point at calibration ---")
t_solve = @elapsed (nStatus, objSol, xsol, lambda_, n_fg, n_hess) = inner_loop_KNITRO_compressed(obj, CompressedCBState(obj, cf, 0.0, false))
@printf("inner_loop_KNITRO_compressed: nStatus=%d, objSol=%.6e, n_fg=%d, n_hess=%d, wall=%.2fs\n", nStatus, objSol, n_fg, n_hess, t_solve)
flush(stdout)
if nStatus in (0, -100, -101, -103)
    st2 = CompressedCBState(obj, cf, 0.0, false)
    push!(results, compare_at(collect(xsol), st2, wctx; label = "real solved x*"))
else
    println("WARNING: inner solve infeasible (nStatus=$nStatus) at calibration point -- skipping solved-point check")
end

maxerr = maximum(r.relerr for r in results)
@printf("\nOverall max relative error across %d points: %.3e\n", length(results), maxerr)
println(maxerr < 1e-8 ? "PASS (machine/near-machine-precision agreement at D=20 scale)" : "FAIL (investigate before proceeding)")
flush(stdout)
