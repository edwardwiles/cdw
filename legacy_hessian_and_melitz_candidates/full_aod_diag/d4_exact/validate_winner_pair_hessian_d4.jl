# diag/compressed-hessian-operator-audit-2026-07-25, Phase 5 validation:
# numerical agreement between winner_pair_hessian! and the trusted production
# `hessian!` (PsiObjectiveBundleImplicitMethodB_fullA.jl), both invoked through
# the EXACT same production compressed-callback wiring
# (compressed_live.jl::_callbackEvalFG_inner_compressed!/_callbackEvalH_inner_compressed!),
# at D=4, unrestricted (no CM/ZC), multiple random dual points + the real
# solved (cold-verified) point. No KNITRO solve needed beyond the wrapper's
# own inner solve to get a real solved x -- correctness check is a pure
# fixed-point comparison at that x plus several random perturbations.
include(joinpath(@__DIR__, "context.jl"))
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

"Unpack a packed row-major upper-triangular Hessian vector into a dense symmetric n x n matrix (same convention as c13_validate_hessian_archs.jl)."
function unpack_packed(h::AbstractVector, n::Int)
    Mx = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mx[i, j] = h[k]; Mx[j, i] = h[k]
        k += 1
    end
    return Mx
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
obj = ctx.obj
n = obj.outer_constr_index
println("D=4 unrestricted context: n (outer_constr_index) = $n, W = $(size(obj.U,1))")
flush(stdout)

cf = build_compressed_factual(θ_full, ctx; check_ties = true)
println("CompressedFactual built: oci=$(cf.oci), D=$(cf.D), D_dest=$(cf.D_dest), cf_col=$(cf.cf_col)")
flush(stdout)

# ---- reference dense G (compressed-derived fill, identical to what the live Hessian callback uses) ----
# NOTE: grav_raw/fill_gravity_column! deliberately skipped here -- that writes obj.H[:, 2+obj.d],
# the OUTER gravity-constraint column, strictly outside the H[:, 2:1+outer_constr_index] range
# hessian!/hessian_cm_structured!/winner_pair_hessian! ever read (confirmed:
# PsiObjectiveBundleImplicitMethodB_fullA.jl:185-190's H_copy slice is 2:1+outer_constr_index,
# and cf.oci == obj.outer_constr_index by construction in build_compressed_factual) -- irrelevant
# to this Hessian-correctness check. (In passing: compressed_gravity_raw at compressed_live.jl:95
# calls ctx.D_dest directly with no hasproperty guard, unlike every other D_dest access in this
# codebase -- this D=4 d4_exact_setup context has no D_dest field at all, since it's a
# legacy/square D=D_dest context, so that call throws. Real latent gap, out of scope for this
# Hessian audit; not touched.)
grav_raw = 0.0
st = CompressedCBState(obj, cf, grav_raw, false)

"Set obj.arg0 = q at dual point x=(zeta,lambda) via the SAME production FG callback compressed_cc_value_grad uses, then run BOTH Hessian implementations and compare."
function compare_at(x::AbstractVector, st, wctx; label = "")
    ζ = x[1]; λ = @view x[2:end]
    f, g_ζ, g_λ, q, _ = compressed_cc_value_grad(ζ, λ, st.cf; Psi! = st.obj.Psi!, dPsi! = st.obj.dPsi!)
    st.obj.arg0 .= q

    if !st.dense_materialized
        ncolI = st.cf.oci - 1
        materialize_dense_factual_structured!(@view(st.obj.H[:, 3:2+ncolI]), st.cf)
        fill_gravity_column!(st.obj, st.grav_raw)
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

wctx = build_winner_pair_ctx(cf)

results = NamedTuple[]
println("\n=== D=4 unrestricted: winner_pair_hessian! vs production hessian! ===")

# Point 1: zeta=0, lambda=0
push!(results, compare_at(zeros(n), st, wctx; label = "x=0"))

Random.seed!(20260725)
for i in 1:5
    x = 0.05 .* randn(n)
    push!(results, compare_at(x, st, wctx; label = "random x[$i] (0.05 scale)"))
end
for i in 1:3
    x = 0.5 .* randn(n)
    push!(results, compare_at(x, st, wctx; label = "random x[$i] (0.5 scale, larger)"))
end

# Real solved point via the actual production compressed inner solve
println("\n--- real solved (KNITRO) point ---")
nStatus, objSol, xsol, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed(obj, CompressedCBState(obj, cf, grav_raw, false))
println("inner_loop_KNITRO_compressed: nStatus=$nStatus, objSol=$objSol")
flush(stdout)
if nStatus in (0, -100, -101, -103)
    st2 = CompressedCBState(obj, cf, grav_raw, false)
    push!(results, compare_at(collect(xsol), st2, wctx; label = "real solved x*"))
else
    println("WARNING: inner solve did not report a feasible status ($nStatus) -- skipping solved-point check")
end

maxerr = maximum(r.relerr for r in results)
@printf("\nOverall max relative error across %d points: %.3e\n", length(results), maxerr)
println(maxerr < 1e-9 ? "PASS (machine-precision agreement)" : "FAIL (investigate before proceeding)")
flush(stdout)
