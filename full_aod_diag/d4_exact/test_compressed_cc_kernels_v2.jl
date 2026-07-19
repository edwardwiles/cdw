# ============================================================================
# Continuation 9, Phase 4: equivalence test for the destination-major _v2
# kernels (compressed_cc_kernels_v2.jl) against the originals
# (compressed_moments.jl::compressed_dual_contraction,
# compressed_cc_inner.jl::compressed_transpose_contraction/
# compressed_cc_value_grad/compressed_cc_hvp). Per this file's own header
# argument, the reordering should be BIT-IDENTICAL (0.0 diff), not merely
# close -- checked directly, not assumed.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_cc_kernels_v2.jl"))
using Random, Printf, LinearAlgebra

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; oci = ctx.obj.outer_constr_index; ncol = oci - 1
W = size(ctx.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

xf0 = ctx.θ0_up[ctx.free_idx]
base = solve_base_state(xf0, ctx)
cf = build_compressed_factual(base.θ_full0, ctx)

Random.seed!(1234)
n_pass = 0; n_total = 0
for trial in 1:10
    global n_pass, n_total
    β = randn(ncol)
    weights = randn(W)

    t_orig = compressed_dual_contraction(β, cf)
    t_v2 = compressed_dual_contraction_v2(β, cf)
    d1 = maximum(abs.(t_orig .- t_v2))

    v_orig = compressed_transpose_contraction(weights, cf)
    v_v2 = compressed_transpose_contraction_v2(weights, cf)
    d2 = maximum(abs.(v_orig .- v_v2))

    ok = (d1 == 0.0) && (d2 == 0.0)
    n_total += 1; ok && (n_pass += 1)
    @printf("trial %2d: dual_contraction maxdiff=%.3e  transpose_contraction maxdiff=%.3e  %s\n",
            trial, d1, d2, ok ? "BIT-IDENTICAL" : "DIFFERS")
end

# ---- full compressed_cc_value_grad_v2 / compressed_cc_hvp_v2 vs originals, at the base dual point ----
ζ = base.ζstar; λ = collect(base.λstar)
f0, gz0, gl0, q0, dPsq0 = compressed_cc_value_grad(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
f2, gz2, gl2, q2, dPsq2 = compressed_cc_value_grad_v2(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
d_f = abs(f0 - f2); d_gz = abs(gz0 - gz2); d_gl = maximum(abs.(gl0 .- gl2)); d_q = maximum(abs.(q0 .- q2))
ok_vg = (d_f == 0.0) && (d_gz == 0.0) && (d_gl == 0.0) && (d_q == 0.0)
@printf("value_grad_v2: |df|=%.3e |dgz|=%.3e max|dgl|=%.3e max|dq|=%.3e  %s\n",
        d_f, d_gz, d_gl, d_q, ok_vg ? "BIT-IDENTICAL" : "DIFFERS")
n_total += 1; ok_vg && (n_pass += 1)

Random.seed!(5678)
for trial in 1:5
    global n_pass, n_total
    p = randn(1 + ncol); p ./= norm(p)
    Hz0, Hl0 = compressed_cc_hvp(q0, p[1], p[2:end], cf; ddPsi! = ctx.obj.ddPsi!)
    Hz2, Hl2 = compressed_cc_hvp_v2(q0, p[1], p[2:end], cf; ddPsi! = ctx.obj.ddPsi!)
    dz = abs(Hz0 - Hz2); dl = maximum(abs.(Hl0 .- Hl2))
    ok = (dz == 0.0) && (dl == 0.0)
    n_total += 1; ok && (n_pass += 1)
    @printf("hvp trial %d: |dHz|=%.3e max|dHl|=%.3e  %s\n", trial, dz, dl, ok ? "BIT-IDENTICAL" : "DIFFERS")
end

println("\n", n_pass, "/", n_total, " checks BIT-IDENTICAL")
