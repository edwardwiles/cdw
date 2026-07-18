# ============================================================================
# Phase 6: blockwise gradient re-check. The prior continuation's Phase D used
# FULL-VECTOR cosine similarity, which the task correctly flags as
# concealing large A-block errors: x_free = (gamma_focal_prime, A_od...) and
# the gamma component dominates the gradient's norm (see Phase D's own
# norm_grad column and the gravity_focal_prime-only sensitivity noted in
# sequential_methodology.tex sec 8: "the constraint-gradient magnitude for
# gamma'_focal [is] many orders of magnitude larger than for the A block").
# This script decomposes EVERY comparison into a gamma-component-only part
# and an A-block-only part (both raw and gravity-tangent-projected), so an
# artificially good full-vector cosine cannot hide a bad A-block.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "derivative_methods.jl"))
using Random, LinearAlgebra, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phase6_blockwise_gradient")
mkpath(OUTDIR)
const H = 0.01

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
w_to_xfree(w) = (z = pivot_expand(w[2:end], pe); vcat(w[1], vec(exp.(z))))

x0_calib = CS.pack_free(ctx.θ0_up, ctx.m)
w_maxit15 = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069,
    0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011,
    1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663,
    0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]
w_maxit40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
    0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
    1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
    0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]

points = [("calibration", x0_calib), ("upper_maxit15", w_to_xfree(w_maxit15)), ("upper_maxit40", w_to_xfree(w_maxit40))]

# x_free layout (17-dim, per FreeParamMap.free_idx in context.jl): [1]=gamma_focal_prime, [2:17]=A_od
# (column-major D x D, matching ctx.free_idx = vcat(3+D, Aod_offset+1:Aod_offset+D^2))
const GAMMA_IDX = 1
const A_IDX = 2:(1+D2)

# gravity-tangent projection: any direction confined to the A-block that keeps gravity satisfied
# exactly is, in log(A_od) space, any vector orthogonal to the gravity coefficient c (see
# gravity_elimination.jl). We build the SAME orthonormal nullspace basis used there, but expressed
# directly in the RAW (non-log) A_od coordinate's local tangent (d/d(logA) = A .* d/dA at the base
# point) so it can be applied to gradients taken directly in x_free (not the pivot-reduced w) space.
function gravity_tangent_basis(ctx, x_free_A::AbstractVector)
    c = vec(gravity_linear_coeffs(ctx))              # D^2, coefficients in log(A) space
    Z = nullspace(reshape(c, 1, D2))                  # D^2 x (D^2-1), orthonormal in log(A) space
    # chain rule: a unit step in RAW A-space direction u corresponds to a log(A)-space step
    # u ./ A_od (elementwise) -- so the tangent-consistent RAW-space basis is diag(A_od) * Z,
    # re-orthonormalized (QR) since diag(A_od)*Z is not orthonormal in general.
    raw_basis = x_free_A .* Z            # D^2 x (D^2-1), broadcasting A_od down each column
    Q = Matrix(qr(raw_basis).Q)
    return Q[:, 1:size(Z,2)]              # D^2 x (D^2-1) orthonormal basis, RAW A-space tangent directions
end

function central_fd_grad(f, x0::Vector{Float64}, h::Float64)
    n = length(x0); g = zeros(n)
    for i in 1:n
        xp = copy(x0); xp[i] += h; xm = copy(x0); xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

cossim(a, b) = dot(a, b) / max(norm(a) * norm(b), 1e-300)

rows = NamedTuple[]
for (label, x_free) in points
    println("="^78); println("POINT: $label"); println("="^78); flush(stdout)
    base = solve_base_state(x_free, ctx)
    x_free_A = x_free[A_IDX]
    Ttan = gravity_tangent_basis(ctx, x_free_A)   # D^2 x (D^2-1)

    g_A_pathwise = method_A_pathwise_ad(x_free, ctx, base)
    g_Qadj = central_fd_grad(x -> frozen_adjoint_Q(x, ctx, base), x_free, H)
    g_Lfix = central_fd_grad(x -> fixed_dual_L(x, ctx, base), x_free, H)
    g_Delta = central_fd_grad(x -> optimized_Delta(x, ctx; warm=true), x_free, H)

    # blockwise-rescaled L_fix: separately estimate a gamma-scale and an A-scale factor by
    # regressing L_fix's block magnitude against Delta_FD's block magnitude (least-squares scalar
    # per block, not one global scalar -- exactly what the task's Phase 6 instructions require)
    gamma_scale = g_Delta[GAMMA_IDX] / max(abs(g_Lfix[GAMMA_IDX]), 1e-300) * sign(g_Lfix[GAMMA_IDX]) * sign(g_Delta[GAMMA_IDX])
    a_scale = dot(g_Lfix[A_IDX], g_Delta[A_IDX]) / max(dot(g_Lfix[A_IDX], g_Lfix[A_IDX]), 1e-300)
    g_Lfix_rescaled = copy(g_Lfix)
    g_Lfix_rescaled[GAMMA_IDX] *= abs(g_Delta[GAMMA_IDX] / max(abs(g_Lfix[GAMMA_IDX]), 1e-300))
    g_Lfix_rescaled[A_IDX] .*= a_scale

    for (mname, g) in (("A_pathwise_AD", g_A_pathwise), ("Q_adj_FD", g_Qadj), ("L_fix_FD", g_Lfix), ("L_fix_FD_blockrescaled", g_Lfix_rescaled))
        full_cos = cossim(g, g_Delta)
        gamma_err = g[GAMMA_IDX] - g_Delta[GAMMA_IDX]
        gamma_rel_err = abs(gamma_err) / max(abs(g_Delta[GAMMA_IDX]), 1e-300)
        A_cos = cossim(g[A_IDX], g_Delta[A_IDX])
        A_norm_ratio = norm(g[A_IDX]) / max(norm(g_Delta[A_IDX]), 1e-300)
        A_sign_agree = count(sign.(g[A_IDX]) .== sign.(g_Delta[A_IDX])) / D2
        # project both gradients' A-blocks onto the gravity-tangent basis
        gt_g = Ttan' * g[A_IDX]; gt_Delta = Ttan' * g_Delta[A_IDX]
        gt_cos = cossim(gt_g, gt_Delta)
        gt_norm_ratio = norm(gt_g) / max(norm(gt_Delta), 1e-300)
        push!(rows, (point = label, method = mname,
            full_vector_cosine = full_cos,
            gamma_component_abs_err = abs(gamma_err), gamma_component_rel_err = gamma_rel_err,
            A_block_cosine = A_cos, A_block_norm_ratio = A_norm_ratio, A_block_sign_agreement = A_sign_agree,
            gravity_tangent_A_cosine = gt_cos, gravity_tangent_A_norm_ratio = gt_norm_ratio))
        @printf("  [%-24s] full_cos=%.4f | gamma_relerr=%.3f | A_cos=%.4f A_normratio=%.3f A_signagree=%.3f | gravtan_A_cos=%.4f gravtan_normratio=%.3f\n",
            mname, full_cos, gamma_rel_err, A_cos, A_norm_ratio, A_sign_agree, gt_cos, gt_norm_ratio)
    end
    flush(stdout)
end

write_csv_rows = function(path, rows)
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((r[c] for c in cols), ","))
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "phase6_blockwise_gradient.csv"), rows)
println("\nWrote ", joinpath(OUTDIR, "phase6_blockwise_gradient.csv"))
