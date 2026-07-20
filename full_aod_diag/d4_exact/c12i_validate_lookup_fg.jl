# Part B correctness validation: the lookup-based FG evaluator (cm_lookup_kernels.jl,
# CMLookupState) must reproduce the DENSE `obj(x,g)` callable's (f,g) EXACTLY (to floating-point
# accumulation-order tolerance, not "close") at a battery of (ζ,λ) points -- not just at the
# converged optimum (where errors could cancel), but along an actual KNITRO inner-solve
# trajectory AND at random perturbations, for both :interval and :suffix (cumulative diagnostic)
# lookup methods and both :anchored / :orthonormal contrasts, at L in {10,20,50}.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
using Printf, Random, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]

# ---- trajectory capture: instrument obj(x,g) calls during a real dense inner solve at the
# calibration point, by wrapping obj.moments!  -- simplest robust way to capture the ACTUAL
# sequence of x's KNITRO visited is to monkey-patch the FG callback is not available without
# touching cc_algo, so instead we just build a battery of x points: the converged optimum,
# several random points scaled like the optimum (since KNITRO's own inner iterates live in a
# similar magnitude range), and zero. This exercises the lookup kernels away from the optimum
# (where cancellation could hide a bug), not just at a single converged point.
function random_x_battery(n_var::Int; scale = 1.0, n = 8, seed = 7171)
    rng = MersenneTwister(seed)
    return [scale .* randn(rng, n_var) for _ in 1:n]
end

function compare_fg(dense_obj, st::CMLookupState, x::Vector{Float64}; label = "")
    n_var = length(x)
    gd = zeros(n_var); gl = zeros(n_var)
    fd = dense_obj(x, gd)
    fl = st(x, gl)
    ferr = abs(fd - fl)
    frel = ferr / max(1.0, abs(fd))
    gerr = maximum(abs.(gd .- gl))
    grel = gerr / max(1.0, maximum(abs.(gd)))
    return (label = label, fd = fd, fl = fl, ferr = ferr, frel = frel, gerr = gerr, grel = grel, n_var = n_var)
end

all_ok = Ref(true)
for L in (10, 20, 50), contrasts in (:anchored, :orthonormal), method in (:interval, :suffix)
    # method=:interval must be compared against the INTERVAL-basis dense reference (its stored
    # lambda ARE interval-basis coefficients); method=:suffix (cumulative-basis diagnostic) must
    # be compared against the CUMULATIVE-basis dense reference (common_marginals_moments.jl's
    # own build_cm_augmented_obj) -- comparing suffix against the interval-augmented obj is a
    # category error (different bases -> different lambda meaning), not a bug in the kernel; an
    # earlier version of this script made exactly that mistake for :suffix and is fixed here.
    aug_int = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = contrasts)
    aug = method == :interval ? aug_int : build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))

    # populate obj.H (dense) at the calibration theta -- required both for the dense baseline AND
    # so CMLookupState's core-column BLAS slice (obj.H[:,2:2+ncore1]) is populated identically.
    θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
    W = size(ctx.U, 1)
    aug.obj_cm.moments!(@view(aug.obj_cm.H[:, 1]), CS.select_G_from_H(aug.obj_cm, aug.obj_cm.H), θ_full, ctx.U, aug.obj_cm)
    aug.obj_cm.H[:, 2] .= 1.0

    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    st = CMLookupState(aug.obj_cm, aug.ncore, aug.ncm, L, aug.origins, aug.refIndex1, aug_int.bins, R;
                        method = method, nthreads_use = 1)

    n_var = aug.obj_cm.outer_constr_index   # = number of inner variables (ζ + all λ)
    battery = vcat([zeros(n_var)], random_x_battery(n_var; scale = 0.02, n = 5),
                    random_x_battery(n_var; scale = 0.2, n = 3))

    # Also include the ACTUAL converged inner solution (a realistic, non-adversarial point).
    r = evaluate_fullA(x_free_calib, ctx_cm; use_cache = false, warm = false)
    if r.inner_status in (0, -100, -101, -103)
        push!(battery, vcat(r.zeta, r.lambda))
    end

    worst = (ferr = 0.0, grel = 0.0)
    for (i, x) in enumerate(battery)
        c = compare_fg(aug.obj_cm, st, x; label = "pt$i")
        worst = (ferr = max(worst.ferr, c.ferr), grel = max(worst.grel, c.grel))
        if c.ferr > 1e-9 || c.grel > 1e-9
            @printf("  MISMATCH L=%d %-10s %-8s %-4s  f_dense=%.10f f_lookup=%.10f ferr=%.3e  max|g_dense-g_lookup|/scale=%.3e\n",
                    L, string(contrasts), string(method), c.label, c.fd, c.fl, c.ferr, c.grel)
            all_ok[] = false
        end
    end
    @printf("[L=%2d %-10s %-8s] n_var=%3d  battery=%2d pts  worst ferr=%.3e  worst grel=%.3e\n",
            L, string(contrasts), string(method), n_var, length(battery), worst.ferr, worst.grel)
end

println(all_ok[] ? "\nALL FG COMPARISONS PASS (tol 1e-9 abs f / rel g)." : "\nFAILURES ABOVE.")
