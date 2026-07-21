# Continuation 12, section 12: D=4 fixed-parameter validation battery for the dense CM reference.
# Points: calibration, the registered upper_maxit40 headline candidate (kappa~0.1725, unrestricted
# optimum -- CM restriction may make this infeasible, that's an expected/interesting finding per
# the brief, not a bug), a mildly perturbed feasible point, and a deliberately structurally
# infeasible point (one A_od entry driven to ~0, making that origin unable to ever win a
# destination on this draw support -- should fail the inner solve, nStatus != 0/-100/-101/-103).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
using Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# ---- candidate points. NOTE: ctx.free_idx's RAW layout is [gamma'_focal; vec(A_od) level, not
# log] (length 1+D^2=17, confirmed by d4_exact_setup). This is DIFFERENT from candidate_registry's
# pivot-reduced-log `w` layout (length 1+(D^2-1)=16, `x_free_from_w` exponentiates); do not mix
# the two coordinate systems in one vector, as an earlier version of this script did (caught via
# a DomainError from an un-exponentiated, wrong-length vector reaching evaluate_fullA). Use the
# RAW layout throughout except for w_up40, which only exists as a pivot-w literal.
x_calib_raw = ctx.θ0_up[ctx.free_idx]   # gamma'_focal, then all-ones A_od

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
x_up40 = x_free_from_w(w_up40)   # this one legitimately needs pivot-expansion; different free-param map, still a valid ctx.obj input (same underlying theta reconstruction target)

# perturbed feasible: small multiplicative jitter around calibration's A_od entries, gamma'
# untouched (keeps it well inside its box bounds); modest so it stays economically sensible
Random.seed!(4243)
x_perturbed = copy(x_calib_raw)
x_perturbed[2:end] .*= exp.(0.05 .* randn(length(x_perturbed) - 1))

# structurally infeasible: drive one A_od theta entry (natural-theta, level not log) toward 0 so
# that origin can essentially never win that destination cell under exact hard-max
x_infeasible_raw = copy(x_calib_raw)
raw_Aod_pos_11 = findfirst(==(ctx.Aod_offset + 1 + (1 - 1) * D), ctx.free_idx)  # A_od[1,1] free-index position
x_infeasible_raw[raw_Aod_pos_11] = 1e-6

function run_point(label, x_free_for, ctx_use; is_raw = false)
    r = evaluate_fullA(x_free_for, ctx_use; use_cache = false, warm = false)
    ok = r.inner_status in (0, -100, -101, -103)
    @printf("%-30s nStatus=%-5d ok=%-5s", label, r.inner_status, ok)
    if ok
        @printf("  Delta_dual=%.6f  gamma'=%.6f  max_abs_moment_kkt_resid=%.2e\n", r.Delta_dual, r.gamma_focal_prime, r.max_abs_moment_kkt_resid)
    else
        println()
    end
    return (label = label, ok = ok, nStatus = r.inner_status, r = r)
end

for L in (10, 20, 50)
    println("\n--- L=$L (anchored) ---")
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    run_point("calibration", x_calib_raw, ctx_cm)
    run_point("upper_maxit40 (unrestricted opt)", x_up40, ctx_cm)
    run_point("perturbed_feasible", x_perturbed, ctx_cm)

    # infeasible point uses the RAW (non-pivot) free-param layout -- build a matching CM-augmented
    # obj off the SAME ctx (pivot elimination doesn't change ctx.obj/ctx.free_idx, only how outer
    # scripts parameterize the search space), so ctx_cm is reusable unchanged.
    run_point("structurally_infeasible (A[1,1]~=1e-6)", x_infeasible_raw, ctx_cm)
end
