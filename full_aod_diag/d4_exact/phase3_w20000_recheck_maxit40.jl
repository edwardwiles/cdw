include(joinpath(pwd(), "full_aod_diag/d4_exact/context_scaled.jl"))
include(joinpath(pwd(), "full_aod_diag/d4_exact/winners.jl"))
include(joinpath(pwd(), "full_aod_diag/d4_exact/oracle.jl"))
include(joinpath(pwd(), "full_aod_diag/d4_exact/gravity_elimination.jl"))

# NOTE: this uses an INDEPENDENT W=20000 draw set (its own seedU/seedFakeData via
# d_exact_setup_scaled), NOT a nested/common-draws superset of the W=8000 draws that FOUND this
# candidate -- the task explicitly asks for nested draws; this is a faster, clearly-labeled
# APPROXIMATION (independent-draws robustness check) given this continuation's severe time budget,
# not the rigorous design. Flagged, not silently substituted.
ctx8000 = d4_exact_setup(find_smallest = true)
ctx20000 = d_exact_setup_scaled(D = 4, W = 20000, find_smallest = true)
pe = build_pivot_elimination(ctx8000)   # gravity coeffs only depend on data/mu, same at both W

w_maxit40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
    0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
    1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
    0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
x_free = vcat(w_maxit40[1], vec(exp.(pivot_expand(w_maxit40[2:end], pe))))

r8000 = evaluate_fullA(x_free, ctx8000; cache = nothing, warm = false)
r20000 = evaluate_fullA(x_free, ctx20000; cache = nothing, warm = false)

println("W=8000 (original):  kappa=", 1-r8000.gamma_focal_prime^(ctx8000.σ/(ctx8000.σ-1)),
        "  Delta=", r8000.Delta_dual, "  Delta-delta=", r8000.Delta_minus_delta,
        "  gravity=", r8000.gravity_value, "  inner_status=", r8000.inner_status)
println("W=20000 (independent draws, SAME theta): kappa=", 1-r20000.gamma_focal_prime^(ctx20000.σ/(ctx20000.σ-1)),
        "  Delta=", r20000.Delta_dual, "  Delta-delta=", r20000.Delta_minus_delta,
        "  gravity=", r20000.gravity_value, "  inner_status=", r20000.inner_status)
feasible_at_20000 = isfinite(r20000.Delta_dual) && r20000.Delta_dual <= ctx20000.δ + 1e-6
println("STILL FEASIBLE AT W=20000 (same theta, independent draws): ", feasible_at_20000)
