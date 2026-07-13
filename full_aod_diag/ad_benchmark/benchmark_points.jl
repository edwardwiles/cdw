# ============================================================================
# §1: freeze the D=4 benchmark points, for the γ_d≡1 + direct-γ' variant
# (compare_directgp.jl), the best-conditioned configuration found so far.
#
# ADAPTATION NOTE (documented, not hidden): the original spec's benchmark
# points (A: init, B: "reported stalled full lower-bound point", C: "superior
# reduced lower-bound point... through the full callback", D: full
# upper-bound point) describe the OLD A[1,d]=1-normalized full-A_od diagnostic
# (full_aod_diag/report.md), whose defining pathology (full lower bound
# stalling badly below a reduced/restricted alternative) this session's
# γ_d≡1+direct-γ' change has now LARGELY FIXED (opt_err 0.598→0.0038 lower,
# 1.51→0.0087 upper; see out/compare_directgp_summary.txt). There is no
# "stalled vs superior-reduced" pair to reuse under the new, better-behaved
# variant. We therefore substitute four real, already-computed points from
# TODAY's runs that still span meaningfully different regions of θ-space:
#   A = θ_initial (compensating-scale start point, build_theta_gammanorm)
#   B = θ_lo from compare_directgp.jl (near-lower-κ, direct-γ' objective; the
#       closest analogue to "the reported lower-bound point", but well-
#       converged here, not stalled — opt_err 0.0038)
#   C = θ_lo from compare_gammanorm.jl's κ-objective run (out/compare_gammad_norm.jld2),
#       evaluated through the direct-γ' callback. Same θ-coordinate layout
#       (verified in the driver scripts), genuinely different point (different
#       objective drove the search there) — serves the spec's intent of "a
#       point reached under different search dynamics, evaluated through the
#       SAME target callback" without a fabricated "reduced-variant" analogue.
#   D = θ_up from compare_directgp.jl (near-upper-κ, direct-γ' objective)
# ============================================================================
include("setup_context.jl")

so, pp = build_ad_context()
@unpack θ_initial, U, γ, outer_constr_index, nTotalMoments = pp
D = so.D; bi = AD_PARAMS.baseIndex; σ = AD_PARAMS.σHat; μHat = γ.μHat

θ_A = build_theta_gammanorm(θ_initial, D, bi, μHat, σ)

directgp = JLD2.load(joinpath(@__DIR__, "..", "out", "compare_directgp.jld2"))
θ_B = directgp["θ_lo"]
θ_D = directgp["θ_up"]

gammad = JLD2.load(joinpath(@__DIR__, "..", "out", "compare_gammad_norm.jld2"))
θ_C = gammad["θ_lo"]   # cross-point: reached under the κ-objective search, evaluated via direct-γ' callback

println("θ_A[7] (γ'_focal) = ", θ_A[7])
println("θ_B[7] = ", θ_B[7], "  (from compare_directgp lower)")
println("θ_C[7] = ", θ_C[7], "  (from compare_gammad_norm lower, cross-evaluated)")
println("θ_D[7] = ", θ_D[7], "  (from compare_directgp upper)")

"""
    solve_inner_and_fix(obj, θ)

Run the REAL inner min-divergence solve at θ (KNITRO), then extract the
envelope-theorem "fixed context" (λ, arg1 — the per-draw Ψ-derivative weight)
exactly as the production callback would hold them fixed while differentiating
w.r.t. θ. Returns (λ_fixed, arg1_fixed, x, nStatus, objSol).
"""
function solve_inner_and_fix(obj, θ)
    objSol, x, nStatus = CS.inner_loop_internal(obj, θ)
    # populate obj.arg0/obj.arg1 at this (x, θ) exactly as the production
    # gradient callback does (dPsi! is triggered whenever constr is non-empty)
    obj(x, Float64[], Float64[]; constr = zeros(obj.d - obj.outer_constr_index + 2))
    λ_fixed = copy(x[2:end])
    arg1_fixed = copy(obj.arg1)
    return λ_fixed, arg1_fixed, copy(x), nStatus, objSol
end

points = Dict{Symbol,Any}()
for (name, θpt) in ((:A, θ_A), (:B, θ_B), (:C, θ_C), (:D, θ_D))
    obj = make_ad_obj(pp, so)
    λ_fixed, arg1_fixed, x, nStatus, objSol = solve_inner_and_fix(obj, θpt)
    points[name] = (θ=θpt, λ=λ_fixed, arg1=arg1_fixed, x=x, nStatus=nStatus, objSol=objSol)
    println(">>> point $name: inner nStatus=$nStatus  objSol=$objSol  |λ|=$(length(λ_fixed))")
end

@save joinpath(@__DIR__, "benchmark_points.jld2") points θ_initial D bi σ μHat outer_constr_index nTotalMoments
println("Saved benchmark_points.jld2 with points A,B,C,D")
