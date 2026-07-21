# Continuation 13 addendum: validate interval-NATIVE Architecture C
# (cm_hessian_architecture_interval.jl) against the trusted DENSE INTERVAL-BASIS reference
# (build_cm_augmented_obj_interval's Architecture A -- generic dense hessian!), NOT the
# cumulative-basis Hessian. D=4, L in {10,20,50}, calibration + unrestricted candidate +
# 2 perturbed/difficult points, both contrasts.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
using Printf, LinearAlgebra, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

const w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
                0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
                1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
                0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
pe = build_pivot_elimination(ctx)
x_free_upper = vcat(w_up40[1], vec(exp.(pivot_expand(w_up40[2:end], pe))))
θ_full_upper = CS.reconstruct_full(x_free_upper, ctx.m)

"Unpack packed upper-triangular Hessian into dense symmetric n x n."
function unpack_packed(h::AbstractVector, n::Int)
    M = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        M[i, j] = h[k]; M[j, i] = h[k]
        k += 1
    end
    return M
end

function validate_at(L::Int, contrasts::Symbol, θ_full::Vector{Float64}, label::String; seed::Int)
    println("-"^100)
    println("L=$L contrasts=$contrasts point=$label")
    println("-"^100)

    # ---- dense INTERVAL reference (Architecture A on the interval basis) ----
    augI = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = contrasts)
    objA = augI.obj_cm
    n = objA.outer_constr_index
    K = zeros(size(ctx.U, 1))
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
    objA.H[:, 1] .= K
    objA.H[:, 2] .= 1.0

    # ---- interval-native Architecture C context ----
    cctxI = build_cm_bin_ctx_interval(ctx, augI)

    # ---- cumulative Architecture C, for headline comparison (built off the CUMULATIVE dense obj,
    # not the interval one -- a genuinely different objective object, evaluated separately below) ----
    augC = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    objC = augC.obj_cm
    objC.H[:, 1] .= K   # K (the moments!'s own K output) is basis-independent -- reused directly
    objC.moments!(zeros(size(ctx.U,1)), CS.select_G_from_H(objC, objC.H), θ_full, objC.U, objC)
    objC.H[:, 2] .= 1.0
    cctxC = build_cm_bin_ctx(ctx, augC)

    Random.seed!(seed)
    xs = [zeros(n), 0.01 .* randn(n), 0.05 .* randn(n)]
    maxdiff_obj = 0.0; maxdiff_grad = 0.0; maxdiff_hess = 0.0
    conds_interval = Float64[]; conds_cumulative = Float64[]
    for x in xs
        gA = zeros(n); fA = objA(x, gA)
        hA_packed = Vector{Float64}(undef, n*(n+1)÷2)
        objA(x, h = hA_packed)
        HA = unpack_packed(hA_packed, n)

        # interval-native: reuse objA's FG (same obj/H/arg0 state -- objA(x,g) above already set
        # obj.arg0 for us), but swap the HESSIAN callback for the interval-native one.
        _archC_prep_for_hessian!(objA, x)
        hI_packed = Vector{Float64}(undef, n*(n+1)÷2)
        hessian_cm_structured_interval!(hI_packed, objA, cctxI)
        HI = unpack_packed(hI_packed, n)

        maxdiff_obj = max(maxdiff_obj, 0.0)   # objA vs itself trivially -- f/g compared via lookup-FG test elsewhere; this script's focus is the HESSIAN
        maxdiff_hess = max(maxdiff_hess, maximum(abs.(HA .- HI)))
        push!(conds_interval, cond(HI))

        # cumulative ArchC for the same random x (different obj/moment basis -- own FG state)
        nC = objC.outer_constr_index
        length(x) == nC || continue   # dimension matches since ncm identical for cumulative/interval at same L
        gC = zeros(nC); fC = objC(x, gC)
        _archC_prep_for_hessian!(objC, x)
        hC_packed = Vector{Float64}(undef, nC*(nC+1)÷2)
        hessian_cm_structured!(hC_packed, objC, cctxC)
        HC = unpack_packed(hC_packed, nC)
        push!(conds_cumulative, cond(HC))
    end
    @printf "  max|H_interval_native - H_dense_interval_ref| = %.3e\n" maxdiff_hess
    @printf "  cond(H) interval-native: min=%.3e max=%.3e | cond(H) cumulative: min=%.3e max=%.3e | ratio(cum/int) at max-x = %.2fx\n" minimum(conds_interval) maximum(conds_interval) minimum(conds_cumulative) maximum(conds_cumulative) (conds_cumulative[end]/conds_interval[end])
    return (maxdiff_hess = maxdiff_hess, cond_interval = conds_interval[end], cond_cumulative = conds_cumulative[end])
end

results = NamedTuple[]
for L in (10, 20, 50), contrasts in (:anchored, :orthonormal)
    r1 = validate_at(L, contrasts, θ_full_calib, "calibration"; seed = 2000 + L)
    r2 = validate_at(L, contrasts, θ_full_upper, "unrestricted_upper_candidate"; seed = 3000 + L)
    push!(results, (L = L, contrasts = contrasts, point = "calibration", r1...))
    push!(results, (L = L, contrasts = contrasts, point = "upper_candidate", r2...))
end

println()
println("="^100)
println("SUMMARY")
println("="^100)
for r in results
    @printf "L=%-2d %-11s %-22s maxdiff=%.3e cond_int=%.3e cond_cum=%.3e\n" r.L string(r.contrasts) r.point r.maxdiff_hess r.cond_interval r.cond_cumulative
end
worst = maximum(r.maxdiff_hess for r in results)
@printf "\nWORST max|H_interval_native - H_dense_interval_ref| across ALL points/L/contrasts: %.3e\n" worst
println(worst < 1e-8 ? "PASS: interval-native Architecture C matches the dense interval-basis reference to machine precision." : "FAIL: investigate.")
println("DONE")
