# Phase C (2026-07-27), item 14: four-arm (cm_basis x origin_contrasts) comparison at D=4.
#
# Reuses `compare_bases` VERBATIM from c13_cumulative_vs_interval_native_comparison.jl (copied,
# not reimplemented -- that script's own bottom half has a side-effecting main block that runs on
# include(), so this file duplicates just the reusable function, exactly as
# section7_cm_basis_recheck_d20.jl already does for its own D20 extension). `compare_bases` already
# accepts a `contrasts` argument and threads it through BOTH `build_cm_augmented_obj` (cumulative)
# and `build_cm_augmented_obj_interval` (interval) -- so looping contrasts in ADDITION TO the basis
# comparison compare_bases already does per call gives all four arms
# (cumulative+anchored, cumulative+orthonormal, interval+anchored, interval+orthonormal) with zero
# new numerical-kernel code, only a new driver loop. This is deliberate: the equivalence math
# (transform matrix, end-to-end inner-solve agreement, interval-native Hessian correctness) was
# already exhaustively re-validated at both contrasts in c12i_validate_interval_equiv.jl /
# c13_validate_interval_native_archC.jl (re-run fresh on this session's HEAD, see
# docs/key_results/c12i_rerun_2026-07-27.log and c13_validate_interval_native_archC_rerun_2026-07-27.log)
# -- this script's OWN job is the four-arm timing/conditioning comparison task item 14 additionally
# asks for, not re-deriving equivalence a third time.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
using Printf, LinearAlgebra, Random, Statistics

unpack_packed(h, n) = (M = Matrix{Float64}(undef, n, n); k = 1; for i in 1:n, j in i:n; M[i,j]=h[k]; M[j,i]=h[k]; k+=1; end; M)

"Verbatim copy of c13_cumulative_vs_interval_native_comparison.jl::compare_bases (D4/D20 shared)."
function compare_bases(ctx, θ_full::Vector{Float64}, L::Int, contrasts::Symbol, label::String)
    println("-"^100)
    println("$label  L=$L  contrasts=$contrasts")
    println("-"^100)

    augC = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    cctxC = build_cm_bin_ctx(ctx, augC)
    augI = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = contrasts)
    cctxI = build_cm_bin_ctx_interval(ctx, augI)

    tC1 = @elapsed (KC, xC, statusC, nfgC, nhessC) = inner_loop_internal_archgeneric(augC.obj_cm, θ_full; hess_cb_builder = _o -> archC_hess_cb_builder(cctxC))
    tI1 = @elapsed (KI, xI, statusI, nfgI, nhessI) = inner_loop_internal_archgeneric(augI.obj_cm, θ_full; hess_cb_builder = _o -> archC_interval_hess_cb_builder(cctxI))
    tC2 = @elapsed (KC2, xC2, statusC2, nfgC2, nhessC2) = inner_loop_internal_archgeneric(augC.obj_cm, θ_full; hess_cb_builder = _o -> archC_hess_cb_builder(cctxC))
    tI2 = @elapsed (KI2, xI2, statusI2, nfgI2, nhessI2) = inner_loop_internal_archgeneric(augI.obj_cm, θ_full; hess_cb_builder = _o -> archC_interval_hess_cb_builder(cctxI))

    ζC = xC[1]; λC = xC[2:end]; ζI = xI[1]; λI = xI[2:end]
    DeltaC = -ζC; DeltaI = -ζI

    @printf "  cumulative:      status=%-4d  Delta_dual=%.10f  cold=%.3fs(nfg=%d,nhess=%d)  warm=%.3fs(nfg=%d,nhess=%d)\n" statusC DeltaC tC1 nfgC nhessC tC2 nfgC2 nhessC2
    @printf "  interval-native: status=%-4d  Delta_dual=%.10f  cold=%.3fs(nfg=%d,nhess=%d)  warm=%.3fs(nfg=%d,nhess=%d)\n" statusI DeltaI tI1 nfgI nhessI tI2 nfgI2 nhessI2
    @printf "  |Delta_dual diff| = %.3e   (should be ~0 -- same restriction, exact reparameterization)\n" abs(DeltaC - DeltaI)

    both_ok = statusC in (0,-100,-101,-103) && statusI in (0,-100,-101,-103)
    condC = NaN; condI = NaN
    rankC = -1; rankI = -1
    if both_ok
        nC = augC.obj_cm.outer_constr_index; nI = augI.obj_cm.outer_constr_index
        hCp = Vector{Float64}(undef, nC*(nC+1)÷2); hessian_cm_structured!(hCp, augC.obj_cm, cctxC)
        hIp = Vector{Float64}(undef, nI*(nI+1)÷2); hessian_cm_structured_interval!(hIp, augI.obj_cm, cctxI)
        condC = cond(unpack_packed(hCp, nC)); condI = cond(unpack_packed(hIp, nI))
        rankC = rank(augC.CM); rankI = rank(augI.CM)
        @printf "  cond(H) cumulative=%.4e  cond(H) interval-native=%.4e  (interval/cumulative ratio=%.4f, <1 means BETTER conditioned)\n" condC condI (condI/condC)
        @printf "  rank(CM) cumulative=%d  rank(CM) interval=%d  (ncm=%d, both should be full column rank)\n" rankC rankI size(augC.CM,2)
    end
    return (label = label, L = L, contrasts = contrasts, statusC = statusC, statusI = statusI,
            DeltaC = DeltaC, DeltaI = DeltaI, diff = abs(DeltaC - DeltaI),
            tC_cold = tC1, tI_cold = tI1, tC_warm = tC2, tI_warm = tI2, condC = condC, condI = condI,
            rankC = rankC, rankI = rankI, ncm = size(augC.CM,2))
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
pe = build_pivot_elimination(ctx)
const_w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
                0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
                1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
                0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
x_free_upper = vcat(const_w_up40[1], vec(exp.(pivot_expand(const_w_up40[2:end], pe))))
θ_upper = CS.reconstruct_full(x_free_upper, ctx.m)

Random.seed!(9901)
x_free_pert1 = copy(x_free_calib); x_free_pert1[2:end] .*= exp.(0.05 .* randn(length(x_free_pert1)-1))
θ_pert1 = CS.reconstruct_full(x_free_pert1, ctx.m)
x_free_pert2 = copy(x_free_calib); x_free_pert2[2:end] .*= exp.(0.15 .* randn(length(x_free_pert2)-1))
θ_pert2 = CS.reconstruct_full(x_free_pert2, ctx.m)

results = NamedTuple[]
for L in (10, 50), contrasts in (:anchored, :orthonormal)
    push!(results, compare_bases(ctx, θ_calib, L, contrasts, "calibration"))
    push!(results, compare_bases(ctx, θ_upper, L, contrasts, "unrestricted_upper_candidate"))
    push!(results, compare_bases(ctx, θ_pert1, L, contrasts, "perturbed_5pct"))
    push!(results, compare_bases(ctx, θ_pert2, L, contrasts, "difficult_15pct"))
end

println()
println("="^110)
println("SUMMARY: 4 arms (cm_basis x origin_contrasts) at D=4")
println("="^110)
@printf "%-30s %-3s %-11s %10s %8s %8s %8s %8s %12s %12s %6s %6s\n" "point" "L" "contrasts" "Ddual_dif" "cum_cold" "int_cold" "cum_warm" "int_warm" "cond_cum" "cond_int" "rk_C" "rk_I"
for r in results
    @printf "%-30s %-3d %-11s %10.2e %8.3f %8.3f %8.3f %8.3f %12.4e %12.4e %6d %6d\n" r.label r.L string(r.contrasts) r.diff r.tC_cold r.tI_cold r.tC_warm r.tI_warm r.condC r.condI r.rankC r.rankI
end

n_agree = count(r -> r.statusC in (0,-100,-101,-103) && r.statusI in (0,-100,-101,-103) && r.diff < 1e-6, results)
n_fail = count(r -> !(r.statusC in (0,-100,-101,-103)) || !(r.statusI in (0,-100,-101,-103)), results)
n_full_rank = count(r -> r.rankC == r.ncm && r.rankI == r.ncm, results)
@printf "\n%d/%d points: both bases converged AND Delta_dual agrees to <1e-6.  %d points: at least one architecture failed.  %d/%d points: both CM matrices full column rank.\n" n_agree length(results) n_fail n_full_rank length(results)

# anchored vs orthonormal conditioning comparison, within each basis, at each L
println()
println("="^110)
println("Anchored vs orthonormal conditioning ratio (orthonormal/anchored), by basis and L -- D=4")
println("="^110)
for L in (10, 50), (basisname, key) in (("cumulative", :condC), ("interval", :condI))
    rA = only(filter(r -> r.L == L && r.contrasts == :anchored && r.label == "calibration", results))
    rO = only(filter(r -> r.L == L && r.contrasts == :orthonormal && r.label == "calibration", results))
    condA = getproperty(rA, key); condO = getproperty(rO, key)
    @printf "  L=%-3d basis=%-11s cond(anchored)=%.4e  cond(orthonormal)=%.4e  ratio(ortho/anchored)=%.4f  (%s)\n" L basisname condA condO (condO/condA) (condO < condA ? "orthonormal BETTER" : "anchored BETTER")
end
println("DONE")
