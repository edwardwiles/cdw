# Production integration continuation, Section 7: re-evaluate cumulative vs. interval-native at
# more points than Continuation 13's original D20 comparison (which only tested calibration + one
# generic 2% perturbation). Reuses `compare_bases` from
# c13_cumulative_vs_interval_native_comparison.jl VERBATIM (copied, not modified -- that script's
# own bottom half has a side-effecting main block that runs on include(), so this file duplicates
# just the reusable function rather than including the whole script).
#
# Points tested, all D20/W=80000:
#   1. calibration
#   2. CM outer-trajectory accepted point, L=10 stage (results/fullA_d4/c13_d20_cm_continuation/stage_L10_latest.jls)
#   3. CM outer-trajectory accepted point, L=20 stage (stage_L20_latest.jls)
#   4. CM outer-trajectory accepted point, L=50 stage (stage_L50_latest.jls) -- this IS "the latest
#      L=50 CM candidate" the brief separately asks for
#   5. a constructed near-infeasible point (4% multiplicative perturbation of calibration)
#
# NOT included (honestly, not fabricated): a "latest unrestricted delta=1 candidate" checkpoint --
# the canonical D20 unrestricted rerun lives on a DIFFERENT worktree
# (gravity-fullA-d20-canonical-rerun, diag/fullA-d20-canonical-rerun) not merged into this
# integration's history; loading it here would require reading another branch's untracked result
# files, out of scope for this pass. Documented as a gap in the production integration doc.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization

unpack_packed(h, n) = (M = Matrix{Float64}(undef, n, n); k = 1; for i in 1:n, j in i:n; M[i,j]=h[k]; M[j,i]=h[k]; k+=1; end; M)

"Verbatim copy of c13_cumulative_vs_interval_native_comparison.jl::compare_bases -- see that file for the original."
function compare_bases(ctx, θ_full::Vector{Float64}, L::Int, contrasts::Symbol, label::String; probs = nothing)
    println("-"^100)
    println("$label  L=$L  contrasts=$contrasts  probs=$(probs === nothing ? "default k/L" : "explicit ($(length(probs)) pts)")")
    println("-"^100)

    augC = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts, probs = probs)
    cctxC = build_cm_bin_ctx(ctx, augC)
    augI = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = contrasts, probs = probs)
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
    if both_ok
        nC = augC.obj_cm.outer_constr_index; nI = augI.obj_cm.outer_constr_index
        hCp = Vector{Float64}(undef, nC*(nC+1)÷2); hessian_cm_structured!(hCp, augC.obj_cm, cctxC)
        hIp = Vector{Float64}(undef, nI*(nI+1)÷2); hessian_cm_structured_interval!(hIp, augI.obj_cm, cctxI)
        condC = cond(unpack_packed(hCp, nC)); condI = cond(unpack_packed(hIp, nI))
        @printf "  cond(H) cumulative=%.4e  cond(H) interval-native=%.4e  (interval/cumulative ratio=%.4f, <1 means BETTER conditioned)\n" condC condI (condI/condC)
    end
    return (label = label, L = L, statusC = statusC, statusI = statusI,
            DeltaC = DeltaC, DeltaI = DeltaI, diff = abs(DeltaC - DeltaI),
            tC_cold = tC1, tI_cold = tI1, tC_warm = tC2, tI_warm = tI2, condC = condC, condI = condI)
end

W = 80000; DELTA = 1.0
println(">>> building D20 real-data context, W=$W ...")
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

results = NamedTuple[]
push!(results, compare_bases(ctx, θ_calib, 50, :anchored, "calibration"))

ckpt_dir = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation")
for L in (10, 20, 50)
    path = joinpath(ckpt_dir, "stage_L$(L)_latest.jls")
    if !isfile(path)
        println(">>> checkpoint not found: $path -- skipping")
        continue
    end
    payload = deserialize(path)
    if payload.best_w === nothing
        println(">>> stage_L$(L) has no feasible incumbent recorded -- skipping")
        continue
    end
    x_free = x_free_from_w(payload.best_w)
    θ_full = CS.reconstruct_full(x_free, ctx.m)
    push!(results, compare_bases(ctx, θ_full, L, :anchored, "cm_outer_trajectory_stage_L$(L)"; probs = payload.probs))
end

Random.seed!(20260721)
x_free_near_infeasible = copy(x_free_calib)
x_free_near_infeasible[2:end] .*= exp.(0.04 .* randn(length(x_free_near_infeasible) - 1))
θ_near_infeasible = CS.reconstruct_full(x_free_near_infeasible, ctx.m)
push!(results, compare_bases(ctx, θ_near_infeasible, 50, :anchored, "near_infeasible_4pct_perturbation"))

println()
println("="^100)
println("SUMMARY")
println("="^100)
for r in results
    @printf "%-38s L=%-3d |Delta diff|=%.2e  cold: cum=%.3fs int=%.3fs  warm: cum=%.3fs int=%.3fs  cond: cum=%.3e int=%.3e ratio=%.2fx (statusC=%d statusI=%d)\n" r.label r.L r.diff r.tC_cold r.tI_cold r.tC_warm r.tI_warm r.condC r.condI (isnan(r.condC)||isnan(r.condI) ? NaN : r.condI/r.condC) r.statusC r.statusI
end
n_agree = count(r -> r.statusC in (0,-100,-101,-103) && r.statusI in (0,-100,-101,-103) && r.diff < 1e-6, results)
println()
@printf "%d/%d points: both converged AND Delta_dual agrees to <1e-6.\n" n_agree length(results)
println("DONE")
