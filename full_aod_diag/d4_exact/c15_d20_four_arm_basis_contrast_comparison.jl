# Phase C (2026-07-27), item 14: four-arm (cm_basis x origin_contrasts) comparison at real
# D=20/W=80,000 (seed 20260719, destination_sample=:exclude_row, fixed theta, near-delta=1).
#
# This is the genuinely NEW evidence this session adds: every prior anchored-vs-orthonormal
# conditioning claim in this codebase's history (docs/fullA_cm_conditioning_and_adaptive_grid_report.md,
# docs/ORTHONORMAL_ORIGIN_CONTRAST_BENCHMARK_2026-07-26.md) was measured ONLY at D=4 -- the D=20
# case was explicitly flagged as "NOT run -- qualitative projection only"
# (docs/fullA_cm_hessian_architecture_report.md section 10). Given this exact codebase's own
# documented, repeated pattern of D4 conditioning findings REVERSING at D20 (the cumulative-vs-
# interval basis finding itself: interval strictly BETTER at D4, ~2.6x-39x WORSE at D20 --
# docs/fullA_common_marginals_production_integration.md section 4), extrapolating the D4
# anchored-vs-orthonormal finding to D20 without checking would repeat exactly the trap
# CLAUDE.md's standing feedback (feedback-gravity-elimination-zero-is-not-calibration,
# feedback-verify-before-causal-claims) warns against. This script closes that gap directly.
#
# Uses `d20_real_setup_design` (NOT bare `d20_real_setup`) per the task's exact real-D20 recipe --
# `d20_real_setup_design` explicitly seeds `Random.seed!(draw_seed)` immediately before building the
# context (see gateC-d20-real-setup-vs-d20-real-setup-design-seeding.md: bare d20_real_setup is
# UNSEEDED, a known live gotcha in this codebase).
#
# `compare_bases` reused VERBATIM from c13_cumulative_vs_interval_native_comparison.jl /
# c15_d4_four_arm_basis_contrast_comparison.jl -- same function, same architecture-C callbacks,
# just pointed at a D20 ctx and looped over both `contrasts` values (the D20 recheck scripts that
# already existed, section7_cm_basis_recheck_d20.jl, only ever ran contrasts=:anchored).
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
using Printf, LinearAlgebra, Random, Statistics
flush(stdout)

unpack_packed(h, n) = (M = Matrix{Float64}(undef, n, n); k = 1; for i in 1:n, j in i:n; M[i,j]=h[k]; M[j,i]=h[k]; k+=1; end; M)

"Verbatim copy of c13_cumulative_vs_interval_native_comparison.jl::compare_bases."
function compare_bases(ctx, θ_full::Vector{Float64}, L::Int, contrasts::Symbol, label::String)
    println("-"^100); flush(stdout)
    println("$label  L=$L  contrasts=$contrasts")
    println("-"^100)

    augC = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    cctxC = build_cm_bin_ctx(ctx, augC)
    augI = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = contrasts)
    cctxI = build_cm_bin_ctx_interval(ctx, augI)

    tC1 = @elapsed (KC, xC, statusC, nfgC, nhessC) = inner_loop_internal_archgeneric(augC.obj_cm, θ_full; hess_cb_builder = _o -> archC_hess_cb_builder(cctxC))
    flush(stdout)
    tI1 = @elapsed (KI, xI, statusI, nfgI, nhessI) = inner_loop_internal_archgeneric(augI.obj_cm, θ_full; hess_cb_builder = _o -> archC_interval_hess_cb_builder(cctxI))
    flush(stdout)
    tC2 = @elapsed (KC2, xC2, statusC2, nfgC2, nhessC2) = inner_loop_internal_archgeneric(augC.obj_cm, θ_full; hess_cb_builder = _o -> archC_hess_cb_builder(cctxC))
    tI2 = @elapsed (KI2, xI2, statusI2, nfgI2, nhessI2) = inner_loop_internal_archgeneric(augI.obj_cm, θ_full; hess_cb_builder = _o -> archC_interval_hess_cb_builder(cctxI))

    ζC = xC[1]; λC = xC[2:end]; ζI = xI[1]; λI = xI[2:end]
    DeltaC = -ζC; DeltaI = -ζI

    @printf "  cumulative:      status=%-4d  Delta_dual=%.10f  cold=%.3fs(nfg=%d,nhess=%d)  warm=%.3fs(nfg=%d,nhess=%d)\n" statusC DeltaC tC1 nfgC nhessC tC2 nfgC2 nhessC2
    @printf "  interval-native: status=%-4d  Delta_dual=%.10f  cold=%.3fs(nfg=%d,nhess=%d)  warm=%.3fs(nfg=%d,nhess=%d)\n" statusI DeltaI tI1 nfgI nhessI tI2 nfgI2 nhessI2
    @printf "  |Delta_dual diff| = %.3e   (should be ~0 -- same restriction, exact reparameterization)\n" abs(DeltaC - DeltaI)
    flush(stdout)

    both_ok = statusC in (0,-100,-101,-103) && statusI in (0,-100,-101,-103)
    condC = NaN; condI = NaN
    if both_ok
        nC = augC.obj_cm.outer_constr_index; nI = augI.obj_cm.outer_constr_index
        hCp = Vector{Float64}(undef, nC*(nC+1)÷2); hessian_cm_structured!(hCp, augC.obj_cm, cctxC)
        hIp = Vector{Float64}(undef, nI*(nI+1)÷2); hessian_cm_structured_interval!(hIp, augI.obj_cm, cctxI)
        condC = cond(unpack_packed(hCp, nC)); condI = cond(unpack_packed(hIp, nI))
        @printf "  cond(H) cumulative=%.4e  cond(H) interval-native=%.4e  (interval/cumulative ratio=%.4f, <1 means BETTER conditioned)\n" condC condI (condI/condC)
        flush(stdout)
    end
    return (label = label, L = L, contrasts = contrasts, statusC = statusC, statusI = statusI,
            DeltaC = DeltaC, DeltaI = DeltaI, diff = abs(DeltaC - DeltaI),
            tC_cold = tC1, tI_cold = tI1, tC_warm = tC2, tI_warm = tI2, condC = condC, condI = condI)
end

println(">>> building D20 real-data context via d20_real_setup_design (W=80000, seed=20260719, destination_sample=:exclude_row) ...")
flush(stdout)
t_ctx = @elapsed ctx = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true)
@printf ">>> context built in %.1fs, D=%d, draw_design=%s, draw_seed=%d\n" t_ctx ctx.D ctx.draw_design ctx.draw_seed
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(9902)
x_free_pert1 = copy(x_free_calib); x_free_pert1[2:end] .*= exp.(0.02 .* randn(length(x_free_pert1)-1))
θ_pert1 = CS.reconstruct_full(x_free_pert1, ctx.m)

results = NamedTuple[]
for contrasts in (:anchored, :orthonormal)
    push!(results, compare_bases(ctx, θ_calib, 50, contrasts, "calibration"))
    push!(results, compare_bases(ctx, θ_pert1, 50, contrasts, "perturbed_2pct"))
end

println()
println("="^115)
println("SUMMARY: 4 arms (cm_basis x origin_contrasts) at real D=20/W=80,000/seed=20260719/exclude_row/L=50")
println("="^115)
for r in results
    @printf "%-20s L=%-3d contrasts=%-11s |Delta diff|=%.2e  cold: cum=%.3fs int=%.3fs  warm: cum=%.3fs int=%.3fs  cond: cum=%.4e int=%.4e ratio(int/cum)=%.3fx (statusC=%d statusI=%d)\n" r.label r.L string(r.contrasts) r.diff r.tC_cold r.tI_cold r.tC_warm r.tI_warm r.condC r.condI (r.condI/r.condC) r.statusC r.statusI
end

n_agree = count(r -> r.statusC in (0,-100,-101,-103) && r.statusI in (0,-100,-101,-103) && r.diff < 1e-6, results)
println()
@printf "%d/%d points: both bases converged AND Delta_dual agrees to <1e-6 (equivalence check).\n" n_agree length(results)

println()
println("="^115)
println("Anchored vs orthonormal conditioning ratio (orthonormal/anchored), by basis -- real D=20")
println("="^115)
for (basisname, key) in (("cumulative", :condC), ("interval", :condI))
    for lbl in ("calibration", "perturbed_2pct")
        rA = only(filter(r -> r.contrasts == :anchored && r.label == lbl, results))
        rO = only(filter(r -> r.contrasts == :orthonormal && r.label == lbl, results))
        condA = getproperty(rA, key); condO = getproperty(rO, key)
        @printf "  point=%-16s basis=%-11s cond(anchored)=%.4e  cond(orthonormal)=%.4e  ratio(ortho/anchored)=%.4f  (%s)\n" lbl basisname condA condO (condO/condA) (condO < condA ? "orthonormal BETTER" : "anchored BETTER")
    end
end
println("DONE")
flush(stdout)
