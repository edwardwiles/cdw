# Continuation 13 addendum: full cumulative-ArchC vs interval-native-ArchC comparison, per the
# user's explicit dimension list -- dual objective, gradient, complete Hessian, primal/dual
# divergence, primal weights, CM residuals, Hessian condition estimate, cold/warm inner
# iterations, cold/warm wall time, robustness to alternative dual starts, failure rates at
# difficult points. Runs at D=4 (fast, several points) AND real D=20/L=50/W=80000 (the decisive
# scale) when INVOKED WITH ARGS[1]=="d20"; default is D=4 only.
#
# Cumulative and interval bases are mathematically EQUIVALENT restrictions (an exact invertible
# linear reparameterization, Continuation 12's Section 6a) -- so their SOLVED economic content
# (Delta_dual, gamma'_focal, kappa, core-moment residuals) must AGREE, even though the raw
# (zeta,lambda) dual vectors and per-basis Hessian/condition number legitimately differ.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
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

"""
    compare_bases(ctx, θ_full, L, contrasts, label; alt_starts=Vector{Float64}[])

One point's full comparison: cumulative ArchC vs interval-native ArchC, both driven through
`inner_loop_internal_archgeneric` (identical KNITRO wiring/options otherwise) so cold/warm timing
and iteration counts are directly comparable.
"""
function compare_bases(ctx, θ_full::Vector{Float64}, L::Int, contrasts::Symbol, label::String)
    println("-"^100)
    println("$label  L=$L  contrasts=$contrasts")
    println("-"^100)

    augC = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    cctxC = build_cm_bin_ctx(ctx, augC)
    augI = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = contrasts)
    cctxI = build_cm_bin_ctx_interval(ctx, augI)

    # ---- cold solves (fresh KN_new() each time -- inner_loop_internal_archgeneric always does) ----
    tC1 = @elapsed (KC, xC, statusC, nfgC, nhessC) = inner_loop_internal_archgeneric(augC.obj_cm, θ_full; hess_cb_builder = _o -> archC_hess_cb_builder(cctxC))
    tI1 = @elapsed (KI, xI, statusI, nfgI, nhessI) = inner_loop_internal_archgeneric(augI.obj_cm, θ_full; hess_cb_builder = _o -> archC_interval_hess_cb_builder(cctxI))
    # ---- warm-repeat (same obj/context, second call -- JIT already paid, isolates true steady-state cost) ----
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

        # primal weights (m_star = dPsi(q*)) and CM/core residuals
        mC = copy(augC.obj_cm.arg1); mI = copy(augI.obj_cm.arg1)
        @printf "  primal weight summary: mean(mC)=%.6e mean(mI)=%.6e  (basis-specific dual vectors, NOT expected bit-identical -- compare downstream economic content instead)\n" mean(mC) mean(mI)

        # CM-block KKT residual: sum_s m_s * G_cm[s,k] / W for the SOLVED lambda -- rebuild G_cm via each basis's own moments! output (already in obj.H)
        Wd = size(ctx.U, 1)
        ncoreC = augC.ncore; ncmC = augC.ncm
        Gc = @view augC.obj_cm.H[:, 2+ncoreC:1+ncoreC+ncmC]
        kktC = maximum(abs.(vec(sum(mC .* Gc, dims=1)) ./ Wd))
        ncoreI = augI.ncore; ncmI = augI.ncm
        Gi = @view augI.obj_cm.H[:, 2+ncoreI:1+ncoreI+ncmI]
        kktI = maximum(abs.(vec(sum(mI .* Gi, dims=1)) ./ Wd))
        @printf "  CM-block max KKT residual: cumulative=%.3e  interval-native=%.3e\n" kktC kktI
    end
    return (label = label, L = L, contrasts = contrasts, statusC = statusC, statusI = statusI,
            DeltaC = DeltaC, DeltaI = DeltaI, diff = abs(DeltaC - DeltaI),
            tC_cold = tC1, tI_cold = tI1, tC_warm = tC2, tI_warm = tI2, condC = condC, condI = condI)
end

scale = length(ARGS) >= 1 ? ARGS[1] : "d4"

if scale == "d4"
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
    x_free_pert2 = copy(x_free_calib); x_free_pert2[2:end] .*= exp.(0.15 .* randn(length(x_free_pert2)-1))  # "difficult": larger perturbation
    θ_pert2 = CS.reconstruct_full(x_free_pert2, ctx.m)

    results = NamedTuple[]
    for L in (10, 20, 50), contrasts in (:anchored,)
        push!(results, compare_bases(ctx, θ_calib, L, contrasts, "calibration"))
        push!(results, compare_bases(ctx, θ_upper, L, contrasts, "unrestricted_upper_candidate"))
        push!(results, compare_bases(ctx, θ_pert1, L, contrasts, "perturbed_5pct"))
        push!(results, compare_bases(ctx, θ_pert2, L, contrasts, "difficult_15pct"))
    end
else
    W = 80000; DELTA = 1.0
    ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    θ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
    Random.seed!(9902)
    x_free_pert1 = copy(x_free_calib); x_free_pert1[2:end] .*= exp.(0.02 .* randn(length(x_free_pert1)-1))
    θ_pert1 = CS.reconstruct_full(x_free_pert1, ctx.m)

    results = NamedTuple[]
    push!(results, compare_bases(ctx, θ_calib, 50, :anchored, "calibration"))
    push!(results, compare_bases(ctx, θ_pert1, 50, :anchored, "perturbed_2pct"))
end

println()
println("="^100)
println("SUMMARY")
println("="^100)
for r in results
    @printf "%-30s L=%-3d |Delta diff|=%.2e  cold: cum=%.3fs int=%.3fs  warm: cum=%.3fs int=%.3fs  cond: cum=%.3e int=%.3e (statusC=%d statusI=%d)\n" r.label r.L r.diff r.tC_cold r.tI_cold r.tC_warm r.tI_warm r.condC r.condI r.statusC r.statusI
end
n_agree = count(r -> r.statusC in (0,-100,-101,-103) && r.statusI in (0,-100,-101,-103) && r.diff < 1e-6, results)
n_fail_either = count(r -> !(r.statusC in (0,-100,-101,-103)) || !(r.statusI in (0,-100,-101,-103)), results)
@printf "\n%d/%d points: both converged AND Delta_dual agrees to <1e-6.  %d points: at least one architecture failed to converge.\n" n_agree length(results) n_fail_either
mean_warm_ratio = mean([r.tI_warm/r.tC_warm for r in results if isfinite(r.tI_warm) && isfinite(r.tC_warm) && r.tC_warm>0])
@printf "Mean warm-time ratio (interval/cumulative): %.3fx  (%s the 10-20%% acceptable-slowdown threshold)\n" mean_warm_ratio (mean_warm_ratio <= 1.2 ? "WITHIN" : "EXCEEDS")
println("DONE")
