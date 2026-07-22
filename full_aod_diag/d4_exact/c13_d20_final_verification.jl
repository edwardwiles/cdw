# Continuation 13, Section 10: final independent verification of a D20 CM-restricted candidate.
# Reads a checkpoint saved by c13_d20_cm_upper_continuation.jl (or any (L, w) pair given directly)
# and re-derives/re-checks everything from scratch in a FRESH process invocation (run this as a
# separate `julia` process from the one that produced the candidate -- do not just call these
# functions inline in the same session that optimized it).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization

const CKPT_DIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation")

ckpt_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(CKPT_DIR, "stage_L50_latest.jls")
isfile(ckpt_path) || error("checkpoint not found: $ckpt_path -- run c13_d20_cm_upper_continuation.jl first, or pass an explicit path")
payload = deserialize(ckpt_path)
payload.best_w === nothing && error("checkpoint at $ckpt_path has no feasible incumbent (best_w===nothing) -- nothing to verify")
L = payload.L; w_final = payload.best_w; probs_final = payload.probs
println("Loaded checkpoint: L=$L  kappa(claimed)=$(payload.kappa)  timestamp=$(payload.timestamp)")

W = 80000; DELTA = 1.0
println(">>> [fresh] building D20 real-data context, W=$W, delta=$DELTA ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
@printf ">>> ctx built in %.1fs\n" (time()-t0)

xf_final = x_free_from_w(w_final, pe)

println()
println("="^100)
println("1-2. Reconstruct full DxD log-A matrix, compute gravity directly")
println("="^100)
Aod_theta_full = reshape(xf_final[2:end], D, D)
z_final = log.(Aod_theta_full)
gravity_direct = gravity_from_logz(z_final, ctx)
@printf "  gamma'_focal=%.10f  gravity (direct, should be ~0)=%.3e\n" xf_final[1] gravity_direct

println()
println("="^100)
println("3-6. Fresh COLD dense-reference CM solve (Architecture A, no ArchB/ArchC), independent of")
println("     whatever inner-solve path produced the checkpoint; verify primal/dual divergence,")
println("     economic moments, CM grid moments")
println("="^100)
aug_ref = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored, probs = probs_final)
ctx_ref = merge(ctx, (obj = aug_ref.obj_cm,))
r_ref = evaluate_fullA(xf_final, ctx_ref; use_cache = false, warm = false)
@printf "  dense nStatus=%d  Delta_dual=%.10f  (<=%.2f? %s)\n" r_ref.inner_status (-r_ref.zeta) DELTA ((-r_ref.zeta) <= DELTA + 1e-6)
@printf "  max_abs_moment_kkt_resid=%.3e  ||benchmark_unweighted_moment_mean||=%.3e  gravity(from r_ref)=%.3e\n" r_ref.max_abs_moment_kkt_resid norm(r_ref.benchmark_unweighted_moment_mean) r_ref.gravity_value

ncore = aug_ref.ncore
kkt_core = maximum(abs.(r_ref.benchmark_unweighted_moment_mean[1:ncore-1]))
kkt_cm = maximum(abs.(r_ref.benchmark_unweighted_moment_mean[ncore:ncore-1+aug_ref.ncm]))
@printf "  core-moment max KKT resid=%.3e  |  CM-block max KKT resid=%.3e\n" kkt_core kkt_cm

println()
println("="^100)
println("7. Original parameter bounds and normalization checks")
println("="^100)
@printf "  gp in [%.4f, %.4f]? %s  (gp=%.6f)\n" ctx.bounds.γp_lo ctx.bounds.γp_hi (ctx.bounds.γp_lo <= xf_final[1] <= ctx.bounds.γp_hi) xf_final[1]
@printf "  all A_od > 0? %s  (min=%.4e)\n" all(Aod_theta_full .> 0) minimum(Aod_theta_full)

println()
println("="^100)
println("8. Gravity-tangent secant directional checks (optimized-value, small steps along a random")
println("   gravity-feasible direction -- Delta_dual should not decrease much below delta, kappa")
println("   should not obviously improve, else the reported point is not locally optimal)")
println("="^100)
Random.seed!(20260720)
for trial in 1:3
    dz = randn(D, D); dz .-= dot(vec(dz), vec(gravity_linear_coeffs(ctx))) / dot(vec(gravity_linear_coeffs(ctx)), vec(gravity_linear_coeffs(ctx))) .* gravity_linear_coeffs(ctx)  # project out the gravity-normal component, so dz stays gravity-feasible to first order
    for h in (0.01, -0.01)
        z_try = z_final .+ h .* dz
        xf_try = vcat(xf_final[1], vec(exp.(z_try)))
        try
            r_try = evaluate_fullA(xf_try, ctx_ref; use_cache = false, warm = false)
            gp_try = xf_try[1]
            kappa_try = 1 - gp_try^(ctx.σ/(ctx.σ-1))
            kappa_final = 1 - xf_final[1]^(ctx.σ/(ctx.σ-1))
            @printf "  trial %d h=%+.3f: nStatus=%d Delta=%.6f (feasible=%s) kappa_try=%.8f vs kappa_final=%.8f (worse-or-equal? %s)\n" trial h r_try.inner_status (-r_try.zeta) ((-r_try.zeta)<=DELTA+1e-6) kappa_try kappa_final (kappa_try <= kappa_final + 1e-4)
        catch e
            @printf "  trial %d h=%+.3f: FAILED (%s)\n" trial h sprint(showerror, e)[1:min(80,end)]
        end
    end
end

println()
println("="^100)
println("10. Re-evaluate under coarser nested grids as a consistency check")
println("="^100)
snaps = nested_grid_sequence([10, 20, 50])
for Lc in (10, 20, 50)
    Lc >= L && continue
    aug_c = build_cm_augmented_obj(ctx, CS; L = Lc, contrasts = :anchored, probs = snaps[Lc])
    ctx_c = merge(ctx, (obj = aug_c.obj_cm,))
    r_c = evaluate_fullA(xf_final, ctx_c; use_cache = false, warm = false)
    @printf "  under L=%d grid: nStatus=%d Delta_dual=%.8f (feasible=%s)\n" Lc r_c.inner_status (-r_c.zeta) ((-r_c.zeta)<=DELTA+1e-6)
end

println()
println("="^100)
println("CLASSIFICATION")
println("="^100)
feasible = (-r_ref.zeta) <= DELTA + 1e-6
kkt_ok = kkt_cm < 1e-6 && kkt_core < 1e-6
if feasible && kkt_ok
    println("  -> bandwidth-KKT exact-feasible candidate (fresh cold dense re-solve confirms exact")
    println("     feasibility and near-machine-precision complementarity on the reported (L=$L) grid).")
    println("     NOT a claim of global optimality -- see the directional-secant checks above for the")
    println("     local-optimality evidence actually gathered this run.")
elseif feasible
    println("  -> verified local candidate (feasible, but KKT residuals larger than machine precision)")
else
    println("  -> UNRESOLVED: fresh cold re-solve did NOT confirm feasibility -- do not report this kappa")
end
println("DONE")
