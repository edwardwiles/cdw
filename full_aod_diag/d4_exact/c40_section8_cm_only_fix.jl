# Corrective rerun: Section 8's original cm_only baseline used the F1-buggy `-r.zeta` proxy
# instead of the canonical `r.Delta_dual` (a real bug found after the full Section 8 run
# completed -- see c40_section8_d20_fixed_point_trial.jl's fix). This script ONLY recomputes the
# 4 cm_only Delta_dual values (canonical) at the same 4 (point, L) combinations -- the extended
# arms' Delta_floor values already used the canonical verify.Delta_dual and are NOT recomputed
# here (cheap: 4 inner solves, not the full ~40-minute profiling grid).
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
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, Dates, Serialization

println("Start: ", now()); flush(stdout)
t0 = time()
ctx = d20_real_setup(W = 80000, δ = 1.0)
@printf "d20_real_setup wall = %.1fs\n" (time() - t0); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
D = ctx.D
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0_calib = vcat(x_free_calib[1], pivot_reduce(z0, pe))

path = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation", "stage_L50_latest.jls")
d = deserialize(path)
@assert length(d.best_w) == length(w0_calib)
existing_w = d.best_w

POINTS = [("calibration", x_free_calib), ("existing_L50_incumbent", x_free_from_w(existing_w, pe))]

for L in (10, 50)
    for (pname, xf) in POINTS
        aug_cm = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
        ctx_cm_plain = merge(ctx, (obj = aug_cm.obj_cm,))
        t0 = time()
        r_cm = evaluate_fullA(xf, ctx_cm_plain; use_cache = false, warm = false)
        @printf "L=%d point=%s: Delta_dual(canonical)=%.8f  (-zeta proxy would have been %.8f, diff=%.4e)  inner_status=%d  wall=%.1fs\n" L pname r_cm.Delta_dual (-r_cm.zeta) (r_cm.Delta_dual - (-r_cm.zeta)) r_cm.inner_status (time()-t0)
        flush(stdout)
    end
end
println("Done: ", now())
