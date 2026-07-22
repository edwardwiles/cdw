# Smoke/equivalence test for cm_config.jl's CMConfig dispatcher (production integration
# continuation): confirms the new v2 unified entry points reproduce the pre-existing,
# already-validated Continuation 12/13 entry points exactly, for every (basis, backend)
# combination, before this is trusted anywhere near a real outer loop.
# Include list copied verbatim from c13_validate_production_bundle.jl (the last known-working
# smoke test for this exact machinery) with cm_config.jl appended.
include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]

println("="^100)
println("SECTION 1: CMConfig grid resolution")
println("="^100)
cfg_equal10 = CMConfig(cm_grid_rule = :equal, cm_grid_size = 10)
p10 = cm_resolve_probs(cfg_equal10)
old10 = collect(range(1/10, 9/10, length=10))
println("equal L=10 matches old default: ", p10 == old10)

cfg_nest = CMConfig(cm_grid_rule = :nested_family, cm_grid_sizes = [10, 20, 50])
fam = cm_resolve_probs(cfg_nest)
println("nested family sizes: ", sort(collect(keys(fam))))
println("Q10 subset Q20: ", issubset(Set(fam[10]), Set(fam[20])))
println("Q20 subset Q50: ", issubset(Set(fam[20]), Set(fam[50])))
println("Q10 subset Q50: ", issubset(Set(fam[10]), Set(fam[50])))

println()
println("="^100)
println("SECTION 2: v2 dispatcher reproduces cm_production_bundle.jl's cumulative+structured path")
println("="^100)
for L in (10, 20, 50)
    pcx_old = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = cm_equal_grid_probs(L))
    K_old, base_old = cm_production_value(x_free_calib, pcx_old)
    # Remediation fix (task Part A, F1): was `-base_old.ζstar`/`-base_new.ζstar` (mislabeled
    # Delta_dual -- omits mean(Psi(q*))).
    Dd_old = delta_dual_from_base(pcx_old.ctx_cm.obj, base_old)

    cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = L,
                    cm_basis = :cumulative, cm_hessian_backend = :structured)
    pcx_new = build_cm_production_context_v2(ctx, CS, cfg)
    K_new, base_new = cm_production_value_v2(x_free_calib, pcx_new)
    Dd_new = delta_dual_from_base(pcx_new.ctx_cm.obj, base_new)

    println("[L=$L] old Delta_dual=", Dd_old, " new Delta_dual=", Dd_new, " diff=", abs(Dd_old - Dd_new),
            " nStatus_old=", base_old.inner_status, " nStatus_new=", base_new.inner_status)
end

println()
println("="^100)
println("SECTION 3: v2 dense_reference backend agrees with structured (both bases), L=10")
println("="^100)
for basis in (:cumulative, :interval)
    cfg_struct = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = 10,
                           cm_basis = basis, cm_hessian_backend = :structured)
    cfg_dense = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = 10,
                          cm_basis = basis, cm_hessian_backend = :dense_reference)
    pcx_s = build_cm_production_context_v2(ctx, CS, cfg_struct)
    pcx_d = build_cm_production_context_v2(ctx, CS, cfg_dense)
    _, base_s = cm_production_value_v2(x_free_calib, pcx_s)
    _, base_d = cm_production_value_v2(x_free_calib, pcx_d)
    # Remediation fix (task Part A, F1): was `-base_s.ζstar`/`-base_d.ζstar`.
    Ks, Kd = delta_dual_from_base(pcx_s.ctx_cm.obj, base_s), delta_dual_from_base(pcx_d.ctx_cm.obj, base_d)
    println("[basis=$basis] structured Delta_dual=", Ks, " dense_reference Delta_dual=", Kd, " diff=", abs(Ks - Kd))
end

println()
println("="^100)
println("SECTION 4: cm_cache_key sanity")
println("="^100)
cfg_off = CMConfig(common_marginals = false)
cfg_on = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = 10, cm_basis = :cumulative)
println("off key: ", cm_cache_key(cfg_off, 10, hash(ctx.U)))
println("on key:  ", cm_cache_key(cfg_on, 10, hash(ctx.U)))
println("keys differ: ", cm_cache_key(cfg_off, 10, hash(ctx.U)) != cm_cache_key(cfg_on, 10, hash(ctx.U)))
println("DONE")
