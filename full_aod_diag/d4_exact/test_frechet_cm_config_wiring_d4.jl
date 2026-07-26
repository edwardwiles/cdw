# D=4 wiring gate (task Part II, Part VI preview): exercises marginal_restriction=:common_frechet
# through the SAME CMConfig-based production dispatch (build_cm_production_context_v2) real drivers
# use, including a real KNITRO inner solve, and checks it against marginal_restriction=:common_flexible
# at the identical outer point (the "flexible-CM nesting" diagnostic, task §17, previewed here at the
# construction/inner-solve level -- the full campaign-level version is Part VI).
include(joinpath(@__DIR__, "context.jl"))
const _D4E = @__DIR__
for f in ["winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl",
          "cm_frechet_level.jl","cm_config.jl"]
    include(joinpath(_D4E, f))
end
using Printf

npass = 0
nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free0 = ctx.θ0_up[ctx.free_idx]
D = ctx.D
L = 10

println("== startup manifest (marginal_restriction=:common_frechet) ==")
cfg_frechet = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :dense_reference,
                        contrasts = :anchored, marginal_restriction = :common_frechet)
print_frechet_startup_manifest(cfg_frechet, D, L)

println()
println("== startup manifest no-op check (marginal_restriction=:common_flexible) ==")
cfg_flex = CMConfig(common_marginals = true, cm_grid_size = L, cm_hessian_backend = :structured,
                     contrasts = :anchored, marginal_restriction = :common_flexible)
print_frechet_startup_manifest(cfg_flex, D, L)   # should print nothing
println("  (nothing printed above this line for :common_flexible -- correct)")

println()
println("== config validation guard ==")
guard_tripped = false
try
    global guard_tripped
    CMConfig(common_marginals = true, marginal_restriction = :common_frechet, cm_hessian_backend = :structured) |> _cm_validate
catch e
    global guard_tripped = e isa ErrorException
end
check(":common_frechet + :structured raises an error (Part III not yet wired)", guard_tripped)

println()
println("== real KNITRO inner solves via build_cm_production_context_v2 ==")
pcx_flex = build_cm_production_context_v2(ctx, CS, cfg_flex; L = L)
pcx_frechet = build_cm_production_context_v2(ctx, CS, cfg_frechet; L = L)

check("flexible obj.d - core == (D-1)*L", pcx_flex.aug.ncm == (D - 1) * L)
check("frechet obj.d - core == D*L", pcx_frechet.aug.ncm == D * L)
check("frechet has L more moments than flexible", pcx_frechet.aug.ncm - pcx_flex.aug.ncm == L)

base_flex = cm_base_state_v2(x_free0, pcx_flex)
base_frechet = cm_base_state_v2(x_free0, pcx_frechet)
@printf("  flexible CM:  inner_status=%d  ncm=%d\n", base_flex.inner_status, pcx_flex.aug.ncm)
@printf("  common Frechet: inner_status=%d  ncm=%d\n", base_frechet.inner_status, pcx_frechet.aug.ncm)
check("flexible CM inner solve feasible", base_flex.inner_status in (0, -100, -101, -103))
check("common Frechet inner solve feasible", base_frechet.inner_status in (0, -100, -101, -103))
check("common Frechet dual vector is L longer than flexible CM's", length(base_frechet.λstar) - length(base_flex.λstar) == L)

println()
println("== Architecture-B (production) moments match Architecture-A (dense reference) at this point ==")
aug_dense = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
Wn = size(ctx.U, 1)
K_archB = zeros(Wn); G_archB = zeros(Wn, pcx_frechet.ctx_cm.obj.d)
pcx_frechet.ctx_cm.obj.moments!(K_archB, G_archB, base_frechet.θ_full0, ctx.U, pcx_frechet.ctx_cm.obj)
K_archA = zeros(Wn); G_archA = zeros(Wn, aug_dense.obj_cm.d)
aug_dense.obj_cm.moments!(K_archA, G_archA, base_frechet.θ_full0, ctx.U, aug_dense.obj_cm)
max_diff = maximum(abs.(G_archB .- G_archA))
check("Architecture-B moments == Architecture-A dense reference (max abs diff < 1e-10)", max_diff < 1e-10)
@printf("  max|G_archB - G_archA| = %.3e\n", max_diff)

println()
println("==================================================")
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
