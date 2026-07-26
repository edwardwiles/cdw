# Phase E remediation (production-audit continuation, 2026-07-26) correctness gate:
# cf_build/CompressedFactualWorkspace wiring for the four restricted families' hot moment-build
# path. build_compressed_factual! is already independently validated
# (test_compressed_factual_buffer_reuse.jl) to be bit-identical to build_compressed_factual -- this
# gate proves the WIRING (ctx.cf_workspace attachment + cf_build dispatch) reaches production
# correctly, through the real archC_verified_state call path, for plain flexible CM.
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
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
using Printf, Random

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

ctx0 = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
Random.seed!(9090)
x_free_pert = copy(x_free_calib)
x_free_pert[2:end] .*= exp.(0.03 .* randn(length(x_free_pert) - 1))

W = size(ctx0.U, 1)
D = ctx0.D; Ddest = hasproperty(ctx0, :D_dest) ? ctx0.D_dest : ctx0.D

# Path A: no workspace attached (original allocating build_compressed_factual, via cf_build's own fallback branch).
pcx_noWS = build_cm_production_context(ctx0, CS; L = 10, contrasts = :anchored)
check(!hasproperty(pcx_noWS.ctx_cm, :cf_workspace), "control ctx has no cf_workspace attached (fallback path exercised)")

# Path B: workspace attached BEFORE building pcx (production wiring pattern).
ctx_ws = attach_compressed_factual_workspace(ctx0, D, Ddest, W)
check(hasproperty(ctx_ws, :cf_workspace), "attach_compressed_factual_workspace attaches the field")
check(ctx_ws.cf_workspace.D == D && ctx_ws.cf_workspace.Ddest == Ddest && ctx_ws.cf_workspace.W == W, "workspace shape matches ctx")
pcx_WS = build_cm_production_context(ctx_ws, CS; L = 10, contrasts = :anchored)
check(hasproperty(pcx_WS.ctx_cm, :cf_workspace), "pcx.ctx_cm carries the attached cf_workspace through merge(ctx,(obj=...))")

for (label, xf) in (("calib", x_free_calib), ("perturbed", x_free_pert))
    base_noWS, verify_noWS = archC_verified_state(xf, pcx_noWS.ctx_cm, pcx_noWS.cctx)
    base_WS, verify_WS = archC_verified_state(xf, pcx_WS.ctx_cm, pcx_WS.cctx)
    check(verify_noWS.inner_status == verify_WS.inner_status, "$label: inner_status agrees (noWS=$(verify_noWS.inner_status), WS=$(verify_WS.inner_status))")
    d = abs(verify_noWS.Delta_dual - verify_WS.Delta_dual)
    check(d < 1e-9, "$label: Delta_dual agrees, workspace vs no-workspace (diff=$d)")
    ld = maximum(abs.(base_noWS.λstar .- base_WS.λstar))
    check(ld < 1e-8, "$label: lambda* agrees, workspace vs no-workspace (max abs diff=$ld)")
end

lp("==================== SUMMARY ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES: ", FAILURES)
    exit(1)
end
