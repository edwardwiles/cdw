# Continuation 12 (D=20 prep), brief Section 15 "First: production microbenchmarks" -- smoke test
# before the real W=80000 benchmark table: confirm the CM-augmented bundle (cumulative basis +
# Architecture C Hessian, the combination validated at D=4 in c14_combined_bundle_validate.jl)
# builds and solves correctly on REAL D=20 data at a SMALL W first (correctness before scale).
# Do NOT run this at W=80000 without reading the memory notes in context_real_d20.jl first --
# needs_outer_moment_jacobian defaults to false there for exactly this reason.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
using Printf

W = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8000
L = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10

println(">>> D=20 real-data CM smoke test: W=$W, L=$L"); flush(stdout)
t_setup0 = time()
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true)
println(">>> setup done in $(round(time()-t_setup0, digits=1))s. D=$(ctx.D)  ncore(base)=$(ctx.obj.d)  refIndex1=$(ctx.γ.refIndex1)"); flush(stdout)

t_cm0 = time()
aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
println(">>> CM block built in $(round(time()-t_cm0, digits=2))s: ncm=$(aug.ncm)  d_total=$(aug.obj_cm.d)  outer_constr_index=$(aug.obj_cm.outer_constr_index)"); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
println(">>> free dims = $(length(x_free_calib))"); flush(stdout)

# ---- dense reference (Architecture A) ----
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
t0 = time()
r_dense = evaluate_fullA(x_free_calib, ctx_cm; use_cache = false, warm = false)
t_dense = time() - t0
@printf(">>> DENSE (ArchA): nStatus=%d  Delta_dual=%.6f  t=%.3fs  max_kkt_resid=%.2e\n",
        r_dense.inner_status, r_dense.Delta_dual, t_dense, r_dense.max_abs_moment_kkt_resid)
flush(stdout)

@assert r_dense.inner_status in (0, -100, -101, -103) "D=20 CM smoke test: dense reference inner solve FAILED, nStatus=$(r_dense.inner_status) -- stop here, do not proceed to Architecture C or W=80000"

# ---- Architecture C combined bundle ----
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
cctx_t0 = time()
cctx = build_cm_bin_ctx(ctx, aug)
println(">>> Architecture C bin-table context built in $(round(time()-cctx_t0, digits=2))s"); flush(stdout)
cctx_callback = archC_hess_cb_builder(cctx)
hess_builder = (obj_arg) -> cctx_callback

t0 = time()
K_hard, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(aug.obj_cm, θ_full;
    hess_cb_builder = hess_builder, hvp = false)
t_archc = time() - t0
@printf(">>> Architecture C: nStatus=%d  t=%.3fs  n_fg=%d  n_hess=%d  speedup_vs_dense=%.2fx\n",
        nStatus, t_archc, n_fg, n_hess, t_dense / t_archc)
flush(stdout)

if nStatus in (0, -100, -101, -103)
    Wd = size(ctx.U, 1)
    K = zeros(Wd); G = zeros(Wd, aug.obj_cm.d)
    aug.obj_cm.moments!(K, G, θ_full, ctx.U, aug.obj_cm)
    ncon = aug.obj_cm.d - aug.obj_cm.outer_constr_index + 2
    cbuf = zeros(ncon)
    aug.obj_cm(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual_archC = cbuf[1] / 1e10
    @printf(">>> Architecture C recheck: Delta_dual=%.8f  |diff vs dense|=%.2e\n",
            Delta_dual_archC, abs(Delta_dual_archC - r_dense.Delta_dual))
end

println(">>> D=20 CM smoke test at W=$W, L=$L: COMPLETE")
