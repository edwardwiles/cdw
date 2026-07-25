# ============================================================================
# Rectangular-layout (D != D_dest) dimension-safety gate (task brief §2/§11 --
# "square and rectangular layouts"), using REAL D=20 data at a small W for
# speed (this is a correctness/dimension smoke test, not a calibration-
# accuracy or performance benchmark -- W=200 is intentionally tiny).
#
# There is no synthetic "D=4, D_dest=3" economy generator in production
# (d4_exact_setup is square-only); the real omit-ROW production path is
# exactly the D=20/D_dest=19 rectangular case the task brief is worried
# about, so this gate exercises THAT directly at small W rather than
# constructing a synthetic rectangular toy.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
using Printf, LinearAlgebra

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

println("="^100); println("SETUP: real D=20 data, W=4000 (dimension smoke only), destination_sample=:exclude_row"); println("="^100)
# W=200 (first attempt) produced ncm=200 vs W=200 draws -- a degenerate/rank-deficient CC-dual
# design (as many restriction columns as draws), giving nStatus=-300 (unbounded) at the real
# calibration point. This is the well-known small-W numerical-finickiness this codebase's own
# memory documents (rank deficiency resolves at larger W); NOT a sign of a moment/Hessian bug --
# the dimension-safety assertions (GATE R1, ncm=D*L not D_dest*L) already passed cleanly at W=200.
# W=4000 restores a healthy draws:restrictions ratio while staying a fast smoke test.
ctx = d20_real_setup_design(W = 4000, δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
println("ctx.D=$(ctx.D)  ctx.D_dest=$(ctx.D_dest)  ctx.row_idx=$(ctx.row_idx)  ctx.destination_sample=$(ctx.destination_sample)")

check("ctx.D == 20 (origin count, invariant)", ctx.D == 20)
check("ctx.D_dest == 19 (destination count, exclude_row)", ctx.D_dest == 19)
check("ctx.D != ctx.D_dest (genuinely rectangular)", ctx.D != ctx.D_dest)
check("size(ctx.U,2) == ctx.D == 20 (NOT ctx.D_dest == 19)", size(ctx.U, 2) == ctx.D)

const L = 5   # small grid, this is a dimension smoke test not a calibration/perf benchmark
cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
targets = build_frechet_reference_targets(ctx, cfg; L = L)
report_frechet_targets(ctx, cfg, targets)
check("targets.D == 20 (origin count)", targets.D == 20)
check("length(targets.probs) == L == 5", length(targets.probs) == L)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
println("n_free = ", length(x_free_calib), "  (expect 1 + D*D_dest = ", 1 + ctx.D*ctx.D_dest, ")")
check("n_free == 1 + D*D_dest (core free-parameter count reflects RECTANGULAR 20x19, not 20x20)",
      length(x_free_calib) == 1 + ctx.D * ctx.D_dest)

println("="^100); println("GATE R1: build combined CDF+POWER production context against the rectangular core"); println("="^100)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
ncore = fpcx.aug.ncore
ncm = fpcx.aug.ncm
println("ncore=$ncore  ncm=$ncm  (expect ncm == 2*D*L = ", 2*ctx.D*L, ", NOT 2*D_dest*L = ", 2*ctx.D_dest*L, ")")
check("ncm == 2*D*L (origin-indexed, not destination-indexed)", ncm == 2 * ctx.D * L)
check("ncm != 2*D_dest*L (would be the dimension-trap value)", ncm != 2 * ctx.D_dest * L)
check("fpcx.ctx_cm.obj.d == ncore + ncm", fpcx.ctx_cm.obj.d == ncore + ncm)
check("fpcx.ctx_cm.obj.outer_constr_index == fpcx.ctx_cm.obj.d", fpcx.ctx_cm.obj.outer_constr_index == fpcx.ctx_cm.obj.d)
check("fpcx.ctx_cm.D_dest == 19 (rectangular layout preserved through ctx_cm = merge(ctx,...))",
      hasproperty(fpcx.ctx_cm, :D_dest) && fpcx.ctx_cm.D_dest == 19)

println("="^100); println("GATE R2: base-state solve + dense-vs-structured Hessian agreement at the rectangular point"); println("="^100)
base = cm_frechet_base_state(x_free_calib, fpcx)
check("base-state solve feasible", frechet_solve_outcome(base.nStatus) == :feasible)

n_inner = ncore + ncm
nh = div(n_inner * (n_inner + 1), 2)
h_struct = zeros(nh)
_archC_prep_for_hessian!(fpcx.ctx_cm.obj, vcat(base.ζstar, base.λstar))
hessian_cm_frechet_cdf_power_structured!(h_struct, fpcx.ctx_cm.obj, fpcx.fctx)

# dense cross-check via the obj functor's own generic Hessian, at the same solved point
K2, x2, ns2, _, _ = inner_loop_internal_archgeneric(fpcx.ctx_cm.obj, θ_full0; hess_cb_builder = archA_hess_cb_builder)
check("dense re-solve feasible", frechet_solve_outcome(ns2) == :feasible)
h_dense = zeros(nh)
fpcx.ctx_cm.obj(x2, h = h_dense)
_archC_prep_for_hessian!(fpcx.ctx_cm.obj, x2)
h_struct2 = zeros(nh)
hessian_cm_frechet_cdf_power_structured!(h_struct2, fpcx.ctx_cm.obj, fpcx.fctx)
maxerr = maximum(abs.(h_dense .- h_struct2))
println("max|H_dense - H_struct| at rectangular D=20/D_dest=19 point = $maxerr")
check("dense vs structured Hessian match at rectangular layout (1e-6 abs, small-W noise floor)", maxerr < 1e-6)

println()
println("="^100)
@printf "TOTAL: %d PASS, %d FAIL\n" n_pass n_fail
println("="^100)
exit(n_fail == 0 ? 0 : 1)
