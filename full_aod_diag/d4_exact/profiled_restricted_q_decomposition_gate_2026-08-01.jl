# ============================================================================
# Claude Code task 2026-08-01 (profiled-restricted-production-outer-bridge), §6:
# per-draw q-decomposition gate for the four restricted families' NEW
# restriction_contrib0! operators (profiled_restriction_contrib0_operators_
# 2026-08-01.jl), at a REAL, solved D4 production dual for each family.
#
# HONEST SCOPE (see that file's own header for the full derivation): a fully
# faithful §6 gate would compare against
#   -zeta - reduced_homogeneous_dual_contraction(beta_reduced, ...) - restriction_contrib0
# using the family's genuinely REDUCED (anchor-excluded) economic dual. That
# reduced economic dual does not exist for any restricted family in this
# worktree (only for the already-real unrestricted family, via
# build_profiled_operator_bundle) -- confirmed empirically (see this file's
# own PROBE output below) and via one permitted read of the sibling
# architecture/profiled-restricted-inner-endtoend-2026-08-01 worktree's
# uncommitted accessor file, which shows the reduced economic block requires
# NEW fields on CMBinHessCtx/OriginZCCoreHessCtx (`ncore_core`,
# `profiled_layout`, `econ_ctx`, `n_eta`) that do not exist in this worktree
# and that would require editing files on this task's own forbidden list
# (cm_hessian_architectures.jl's moments!/struct definitions) to build.
#
# So THIS gate instead validates restriction_contrib0! (the genuinely NEW
# piece §5 asks for) against the DENSE economic operator (`economic_forward!`,
# the SAME shared kernel every family's own real FG already calls for its E
# block -- not a hand-derived parallel formula) at a REAL solved dense dual:
#   q_recon[w] = -zeta* - economic_forward!(beta_dense_econ)[w] - restriction_contrib0[w]*SW[w]
# compared against q_truth[w] = the family's own real dual_index!/FG arg0,
# UNMODIFIED. This is a decisive, real (not mocked) validation of
# restriction_contrib0!'s correctness and SW convention; it does NOT by
# itself certify the full profiled/reduced five-accessor pipeline (§8), which
# remains blocked for the reason above. The CSV's `gate_type` column makes
# this distinction explicit in every row -- never conflated with a claim of
# full profiled/reduced equivalence.
# ============================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_config.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "production_bundle_api.jl", "dense_reference_diagnostics.jl",
          "economic_operator.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Random
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

rows = NamedTuple[]
function record!(family, point, gate_type, max_abs_q_err, max_rel_q_err, pass, note)
    push!(rows, (family = family, point = point, gate_type = gate_type,
                 max_abs_q_err = max_abs_q_err, max_rel_q_err = max_rel_q_err, pass = pass, note = note))
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(20260730)
probs4 = collect(range(0.0, 1.0; length = 12))[2:end-1]

println("="^90); println("§6 q-decomposition gate -- restriction_contrib0! validation (dense-economic reconstruction)"); println("="^90)

# =====================================================================================
# flexible CM
# =====================================================================================
println("--- flexible CM ---")
pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probs4,
    threaded_bins = true, inner_fg_backend = :cm_lookup, moment_representation = :operator)
cctx = pcx.cctx
base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx)
@printf("  inner_status=%d  zeta*=%.6g\n", base.inner_status, base.ζstar)
cf = cctx.core_cf_ref[]::CompressedFactual
ncore1 = cctx.NCORE - 1
β_dense = base.λstar[1:ncore1]
λ_cm = base.λstar[ncore1+1:ncore1+cctx.ncm]

bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
obj = pcx.ctx_cm.obj
st = CMLookupState(obj, cctx.NCORE, cctx.ncm, cctx.L, cctx.origins, cctx.refIndex1,
                   bins_u, cctx.R; method = :suffix, core_cf_ref = cctx.core_cf_ref)
x_full = vcat(base.ζstar, β_dense, λ_cm)
dual_index!(st, x_full)
q_truth = copy(st.arg0)

econ_ws = economic_operator_workspace(cf)
econ_buf = zeros(cf.W)
economic_forward!(econ_buf, β_dense, cf, econ_ws)

ws_flex = FlexCMRestrictionWorkspace(length(cctx.origins), cctx.L, cf.W)
rc0 = zeros(cf.W)
restriction_contrib0_flexcm!(rc0, λ_cm, bins_u, cctx.refIndex1, cctx.origins, cctx.R, cf.SW, ws_flex)
q_recon = -base.ζstar .- econ_buf .- (cf.SW .* rc0)
err = maximum(abs.(q_truth .- q_recon))
relerr = err / maximum(abs.(q_truth))
@printf("  max_abs_q_err=%.3e  max_rel_q_err=%.3e\n", err, relerr)
record!("flexible_CM", "D4_calibration", "restriction_block_only_dense_economic", err, relerr, err < 1e-8, "restriction_contrib0_flexcm! verified against real production FG")

# =====================================================================================
# common Frechet
# =====================================================================================
println("--- common Frechet ---")
pcx_f = build_cm_frechet_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probs4,
    cm_hessian_backend = :structured, moment_representation = :operator)
cctx_f = pcx_f.cctx
level_targets = pcx_f.aug.level_targets
base_f = archC_frechet_base_state(x_free_calib, pcx_f.ctx_cm, cctx_f, level_targets)
@printf("  inner_status=%d  zeta*=%.6g\n", base_f.inner_status, base_f.ζstar)
cf_f = cctx_f.core_cf_ref[]::CompressedFactual
ncore1_f = cctx_f.NCORE - 1
ncm_level = cctx_f.L
ncm_cm = cctx_f.ncm - ncm_level
β_dense_f = base_f.λstar[1:ncore1_f]
λ_cm_f = base_f.λstar[ncore1_f+1:ncore1_f+ncm_cm]
λ_level_f = base_f.λstar[ncore1_f+ncm_cm+1:ncore1_f+ncm_cm+ncm_level]

bins_uf = cctx_f.Bidx isa Matrix{UInt32} ? cctx_f.Bidx : Matrix{UInt32}(cctx_f.Bidx)
obj_f = pcx_f.ctx_cm.obj
st_f = CMFrechetLookupState(obj_f, cctx_f.NCORE, ncm_cm, cctx_f.L, cctx_f.L, cctx_f.D,
                             cctx_f.origins, cctx_f.refIndex1, bins_uf, cctx_f.R, level_targets;
                             core_cf_ref = cctx_f.core_cf_ref)
x_full_f = vcat(base_f.ζstar, β_dense_f, λ_cm_f, λ_level_f)
dual_index!(st_f, x_full_f)
q_truth_f = copy(st_f.arg0)

econ_ws_f = economic_operator_workspace(cf_f)
econ_buf_f = zeros(cf_f.W)
economic_forward!(econ_buf_f, β_dense_f, cf_f, econ_ws_f)

ws_frechet = FrechetRestrictionWorkspace(length(cctx_f.origins), cctx_f.L, cf_f.W)
rc0_f = zeros(cf_f.W)
restriction_contrib0_frechet!(rc0_f, λ_cm_f, λ_level_f, bins_uf, cctx_f.refIndex1, cctx_f.origins,
                               cctx_f.R, cctx_f.D, level_targets, cf_f.SW, ws_frechet)
q_recon_f = -base_f.ζstar .- econ_buf_f .- (cf_f.SW .* rc0_f)
err_f = maximum(abs.(q_truth_f .- q_recon_f))
relerr_f = err_f / maximum(abs.(q_truth_f))
@printf("  max_abs_q_err=%.3e  max_rel_q_err=%.3e\n", err_f, relerr_f)
record!("common_Frechet", "D4_calibration", "restriction_block_only_dense_economic", err_f, relerr_f, err_f < 1e-8, "restriction_contrib0_frechet! verified against real production FG")

# =====================================================================================
# ZC-only (origin-ZC)
# =====================================================================================
println("--- ZC-only (origin-ZC) ---")
layout_oz = OriginByPowerLayout(ctx.D, 1, 0)
pcx_oz = build_originzc_production_context(ctx, CS, layout_oz; moment_representation = :operator)
νfull_oz = nu0_origin(1, ctx.D)
base_oz = archOZ_base_state(x_free_calib, νfull_oz, pcx_oz.ctx_cm)
@printf("  inner_status=%d  zeta*=%.6g\n", base_oz.inner_status, base_oz.ζstar)
octx = pcx_oz.octx
cf_oz = octx.core_cf_ref[]::CompressedFactual
ncore1_oz = octx.NCORE - 1
zc_op = octx.fg_zc_op
zc_layout = octx.fg_layout
n_mean_oz = n_mean(zc_op); n_pair_oz = n_pair(zc_op)
β_dense_oz = base_oz.λstar[1:ncore1_oz]
λ_mean_oz = base_oz.λstar[ncore1_oz+1:ncore1_oz+n_mean_oz]
λ_pair_oz = base_oz.λstar[ncore1_oz+n_mean_oz+1:ncore1_oz+n_mean_oz+n_pair_oz]

obj_oz = pcx_oz.ctx_cm.obj
zc_ws_probe = ZCRestrictionWorkspace(zc_op)
refresh_zc_targets!(zc_ws_probe, zc_op, zc_layout, νfull_oz)
ov_oz = verify_inner_solution_operator_originzc!(base_oz.ζstar, base_oz.λstar, cf_oz, zc_op, zc_layout, νfull_oz, obj_oz, cf_oz.W)
q_truth_oz = ov_oz.r

econ_ws_oz = economic_operator_workspace(cf_oz)
econ_buf_oz = zeros(cf_oz.W)
economic_forward!(econ_buf_oz, β_dense_oz, cf_oz, econ_ws_oz)

rc0_oz = zeros(cf_oz.W)
restriction_contrib0_originzc!(rc0_oz, λ_mean_oz, λ_pair_oz, zc_op, zc_ws_probe, cf_oz.SW)
q_recon_oz = -base_oz.ζstar .- econ_buf_oz .- (cf_oz.SW .* rc0_oz)
err_oz = maximum(abs.(q_truth_oz .- q_recon_oz))
relerr_oz = err_oz / maximum(abs.(q_truth_oz))
@printf("  max_abs_q_err=%.3e  max_rel_q_err=%.3e\n", err_oz, relerr_oz)
record!("ZC_only", "D4_calibration", "restriction_block_only_dense_economic", err_oz, relerr_oz, err_oz < 1e-8, "restriction_contrib0_originzc! verified against real production FG")

# =====================================================================================
# CM+ZC
# =====================================================================================
println("--- CM+ZC ---")
pcx_mz = build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = 1, K_pair = 0,
    contrasts = :anchored, meanzc_basis = :direct, probs = probs4, moment_representation = :operator)
cctx_mz = pcx_mz.cctx
νvec0 = [1.0]
base_mz = archC_meanzc_base_state(x_free_calib, νvec0, pcx_mz.ctx_cm, cctx_mz)
@printf("  inner_status=%d  zeta*=%.6g\n", base_mz.inner_status, base_mz.ζstar)
@printf("  cctx_mz.NCORE=%d  ncm=%d  (widened-core check: ncore_core field present? %s)\n",
        cctx_mz.NCORE, cctx_mz.ncm, hasproperty(cctx_mz, :ncore_core))
cf_mz = cctx_mz.core_cf_ref[]::CompressedFactual
zc_op_mz = cctx_mz.hzz_zc_op
zc_layout_mz = cctx_mz.hzz_zc_layout
n_mean_mz = n_mean(zc_op_mz); n_pair_mz = n_pair(zc_op_mz)
# IMPORTANT (widened-core finding, task's own §8 CM+ZC caveat, confirmed live 2026-08-01): the
# NAIVE split "economic = 1:(cctx.NCORE-1), CM grid = NCORE:NCORE+ncm-1, Z mean/pair appended
# after" is WRONG for CM+ZC -- cctx_mz.NCORE (22) already WIDENS the "core" to INCLUDE the Z
# mean/pair columns (cctx_mz.NCORE-1 = 21 = cf.oci-1 (17, true economic width) + n_mean (4) +
# n_pair (0)), confirmed by a direct BoundsError when the naive split was tried first (λstar has
# only 51 = (NCORE-1)+ncm entries, not (NCORE-1)+ncm+n_mean+n_pair). The CORRECT split -- taken
# directly from the REAL, already-validated production verifier
# `verify_inner_solution_operator_cmmeanzc!` (operator_verification.jl), not re-derived -- is
# `[economic(cf.oci-1) | Z_mean(n_mean) | Z_pair(n_pair) | CM_grid(ncm)]`, i.e. the economic
# boundary is `cf.oci-1`, NOT `cctx.NCORE-1`. This split is unambiguous for the RESTRICTION-BLOCK
# diagnostic below (both blocks are directly readable off cf.oci/n_mean/n_pair); it does NOT by
# itself resolve the harder problem the master task flagged for §8 (a genuine
# `economic_dual_range`/`restriction_dual_ranges` UnitRange split against a REDUCED economic
# layout, where cctx.NCORE itself no longer cleanly separates "the economic block" from "the
# widened core" at the TYPE/CONTRACT level without extra structural knowledge like this) -- that
# remains blocked, see the CSV's full_profiled_reduced_economic_dual row for CM_plus_ZC.
ncore1_mz = cf_mz.oci - 1
β_dense_mz = base_mz.λstar[1:ncore1_mz]
λ_mean_mz = base_mz.λstar[ncore1_mz+1:ncore1_mz+n_mean_mz]
λ_pair_mz = base_mz.λstar[ncore1_mz+n_mean_mz+1:ncore1_mz+n_mean_mz+n_pair_mz]
λ_cm_mz = base_mz.λstar[ncore1_mz+n_mean_mz+n_pair_mz+1:ncore1_mz+n_mean_mz+n_pair_mz+cctx_mz.ncm]

bins_umz = cctx_mz.Bidx isa Matrix{UInt32} ? cctx_mz.Bidx : Matrix{UInt32}(cctx_mz.Bidx)
zc_ws_mz_probe = ZCRestrictionWorkspace(zc_op_mz)
refresh_zc_targets!(zc_ws_mz_probe, zc_op_mz, zc_layout_mz, νvec0)
obj_mz = pcx_mz.ctx_cm.obj
ov_mz = verify_inner_solution_operator_cmmeanzc!(base_mz.ζstar, base_mz.λstar, cf_mz, zc_op_mz, zc_layout_mz, νvec0,
    cctx_mz.L, length(cctx_mz.origins), cctx_mz.origins, cctx_mz.refIndex1, bins_umz, cctx_mz.R, obj_mz, cf_mz.W)
q_truth_mz = ov_mz.r

econ_ws_mz = economic_operator_workspace(cf_mz)
econ_buf_mz = zeros(cf_mz.W)
economic_forward!(econ_buf_mz, β_dense_mz, cf_mz, econ_ws_mz)

ws_cm_mz = FlexCMRestrictionWorkspace(length(cctx_mz.origins), cctx_mz.L, cf_mz.W)
rc0_mz = zeros(cf_mz.W)
restriction_contrib0_cmzc!(rc0_mz, λ_cm_mz, λ_mean_mz, λ_pair_mz, bins_umz, cctx_mz.refIndex1, cctx_mz.origins,
                            cctx_mz.R, zc_op_mz, zc_ws_mz_probe, cf_mz.SW, ws_cm_mz)
q_recon_mz = -base_mz.ζstar .- econ_buf_mz .- (cf_mz.SW .* rc0_mz)
err_mz = maximum(abs.(q_truth_mz .- q_recon_mz))
relerr_mz = err_mz / maximum(abs.(q_truth_mz))
@printf("  max_abs_q_err=%.3e  max_rel_q_err=%.3e\n", err_mz, relerr_mz)
record!("CM_plus_ZC", "D4_calibration", "restriction_block_only_dense_economic", err_mz, relerr_mz, err_mz < 1e-8,
        "restriction_contrib0_cmzc! verified against real production FG using the CORRECT [economic(cf.oci-1)|Z_mean|Z_pair|CM_grid] split taken directly from verify_inner_solution_operator_cmmeanzc! (NOT cctx.NCORE-based, which is widened and would conflate economic+Z -- confirmed live via BoundsError before this fix). This resolves the restriction-block-only diagnostic but does NOT resolve full economic_dual_range/restriction_dual_ranges as UnitRanges against a reduced economic layout for the five-accessor contract -- see full_profiled_reduced_economic_dual row.")

# =====================================================================================
# Blocked rows: the FULL profiled/reduced gate (reduced economic dual vs real FG), per family
# =====================================================================================
for fam in ("flexible_CM", "common_Frechet", "ZC_only", "CM_plus_ZC")
    record!(fam, "D4_calibration", "full_profiled_reduced_economic_dual", NaN, NaN, false,
            "BLOCKED: no reduced/pivoted inner-solve machinery for this family exists in this worktree (cctx.NCORE-1 == cf.oci-1, the DENSE width, not total_reduced_economic_moments -- confirmed via probe_ncore_vs_reduced_2026-08-01.jl). A closed-form gauge-shift transform from dense to reduced beta was tried and empirically failed (residual ~0.81 at D4, not machine precision). The reduced-economic machinery is separately under construction (uncommitted) in the sibling architecture/profiled-restricted-inner-endtoend-2026-08-01 worktree.")
end

out_path = joinpath(_D4E, "..", "..", "PROFILED_RESTRICTED_Q_DECOMPOSITION_GATE_2026-08-01.csv")
open(out_path, "w") do io
    println(io, "family,point,gate_type,max_abs_q_err,max_rel_q_err,pass,note")
    for r in rows
        note_escaped = replace(r.note, "\"" => "'")
        println(io, "$(r.family),$(r.point),$(r.gate_type),$(r.max_abs_q_err),$(r.max_rel_q_err),$(r.pass),\"$(note_escaped)\"")
    end
end
println("="^90)
println("Wrote ", abspath(out_path))
for r in rows
    @printf("%-16s %-16s %-38s max_abs=%.3e max_rel=%.3e pass=%s\n", r.family, r.point, r.gate_type, r.max_abs_q_err, r.max_rel_q_err, r.pass)
end
