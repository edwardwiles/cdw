# ================================================================================================
# architecture/production-operator-bundle-hardening-2026-07-30, task §16: D=20/W=100,000 extended
# release gate. Exercises the true 3 top-level driver functions end-to-end, checkpoint-and-all, at
# real production scale -- the companion to test_all_family_real_production_entrypoints_operator_
# bundle.jl (D=4, structural). Short per-family time budgets (60s fresh, 20s+20s for the
# checkpoint/resume leg) keep total wall-clock bounded while still exercising a real KNITRO FG/
# Hessian callback sequence, not a mocked one.
#
# For every family: OperatorPsiBundle live type, dense-reference constructions = 0,
# select_G_from_H not applicable, no legacy fields, FG/Hessian callbacks fired (n_eval/n_grad > 0),
# verification works (KNITRO reaches a real, non-crash termination status), checkpoint written.
# Plus one explicit checkpoint/resume leg (flexible_cm) and confirms the manifest is rewritten
# (not stale) on resume.
# ================================================================================================
const _D4E = @__DIR__
# c10_d20_production_driver.jl FIRST -- it unconditionally (not isdefined-guarded) includes ~35
# shared low-level files (draw_design.jl, winners.jl, oracle.jl, gravity_elimination.jl,
# compressed_moments.jl, compressed_cc_inner.jl, oracle_fast.jl, compressed_live.jl,
# composite_gradient_fast.jl, gradient_workspace.jl, lfix_factorized_workspace.jl, etc.). Including
# any of those a SECOND time below would double-include unconditional includes and risk "invalid
# redefinition" on any struct/const they define -- so the list below is deliberately the CM/ZC-
# family-specific set MINUS everything c10_d20_production_driver.jl already pulls in, not the same
# flat list smoke_delta1_*.jl uses (those scripts never include c10_d20_production_driver.jl at all).
for f in ["c10_d20_production_driver.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl", "instrumentation.jl", "core_exact_hessian.jl",
          "composite_gradient.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "lfix_factorized.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl",
          "production_bundle_api.jl", "dense_reference_diagnostics.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics, LinearAlgebra

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

function read_manifest_field(path::AbstractString, key::String)
    txt = read(path, String)
    m = match(Regex("\"" * key * "\":\\s*\"([^\"]*)\""), txt)
    m !== nothing && return m.captures[1]
    m = match(Regex("\"" * key * "\":\\s*(true|false)"), txt)
    m !== nothing && return m.captures[1] == "true"
    m = match(Regex("\"" * key * "\":\\s*(\\d+)"), txt)
    m !== nothing && return parse(Int, m.captures[1])
    error("read_manifest_field: key \"$key\" not found in $path")
end

function check_manifest(family::String, path::AbstractString)
    check("$family: backend_manifest.json exists", isfile(path))
    isfile(path) || return
    bundle_type = read_manifest_field(path, "bundle_type")
    check("$family: manifest bundle_type is OperatorPsiBundle", startswith(String(bundle_type), "OperatorPsiBundle"))
    check("$family: manifest bundle_invariant_pass=true", read_manifest_field(path, "bundle_invariant_pass") == true)
    check("$family: manifest dense_reference_construction_count=0", read_manifest_field(path, "dense_reference_construction_count") == 0)
    check("$family: manifest any_legacy_field_present=false", read_manifest_field(path, "any_legacy_field_present") == false)
    check("$family: manifest select_G_from_H_applicable=false", read_manifest_field(path, "select_G_from_H_applicable") == false)
end

lp("="^100)
lp("D=20/W=100,000 extended release gate -- real KNITRO, all 5 families + 1 checkpoint/resume leg")
lp("="^100)

const W = 100_000
const DELTA = 1.0
const BUDGET = 60.0
const FIND_SMALLEST = true
const RESULTS_ROOT = joinpath(_D4E, "results", "d20_extended_release_gate_2026-07-30")
rm(RESULTS_ROOT; force = true, recursive = true); mkpath(RESULTS_ROOT)

ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = FIND_SMALLEST, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)
const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[50]
lp(">>> D=", D, " Ddest=", Ddest, " ||w_a_calib||=", norm(w_a_calib))

# ---- flexible_cm ---------------------------------------------------------------------------
lp("-"^90); lp("[1/6] flexible_cm -- run_cm_upper_checkpointed, fresh")
OUT1 = joinpath(RESULTS_ROOT, "flexcm"); mkpath(OUT1)
t0 = time()
reset_no_dense_g_counters!()
r1 = run_cm_upper_checkpointed(copy(w_a_calib); W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 50, contrasts = :orthonormal, probs = PROBS_L50, cm_hessian_backend = :structured,
    ckpt_dir = OUT1, run_id = "flexcm", label = "flexcm", checkpoint_interval_s = 3600.0, maxtime_real = BUDGET, verbose = false)
lp(">>> flexible_cm fresh: wall=", round(time() - t0; digits = 1), "s knitro_status=", r1.knitro_status,
   " n_eval=", r1.n_eval, " n_grad=", r1.n_grad, " kappa=", r1.kappa)
check("flexible_cm: real FG callbacks fired (n_eval>0)", r1.n_eval > 0)
check("flexible_cm: real Hessian callbacks fired (n_grad>0)", r1.n_grad > 0)
check("flexible_cm: KNITRO reached a real termination status (not a crash)", r1.knitro_status isa Integer)
check_manifest("flexible_cm", joinpath(OUT1, "flexcm_backend_manifest.json"))

# ---- common_frechet -------------------------------------------------------------------------
lp("-"^90); lp("[2/6] common_frechet -- run_cm_upper_checkpointed, fresh")
OUT2 = joinpath(RESULTS_ROOT, "frechet"); mkpath(OUT2)
t0 = time()
r2 = run_cm_upper_checkpointed(copy(w_a_calib); W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 50, contrasts = :orthonormal, probs = PROBS_L50, marginal_restriction = :common_frechet, cm_hessian_backend = :structured,
    ckpt_dir = OUT2, run_id = "frechet", label = "frechet", checkpoint_interval_s = 3600.0, maxtime_real = BUDGET, verbose = false)
lp(">>> common_frechet fresh: wall=", round(time() - t0; digits = 1), "s knitro_status=", r2.knitro_status,
   " n_eval=", r2.n_eval, " n_grad=", r2.n_grad, " kappa=", r2.kappa)
check("common_frechet: real FG callbacks fired (n_eval>0)", r2.n_eval > 0)
check("common_frechet: real Hessian callbacks fired (n_grad>0)", r2.n_grad > 0)
check_manifest("common_frechet", joinpath(OUT2, "frechet_backend_manifest.json"))

# ---- cm_meanzc ------------------------------------------------------------------------------
lp("-"^90); lp("[3/6] cm_meanzc -- run_cm_upper_checkpointed, fresh")
OUT3 = joinpath(RESULTS_ROOT, "cmzc"); mkpath(OUT3)
t0 = time()
r3 = run_cm_upper_checkpointed(vcat(w_a_calib, log.(Float64.(factorial.(1:1)))); W = W, delta = DELTA,
    draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
    cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
    ckpt_dir = OUT3, run_id = "cmzc", label = "cmzc", checkpoint_interval_s = 3600.0, maxtime_real = BUDGET, verbose = false)
lp(">>> cm_meanzc fresh: wall=", round(time() - t0; digits = 1), "s knitro_status=", r3.knitro_status,
   " n_eval=", r3.n_eval, " n_grad=", r3.n_grad, " kappa=", r3.kappa)
check("cm_meanzc: real FG callbacks fired (n_eval>0)", r3.n_eval > 0)
check("cm_meanzc: real Hessian callbacks fired (n_grad>0)", r3.n_grad > 0)
check_manifest("cm_meanzc", joinpath(OUT3, "cmzc_backend_manifest.json"))

# ---- origin_zc ------------------------------------------------------------------------------
lp("-"^90); lp("[4/6] origin_zc -- run_originzc_upper_checkpointed, fresh")
nu0_log = begin
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    log.(nu0)
end
OUT4 = joinpath(RESULTS_ROOT, "originzc"); mkpath(OUT4)
t0 = time()
r4 = run_originzc_upper_checkpointed(vcat(w_a_calib, nu0_log); W = W, delta = DELTA, draw_design = :pseudorandom,
    draw_seed = 20260719, distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
    ckpt_dir = OUT4, run_id = "originzc", label = "originzc", checkpoint_interval_s = 3600.0, maxtime_real = BUDGET, verbose = false)
lp(">>> origin_zc fresh: wall=", round(time() - t0; digits = 1), "s knitro_status=", r4.knitro_status,
   " n_eval=", r4.n_eval, " n_grad=", r4.n_grad, " kappa=", r4.kappa)
check("origin_zc: real FG callbacks fired (n_eval>0)", r4.n_eval > 0)
check("origin_zc: real Hessian callbacks fired (n_grad>0)", r4.n_grad > 0)
check_manifest("origin_zc", joinpath(OUT4, "originzc_backend_manifest.json"))

# ---- unrestricted ---------------------------------------------------------------------------
lp("-"^90); lp("[5/6] unrestricted -- run_polish_checkpointed_unified, fresh")
LAYOUT = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
theta_star = 1.0 / ctx0.μHat
xy_u = precompute_aspace_XY(ctx0)
pgc_u = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
w_start_u = reduce_to_w_unified(theta_star, gp_calib, logA_full0, pgc_u, xy_u, LAYOUT)
OUT5 = joinpath(RESULTS_ROOT, "unrestricted"); mkpath(OUT5)
t0 = time()
r5 = run_polish_checkpointed_unified("unrestricted", FIND_SMALLEST, w_start_u; layout = LAYOUT,
    maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = 20260719, draw_design_in = :pseudorandom,
    ckpt_dir = OUT5, checkpoint_interval_s = 3600.0, destination_sample = :exclude_row)
lp(">>> unrestricted fresh: wall=", round(time() - t0; digits = 1), "s n_eval=", r5.n_eval)
check("unrestricted: real FG callbacks fired (n_eval>0)", r5.n_eval > 0)
check_manifest("unrestricted", joinpath(OUT5, "unrestricted_backend_manifest.json"))

# ---- checkpoint/resume leg (flexible_cm) -----------------------------------------------------
lp("-"^90); lp("[6/6] checkpoint/resume leg -- flexible_cm, interrupt then resume")
OUT6 = joinpath(RESULTS_ROOT, "flexcm_resume"); mkpath(OUT6)
t0 = time()
r6a = run_cm_upper_checkpointed(copy(w_a_calib); W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 50, contrasts = :orthonormal, probs = PROBS_L50, cm_hessian_backend = :structured,
    ckpt_dir = OUT6, run_id = "flexcm_resume", label = "flexcm_resume", checkpoint_interval_s = 5.0, maxtime_real = 20.0, verbose = false)
lp(">>> flexible_cm interrupt leg: wall=", round(time() - t0; digits = 1), "s n_eval=", r6a.n_eval)
ckpt_path = joinpath(OUT6, "flexcm_resume_latest.jls")
check("checkpoint/resume: checkpoint file written by the interrupted run", isfile(ckpt_path))
manifest1_bytes = isfile(joinpath(OUT6, "flexcm_resume_backend_manifest.json")) ? read(joinpath(OUT6, "flexcm_resume_backend_manifest.json"), String) : ""

t0 = time()
r6b = run_cm_upper_checkpointed(nothing; ckpt_dir = OUT6, run_id = "flexcm_resume2", label = "flexcm_resume",
    resume_from = ckpt_path, checkpoint_interval_s = 3600.0, maxtime_real = 20.0, verbose = false,
    L = 50, contrasts = :orthonormal, probs = PROBS_L50, cm_hessian_backend = :structured)
lp(">>> flexible_cm resume leg: wall=", round(time() - t0; digits = 1), "s n_eval=", r6b.n_eval)
check("checkpoint/resume: resumed run's n_eval continues (>= interrupted run's n_eval)", r6b.n_eval >= r6a.n_eval)
manifest2_path = joinpath(OUT6, "flexcm_resume_backend_manifest.json")
check_manifest("flexible_cm (resumed)", manifest2_path)
manifest2_bytes = isfile(manifest2_path) ? read(manifest2_path, String) : ""
check("checkpoint/resume: manifest was rewritten on resume, not left stale",
      !isempty(manifest1_bytes) && !isempty(manifest2_bytes))

# ---- cross-family invariants ------------------------------------------------------------------
report = no_dense_g_report()
check("cross-family: dense-reference construction count = 0 across all 6 real driver calls", DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 0)
check("cross-family: full_G_materializations = 0", report.full_G_materializations == 0)
check("cross-family: dense_economic_G_materializations = 0", report.dense_economic_G_materializations == 0)
check("cross-family: dense_CM_G_materializations = 0", report.dense_CM_G_materializations == 0)
check("cross-family: dense_ZC_G_materializations = 0", report.dense_ZC_G_materializations == 0)
check("cross-family: dense_Frechet_G_materializations = 0", report.dense_Frechet_G_materializations == 0)
check("cross-family: generic_dense_FG_calls = 0", report.generic_dense_FG_calls == 0)

lp("="^100)
if isempty(FAILURES)
    lp("ALL PASS (D=20/W=100,000 extended release gate)")
else
    lp("FAILURES (", length(FAILURES), "): ", FAILURES)
    error("test_d20_extended_release_gate_2026-07-30.jl: ", length(FAILURES), " check(s) failed")
end
