# Task brief Section 9: short D=20 outer-loop trial (shakedown, NOT a production result). Only
# runs if Section 8's fixed-point gates passed (they did, after the -zeta bugfix -- nesting holds
# cleanly at all 4 tested points). D=20, W=80000, L=50, delta=1, upper-kappa direction
# (find_smallest=true), 20 Julia threads. Two matched economic starts per arm (calibration +
# the existing cold-verified L50 incumbent), 15-minute cap PER (arm,start) combination.
# Include order matches test_cm_checkpoint_original.jl's own known-good order EXACTLY (draw_design.jl
# FIRST, no separate context_real_d20.jl -- draw_design.jl brings in its own equivalent setup
# internally at the right point). Reordering this caused a real "two or more modules export
# different bindings" ambiguity error for PsiObjectiveBundleDelta (found live) -- draw_design.jl's
# QMC context path (qmc_context_real_d20.jl) apparently conflicts with context_real_d20.jl if both
# get loaded/reachable in the wrong relative order.
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
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "mean_zero_cov_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "c40_meanzc_checkpoint.jl"))
using Printf, Dates, Serialization, LinearAlgebra

const W = 80000
const DELTA = 1.0
const L = 50
const DRAW_SEED = 20260719
const MAXTIME = 900.0   # 15-minute cap per (arm,start), per task brief Section 9
const NTHREADS = Threads.nthreads()
println("Section 9 start: ", now(), "  nthreads=", NTHREADS); flush(stdout)

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :sobol_randomized, draw_seed = DRAW_SEED)
pe = build_pivot_elimination(ctx0)
D = ctx0.D
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
z0 = log.(reshape(x_free_calib[2:end], D, D))
w0_calib = vcat(x_free_calib[1], pivot_reduce(z0, pe))

path_incumbent = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation", "stage_L50_latest.jls")
d_inc = deserialize(path_incumbent)
@assert length(d_inc.best_w) == length(w0_calib)
w0_incumbent = d_inc.best_w

STARTS = [("calibration", w0_calib), ("existing_incumbent", w0_incumbent)]
nu_lo, nu_hi = log.(nu_feasible_interval(ctx0.U))

probs = collect(range(1 / L, (L - 1) / L, length = L))   # :equal grid convention (cm_config.jl's
    # cm_equal_grid_probs / precalc_common_marginals_cdf's own probs===nothing default) -- inlined
    # here rather than including cm_config.jl, which has an unrelated pre-existing multi-docstring
    # parse bug under Julia 1.12 (found while building Section 7; cm_config.jl/CMEvalKey is not
    # needed by anything in this file)

ckpt_root = joinpath(@__DIR__, "..", "..", "results", "experiment_cm_pairwise_zero_cov", "section9_checkpoints")
mkpath(ckpt_root)
results = NamedTuple[]

"Local eta_nu profile at a given (raw) start, moderate band, used only to pick eta_nu0 -- mirrors Section 7/8's identical rationale."
function profile_eta0(ctx, x_free0; ngrid = 5, nrefine = 4, lo = -1.0, hi = 1.0, cm_extension, basis = :direct)
    nu_ref = Ref(1.0)
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = cm_extension, meanzc_basis = basis, probs = probs, nu_ref = nu_ref)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    function safe_eval(η)
        nu_ref[] = exp(η)
        try
            base = archC_base_state(x_free0, ctx_cm, cctx)
            return delta_dual_from_base(ctx_cm.obj, base)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            return Inf
        end
    end
    ηs = range(lo, hi, length = ngrid)
    vals = [safe_eval(η) for η in ηs]
    all(isinf, vals) && return 0.0
    i = argmin(vals); ηb = ηs[i]; vb = vals[i]
    lo2 = max(lo, ηb - step(ηs)); hi2 = min(hi, ηb + step(ηs))
    for η in range(lo2, hi2, length = nrefine)
        v = safe_eval(η)
        if v < vb; vb = v; ηb = η; end
    end
    return ηb
end

for (sname, w0econ) in STARTS
    println("\n", "="^100, "\nSTART=$sname\n", "="^100); flush(stdout)

    # ---- CM only (existing production checkpointed driver, unmodified) ----
    label_cm = "cmonly_$(sname)"
    t0 = time()
    out_cm = run_cm_upper_checkpointed(copy(w0econ); W = W, delta = DELTA, draw_design = :sobol_randomized,
        draw_seed = DRAW_SEED, L = L, contrasts = :anchored, probs = probs,
        maxtime_real = MAXTIME, ckpt_dir = joinpath(ckpt_root, "cm_only"), run_id = string(now()),
        label = label_cm, checkpoint_interval_s = 60.0, verbose = true)
    wall_cm = time() - t0
    @printf "[cm_only/%s] kappa=%.6f n_eval=%d n_grad=%d wall=%.1fs status=%d\n" sname out_cm.kappa out_cm.n_eval out_cm.n_grad wall_cm out_cm.knitro_status
    push!(results, (arm = "cm_only", start = sname, kappa = out_cm.kappa, n_eval = out_cm.n_eval,
        n_grad = out_cm.n_grad, wall = wall_cm, knitro_status = out_cm.knitro_status,
        best_gp = out_cm.best === nothing ? NaN : out_cm.best.gp,
        best_Delta = out_cm.best === nothing ? NaN : out_cm.best.Delta, nu = NaN))

    # ---- checkpoint/resume smoke test (cm_only, cheap: reuse its own final checkpoint) ----
    resumed_cm = load_cm_checkpoint(out_cm.ckpt_path)
    @printf "[cm_only/%s] checkpoint/resume smoke test: reloaded schema=%d n_eval=%d reason=%s\n" sname resumed_cm.schema resumed_cm.n_eval resumed_cm.checkpoint_reason

    for arm in (:cm_plus_mean, :cm_plus_mean_zero_covariance)
        η0 = profile_eta0(ctx0, x_free_from_w(w0econ, pe); cm_extension = arm)
        w0 = vcat(w0econ, η0)
        label = "$(arm)_$(sname)"
        @printf "[%s/%s] initial local eta_nu profile: eta0=%.4f (nu0=%.4f)\n" arm sname η0 exp(η0)

        t0 = time()
        out = run_meanzc_upper_checkpointed(copy(w0); W = W, delta = DELTA, draw_design = :sobol_randomized,
            draw_seed = DRAW_SEED, L = L, cm_extension = arm, meanzc_basis = :direct, contrasts = :anchored,
            probs = probs, maxtime_real = MAXTIME, eta_nu_bounds = (nu_lo, nu_hi),
            ckpt_dir = joinpath(ckpt_root, string(arm)), run_id = string(now()), label = label,
            checkpoint_interval_s = 60.0, verbose = true)
        wall = time() - t0
        @printf "[%s/%s] kappa=%.6f n_eval=%d n_grad=%d wall=%.1fs status=%d\n" arm sname out.kappa out.n_eval out.n_grad wall out.knitro_status
        push!(results, (arm = string(arm), start = sname, kappa = out.kappa, n_eval = out.n_eval,
            n_grad = out.n_grad, wall = wall, knitro_status = out.knitro_status,
            best_gp = out.best === nothing ? NaN : out.best.gp,
            best_Delta = out.best === nothing ? NaN : out.best.Delta,
            nu = out.best === nothing ? NaN : out.best.nu))

        resumed = load_cm_meanzc_checkpoint(out.ckpt_path)
        @printf "[%s/%s] checkpoint/resume smoke test: reloaded schema=%d n_eval=%d reason=%s eta_nu=%.4f\n" arm sname resumed.schema resumed.n_eval resumed.checkpoint_reason resumed.eta_nu
    end
end

out_dir = joinpath(@__DIR__, "..", "..", "results", "experiment_cm_pairwise_zero_cov")
mkpath(out_dir)
csv_path = joinpath(out_dir, "section9_d20_outer_trial.csv")
cols = [:arm, :start, :kappa, :n_eval, :n_grad, :wall, :knitro_status, :best_gp, :best_Delta, :nu]
open(csv_path, "w") do io
    println(io, join(String.(cols), ","))
    for r in results
        println(io, join([string(getfield(r, c)) for c in cols], ","))
    end
end
println("\nWrote $csv_path")
println("Section 9 done: ", now())
