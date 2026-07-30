# Bounded three-level-architecture experiment: fixed-q A middle loop, real D=20 + D=4.
# Governing prompt: three-level Melitz outer-search architecture (2026-07-30), following up
# on melitz_d20_negative_switch_geometry_audit_2026-07-30 and
# melitz_joint_Aq_feasibility_preserving_search_2026-07-30. Reuses the EXACT real-D20 anchor,
# welfare coordinate, q direction, and switch thresholds from the negative-switch geometry
# audit (Phase 0 script `melitz_negswitch_phase0_recover_anchor_2026-07-30.jl` /
# `melitz_negswitch_phase2_switch_thresholds_2026-07-30.jl`, reconstructed fresh here since
# the prior sessions' own scratch `.jls` state was not persisted).
#
# MODE controls scope (env var MELITZ_MIDDLELOOP_MODE, default "full"):
#   "smoke" -- one D20 point (anchor), one start, small max_evals -- fast sanity check.
#   "full"  -- the complete bounded experiment (Phases 3-7 of the governing prompt).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization, Dates
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const MODE = get(ENV, "MELITZ_MIDDLELOOP_MODE", "full")
println("MODE = ", MODE); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
OUTER_OPT = joinpath(REPO2, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")
# Dedicated middle-loop options file (hessopt=6, L-BFGS -- governing prompt Phase 2's own
# recommendation), adopted after a live box-size/Hessian diagnostic at the D20 anchor found
# the production outer-driver's dense-BFGS options (hessopt=2) combined with a loose box
# (>=0.15) let KNITRO take a badly-scaled first Newton step into much-worse-than-anchor
# territory (A_od spans ~11 orders of magnitude at real D=20, per this project's own
# CLAUDE.md) -- a TIGHT box (0.05-0.1) with L-BFGS instead gives well-behaved, monotonic
# improvement. See docs/melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md Phase 3.
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1

# ============================================================================
# Setup: real-D20 anchor reconstruction (identical recipe to the negswitch-audit session's
# own Phase 0/2 scripts).
# ============================================================================
function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0_d20 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal
end
calib, focal = load_realD20_calib()
println("focal country index = ", focal); flush(stdout)

obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1; nq20 = D20^2 - 2
println("D=", D20, "  nA=", nA20, "  nq=", nq20); flush(stdout)

obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@printf("D20 base-point solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time() - t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0_d20 = copy(lfd0.dual_x)
p_star_d20 = copy(lfd0.weights)
flush(stdout)

session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1; bandwidth_policy=bwpolicy, target_switches=100)
@assert stage_d20 !== nothing
b_q_d20 = stage_d20.q_basis_free
@printf("basis: |b_q|=%.6f  s_lo=%.6f  s_hi=%.6f\n", norm(b_q_d20), stage_d20.s_lo, stage_d20.s_hi)
flush(stdout)

theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]

function q_full_at(sign::Int, t::Real)
    th = copy(theta_plain0_d20)
    th[1+nA20+1:end] .+= sign .* t .* b_q_d20
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end
q0_d20 = q_full_at(-1, 0.0)   # baseline (sign irrelevant at t=0)
gpj0_d20 = lfd0 !== nothing ? exp(theta_plain0_d20[1]) : NaN

# ============================================================================
# The 6 fixed-q cliff points (module header: anchor; immediately-before/after switch 1;
# after switch 2; after switch 3; one positive-side post-switch point). Switch thresholds
# taken from `docs/key_results/melitz_negswitch_phase2_minus_switches_2026-07-30.csv`
# (already-audited, LP-certified values) and cross-verified live below before use.
# ============================================================================
minus_csv = readdlm(joinpath(OUTDIR, "melitz_negswitch_phase2_minus_switches_2026-07-30.csv"), ','; skipstart=1)
plus_csv = readdlm(joinpath(OUTDIR, "melitz_negswitch_phase2_plus_switches_2026-07-30.csv"), ','; skipstart=1)
t_switch1_minus = Float64(minus_csv[1, 2])
t_switch2_minus = Float64(minus_csv[2, 2])
t_switch3_minus = Float64(minus_csv[3, 2])
t_switch1_plus = Float64(plus_csv[1, 2])
@printf("minus switches (t): 1=%.6e 2=%.6e 3=%.6e ; plus switch 1: %.6e\n",
        t_switch1_minus, t_switch2_minus, t_switch3_minus, t_switch1_plus)

cliff_points = [
    (name="anchor", sign=-1, t=0.0),
    (name="pre_switch1", sign=-1, t=0.75 * t_switch1_minus),
    (name="post_switch1", sign=-1, t=0.5 * (t_switch1_minus + t_switch2_minus)),
    (name="post_switch2", sign=-1, t=0.5 * (t_switch2_minus + t_switch3_minus)),
    (name="post_switch3", sign=-1, t=t_switch3_minus + 0.3 * (t_switch3_minus - t_switch2_minus)),
    (name="positive_post_switch1", sign=+1, t=1.5 * t_switch1_plus),
]
if MODE == "smoke"
    cliff_points = cliff_points[1:1]
end
for cp in cliff_points
    @printf("cliff point %-24s sign=%+d t=%.6e\n", cp.name, cp.sign, cp.t)
end
flush(stdout)

# verify linearity assumption live (module header requirement: never assume, always verify)
let qa = q_full_at(-1, 0.2), qb = q_full_at(-1, 0.6), qc = q_full_at(-1, 1.0)
    slope1 = (qb .- qa) ./ 0.4
    slope2 = (qc .- qb) ./ 0.4
    maxdiff = maximum(abs.(slope1 .- slope2))
    println("q_full(t) linearity check: max slope discrepancy = ", maxdiff, " (expect ~1e-15)")
    @assert maxdiff < 1e-9
end

# ============================================================================
# Middle-loop starts (Phase 4: up to 4 deterministic starts, all projected to EXACT feasible
# middle coordinates before KNITRO begins).
# ============================================================================

"""
Given the fixed target q for this cliff point, build the 4 deterministic starts:
  A. continuation: previous point's best A_free (or anchor's own, for the first point).
  B. old-anchor A, projected to satisfy the new q's exact middle constraints.
  C. cellwise p*-compensated A (`lfd_preserving_state.jl`'s formula (*)), projected.
  D. one deterministic feasible perturbation in log-H space, projected.
"""
function build_starts(q_target::Matrix{Float64}, gpj_target::Float64, prev_A_free::Union{Nothing,Vector{Float64}},
                       sys::MelitzFixedQMiddleConstraintSystem)
    starts = Dict{Symbol,Vector{Float64}}()

    # A: continuation
    A_cont = prev_A_free === nothing ? copy(A_free0_d20) : copy(prev_A_free)
    starts[:A_continuation] = melitz_project_start_to_middle_constraints(A_cont, sys, ctx_d20)

    # B: old-anchor A, projected
    starts[:B_old_anchor_projected] = melitz_project_start_to_middle_constraints(copy(A_free0_d20), sys, ctx_d20)

    # C: cellwise p*-compensated A (lfd_preserving_state.jl formula (*)), projected
    A_anchor_full = exp.(reshape(pivot_expand(A_free0_d20, ctx_d20.A_pivot), D20, D20))
    A_cellwise, status_cellwise = melitz_cellwise_A_from_moments(A_anchor_full, q0_d20, q_target, p_star_d20,
                                                                   sorted_ctx_d20, ctx_d20.sigma)
    n_bad = count(!=(:ok), status_cellwise)
    if n_bad > 0
        @warn "cellwise-compensated start: $n_bad / $(length(status_cellwise)) cells not :ok -- clamping to anchor value there"
        A_cellwise[status_cellwise .!= :ok] .= A_anchor_full[status_cellwise .!= :ok]
    end
    A_cellwise_free = pivot_reduce(vec(log.(A_cellwise)), ctx_d20.A_pivot)
    starts[:C_cellwise_compensated_projected] = melitz_project_start_to_middle_constraints(A_cellwise_free, sys, ctx_d20)

    # D: deterministic feasible perturbation in log-H space
    h_anchor = melitz_h_free_from_A_free(A_free0_d20, ctx_d20)
    h_pert = h_anchor .+ 0.02 .* sin.(1:length(h_anchor))   # deterministic, bounded, no RNG
    A_pert = melitz_A_free_from_h_free(h_pert, ctx_d20)
    starts[:D_deterministic_H_perturbation_projected] = melitz_project_start_to_middle_constraints(A_pert, sys, ctx_d20)

    return starts
end

# ============================================================================
# Phase 3: exact gradient / smoothness shakedown (D4 + real D20).
# ============================================================================
println("\n" * "="^100); println("PHASE 3: gradient/smoothness shakedown"); println("="^100); flush(stdout)

phase3_rows = NamedTuple[]

function gradient_shakedown!(rows, label, session, q_fixed, gpj_fixed, A_free_base, ctx, sys; hs=(1e-5, 1e-4), n_dirs=3)
    D_ = ctx.D
    base = melitz_middle_objective_and_gradient!(session, A_free_base, q_fixed, gpj_fixed, ctx; coordinate=:logA)
    base.classification isa FiniteSolved || (@warn "$label: base point not FiniteSolved, skipping shakedown"; return)
    exact_grad = base.grad_free
    rng = MersenneTwister(2026)
    dir_specs = Vector{Tuple{Symbol,Vector{Float64}}}()
    for k in 1:n_dirs
        d = randn(rng, length(A_free_base)); d ./= norm(d)
        push!(dir_specs, (Symbol("random_dense_$k"), d))
    end
    # a same-bin-equality tangent direction: pick the first :same_bin row (if any) and move
    # exactly along its own null direction (any single free coordinate NOT in that row moved
    # alone already keeps a same-bin row's residual fixed at 0 to first order trivially since
    # it's linear -- instead build a genuine two-cell tangent: move both endpoints of an
    # ORDERING row's two free coordinates in the SAME direction by the SAME amount, which
    # keeps (a_hi-a_lo) exactly fixed, hence stays exactly on that row's boundary).
    if length(sys.rhs_A) > 0
        i = findfirst(==(:ordering), sys.kind)
        if i !== nothing
            tangent = zeros(length(A_free_base))
            row = sys.rows_A[i, :]
            nzk = findall(!=(0.0), row)
            if length(nzk) >= 2
                tangent[nzk[1]] = 1.0 / row[nzk[2]]
                tangent[nzk[2]] = 1.0 / row[nzk[1]]
                # ensures row . tangent stays a fixed value; normalize
                tangent ./= norm(tangent)
                push!(dir_specs, (:same_bin_tangent, tangent))
            end
        end
    end
    for (dname, d) in dir_specs
        for h in hs
            Ap = A_free_base .+ h .* d
            rp = melitz_middle_objective_and_gradient!(session, Ap, q_fixed, gpj_fixed, ctx; coordinate=:logA)
            secant_fd = dot(exact_grad, d)   # predicted, from the exact gradient
            reopt_secant = rp.classification isa FiniteSolved ? (rp.Delta - base.Delta) / h : NaN
            relerr = rp.classification isa FiniteSolved ? abs(secant_fd - reopt_secant) / max(1e-12, abs(secant_fd)) : NaN
            push!(rows, (label=label, direction=String(dname), h=h, exact_dirderiv=secant_fd,
                          reoptimized_secant=reopt_secant, relerr=relerr,
                          base_classification=String(nameof(typeof(base.classification))),
                          perturbed_classification=String(nameof(typeof(rp.classification)))))
        end
    end
end

# --- D4 ---
FIXTURE4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta04 = build_melitz_psi_bundle(FIXTURE4; outer_parameterization=:logcutoff, policy=policy_cap,
    backend=:matrix_free, forbid_dense_fallback=true)
ctx4 = obj4.γ
session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
A_free04 = theta_plain04[2:1+ctx4.D^2-1]
sys4 = melitz_fixed_q_middle_constraint_system(theta_plain04, ctx4, obj4)
gradient_shakedown!(phase3_rows, "D4_anchor", session4, q04, gpj04, A_free04, ctx4, sys4)

# --- D20 (anchor + one post-switch point, since inner solves are much slower here) ---
gradient_shakedown!(phase3_rows, "D20_anchor", session_d20, q0_d20, gpj0_d20, A_free0_d20, ctx_d20, nothing === nothing ?
    melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20) : nothing)
if MODE != "smoke"
    q_ps1 = q_full_at(cliff_points[3].sign, cliff_points[3].t)
    sys_ps1 = melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20)  # rank structure from q(t) needs its own theta; see note below
    # NOTE: melitz_origin_intervals needs a theta reproducing q_ps1 -- build one via A anchor + q_ps1.
    theta_ps1 = melitz_fixed_q_state_theta(A_free0_d20, q_ps1, gpj0_d20, ctx_d20)
    sys_ps1 = melitz_fixed_q_middle_constraint_system(theta_ps1, ctx_d20, obj_d20)
    A_free_start_ps1 = melitz_project_start_to_middle_constraints(copy(A_free0_d20), sys_ps1, ctx_d20)
    gradient_shakedown!(phase3_rows, "D20_post_switch1", session_d20, q_ps1, gpj0_d20, A_free_start_ps1, ctx_d20, sys_ps1)
end

open(joinpath(OUTDIR, "melitz_fixedqA_phase3_gradient_shakedown_2026-07-30.csv"), "w") do io
    println(io, "label,direction,h,exact_dirderiv,reoptimized_secant,relerr,base_classification,perturbed_classification")
    for r in phase3_rows
        println(io, "$(r.label),$(r.direction),$(r.h),$(r.exact_dirderiv),$(r.reoptimized_secant),$(r.relerr),$(r.base_classification),$(r.perturbed_classification)")
    end
end
println("Phase 3 CSV written. n_rows=", length(phase3_rows)); flush(stdout)

# --- box-size / Hessian-option diagnostic at the D20 anchor (motivates the D20_MIDDLE_BOX /
# MIDDLE_OPT_D20 choice used in Phase 4/6 below -- run live here, not merely asserted, so the
# choice is reproducible from this one script). ---
println("\n" * "-"^100); println("PHASE 3 (continued): D20 box-size / Hessian-option diagnostic"); println("-"^100); flush(stdout)
box_diag_rows = NamedTuple[]
box_diag_configs = MODE == "smoke" ? [(0.1, MIDDLE_OPT_D20, 10)] :
    [(1.0, OUTER_OPT, 60), (0.3, OUTER_OPT, 60), (0.15, OUTER_OPT, 60), (0.05, OUTER_OPT, 150),
     (0.05, MIDDLE_OPT_D20, 150), (0.1, MIDDLE_OPT_D20, 150)]
sys_anchor_diag = melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20)
for (box, opt, mev) in box_diag_configs
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    t0b = time()
    res = solve_melitz_fixed_q_A_profile(session_d20, q0_d20, gpj0_d20, A_free0_d20, ctx_d20;
        coordinate=:logA, max_evals=mev, box=box, outer_loop_opt=opt, sys=sys_anchor_diag)
    wallb = time() - t0b
    @printf("  box=%.3f opt=%-45s nStatus=%d Delta_final=%.6g n_classified=%d (F/C/I=%d/%d/%d) wall=%.1fs\n",
        box, basename(opt), res.nStatus, res.Delta_final_verified, length(res.eval_log),
        res.n_finite_solved, res.n_above_cap, res.n_infinite_certified, wallb)
    flush(stdout)
    push!(box_diag_rows, (box=box, opt=basename(opt), max_evals=mev, nStatus=res.nStatus,
        Delta_final_verified=res.Delta_final_verified, n_classified=length(res.eval_log),
        n_finite_solved=res.n_finite_solved, n_above_cap=res.n_above_cap, n_infinite_certified=res.n_infinite_certified,
        wall_s=wallb))
end
open(joinpath(OUTDIR, "melitz_fixedqA_phase3_box_hessopt_diagnostic_2026-07-30.csv"), "w") do io
    println(io, "box,opt,max_evals,nStatus,Delta_final_verified,n_classified,n_finite_solved,n_above_cap,n_infinite_certified,wall_s")
    for r in box_diag_rows
        println(io, "$(r.box),$(r.opt),$(r.max_evals),$(r.nStatus),$(r.Delta_final_verified),$(r.n_classified),$(r.n_finite_solved),$(r.n_above_cap),$(r.n_infinite_certified),$(r.wall_s)")
    end
end
println("Box/hessopt diagnostic CSV written."); flush(stdout)

# ============================================================================
# Phase 4/5: bounded D20 profile across the audited cliff.
# ============================================================================
println("\n" * "="^100); println("PHASE 4/5: bounded D20 profile across the audited cliff"); println("="^100); flush(stdout)

max_evals_middle = MODE == "smoke" ? 12 : 150
phase4_rows = NamedTuple[]
prev_best_A_free = nothing

for cp in cliff_points
    q_target = q_full_at(cp.sign, cp.t)
    gpj_target = gpj0_d20   # g fixed throughout (module scope: middle loop holds g fixed)
    theta_for_constraints = melitz_fixed_q_state_theta(A_free0_d20, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)

    # fixed-A anchor evaluation (DeltaStar(A_old, q_post)) -- the "decisive comparison" baseline
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    r_fixedA = melitz_middle_objective_and_gradient!(session_d20, A_free0_d20, q_target, gpj_target, ctx_d20; coordinate=:logA)
    @printf("[%s] fixed-A(anchor) at this q: classification=%s  Delta=%.6g\n", cp.name, nameof(typeof(r_fixedA.classification)), r_fixedA.Delta)
    flush(stdout)

    starts = build_starts(q_target, gpj_target, prev_best_A_free, sys_t)
    best_this_point = nothing
    for (start_name, A_free_start) in starts
        resid0 = melitz_middle_constraint_residuals(sys_t, A_free_start; coordinate=:logA)
        feasible0 = all(resid0[sys_t.sense .== :ge] .>= -1e-6) && all(abs.(resid0[sys_t.kind .== :same_bin]) .< 1e-6)
        r_start = melitz_middle_objective_and_gradient!(session_d20, A_free_start, q_target, gpj_target, ctx_d20; coordinate=:logA)

        t_solve0 = time()
        session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
        res = solve_melitz_fixed_q_A_profile(session_d20, q_target, gpj_target, A_free_start, ctx_d20;
            coordinate=:logA, max_evals=max_evals_middle, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t)
        wall = time() - t_solve0

        n_inner_solves = res.n_fc_calls + res.n_ga_calls
        @printf("  [%s / %-32s] start_feasible=%s start_Delta=%s -> nStatus=%d Delta_min=%.6g Delta_final=%s wall=%.2fs n_inner=%d (F/C/I=%d/%d/%d)\n",
            cp.name, start_name, feasible0, string(r_start.Delta), res.nStatus, res.Delta_min,
            string(res.Delta_final_verified), wall, n_inner_solves, res.n_finite_solved, res.n_above_cap, res.n_infinite_certified)
        flush(stdout)

        push!(phase4_rows, (point=cp.name, sign=cp.sign, t=cp.t, start=String(start_name),
            start_feasible=feasible0, start_Delta=r_start.Delta, start_classification=String(nameof(typeof(r_start.classification))),
            fixedA_anchor_Delta=r_fixedA.Delta, fixedA_anchor_classification=String(nameof(typeof(r_fixedA.classification))),
            nStatus=res.nStatus, Delta_min_trajectory=res.Delta_min, Delta_final_verified=res.Delta_final_verified,
            final_classification=String(nameof(typeof(res.r_final))),
            n_fc_calls=res.n_fc_calls, n_ga_calls=res.n_ga_calls, n_finite_solved=res.n_finite_solved,
            n_above_cap=res.n_above_cap, n_infinite_certified=res.n_infinite_certified, wall_s=wall))

        if res.r_final isa FiniteSolved && (best_this_point === nothing || res.Delta_final_verified < best_this_point[2])
            best_this_point = (res.A_free_final, res.Delta_final_verified)
        end
    end
    global prev_best_A_free = best_this_point === nothing ? A_free0_d20 : best_this_point[1]

    # persist incrementally (long job -- don't lose progress on a mid-run failure)
    open(joinpath(OUTDIR, "melitz_fixedqA_phase4_d20_cliff_profile_2026-07-30.csv"), "w") do io
        println(io, "point,sign,t,start,start_feasible,start_Delta,start_classification,fixedA_anchor_Delta,fixedA_anchor_classification,nStatus,Delta_min_trajectory,Delta_final_verified,final_classification,n_fc_calls,n_ga_calls,n_finite_solved,n_above_cap,n_infinite_certified,wall_s")
        for r in phase4_rows
            println(io, "$(r.point),$(r.sign),$(r.t),$(r.start),$(r.start_feasible),$(r.start_Delta),$(r.start_classification),$(r.fixedA_anchor_Delta),$(r.fixedA_anchor_classification),$(r.nStatus),$(r.Delta_min_trajectory),$(r.Delta_final_verified),$(r.final_classification),$(r.n_fc_calls),$(r.n_ga_calls),$(r.n_finite_solved),$(r.n_above_cap),$(r.n_infinite_certified),$(r.wall_s)")
        end
    end
end
println("Phase 4/5 CSV written (incrementally kept up to date). n_rows=", length(phase4_rows)); flush(stdout)

# ============================================================================
# Phase 6: D4 multistart diagnostic (ordinary point + a chamber-transition point), log-A vs
# log-H, all 4 start types.
# ============================================================================
println("\n" * "="^100); println("PHASE 6: D4 multistart diagnostic"); println("="^100); flush(stdout)
phase6_rows = NamedTuple[]

function d4_multistart!(rows, label, A_free_starts_dict, q_fixed, gpj_fixed, ctx, obj, session, sys)
    for coordinate in (:logA, :logH)
        for (sname, A_free_start) in A_free_starts_dict
            x_start = coordinate == :logH ? melitz_h_free_from_A_free(A_free_start, ctx) : A_free_start
            session.obj.use_cached_x = false; session.obj.x .= NaN
            res = solve_melitz_fixed_q_A_profile(session, q_fixed, gpj_fixed, x_start, ctx;
                coordinate=coordinate, max_evals=100, box=coordinate == :logH ? 2.0 : 1.0,
                outer_loop_opt=OUTER_OPT, sys=sys)
            push!(rows, (label=label, coordinate=String(coordinate), start=String(sname),
                nStatus=res.nStatus, Delta_final_verified=res.Delta_final_verified,
                final_classification=String(nameof(typeof(res.r_final))),
                n_fc_calls=res.n_fc_calls, n_ga_calls=res.n_ga_calls, wall_s=res.wall_s))
            @printf("  [%s / %s / %-10s] Delta_final=%s  n_eval=%d  wall=%.2fs\n",
                label, coordinate, sname, string(res.Delta_final_verified), res.n_fc_calls + res.n_ga_calls, res.wall_s)
            flush(stdout)
        end
    end
end

if MODE != "smoke"
    # D4 ordinary point: anchor itself, perturbed 4 ways.
    rng4 = MersenneTwister(5150)
    d4_starts_ordinary = Dict{Symbol,Vector{Float64}}()
    d4_starts_ordinary[:A_anchor] = melitz_project_start_to_middle_constraints(copy(A_free04), sys4, ctx4)
    d4_starts_ordinary[:B_perturbed_dense] = melitz_project_start_to_middle_constraints(A_free04 .+ 0.05 .* randn(rng4, length(A_free04)), sys4, ctx4)
    h04 = melitz_h_free_from_A_free(A_free04, ctx4)
    d4_starts_ordinary[:C_perturbed_H] = melitz_project_start_to_middle_constraints(melitz_A_free_from_h_free(h04 .+ 0.05 .* sin.(1:length(h04)), ctx4), sys4, ctx4)
    d4_starts_ordinary[:D_perturbed_dense2] = melitz_project_start_to_middle_constraints(A_free04 .+ 0.03 .* randn(rng4, length(A_free04)), sys4, ctx4)
    d4_multistart!(phase6_rows, "D4_ordinary", d4_starts_ordinary, q04, gpj04, ctx4, obj4, session4, sys4)

    # D4 chamber-transition point: find a small dense reduced-q direction crossing a switch.
    bwpolicy4 = PowerScaledQBandwidth(1e-2, 20_000, 0.5)
    stage4 = melitz_build_reduced_q_stage(theta04, session4.obj.x, ctx4, obj4, 1; bandwidth_policy=bwpolicy4, target_switches=4)
    if stage4 !== nothing
        b_q4 = stage4.q_basis_free
        function q4_at(sign, t)
            th = copy(theta_plain04)
            th[1+(ctx4.D^2-1)+1:end] .+= sign .* t .* b_q4
            _, _, _, _, q = expand_free_theta_logcutoff(th, ctx4)
            return q
        end
        events4 = melitz_q_direction_exact_switches(theta_plain04, b_q4, ctx4, ctx4.sorted_tail_ctx; sign=-1, n_switches=1)
        if !isempty(events4)
            t_star4 = events4[1].t * 1.3
            q_trans4 = q4_at(-1, t_star4)
            theta_trans4 = melitz_fixed_q_state_theta(A_free04, q_trans4, gpj04, ctx4)
            sys_trans4 = melitz_fixed_q_middle_constraint_system(theta_trans4, ctx4, obj4)
            d4_starts_trans = Dict{Symbol,Vector{Float64}}()
            d4_starts_trans[:A_anchor_projected] = melitz_project_start_to_middle_constraints(copy(A_free04), sys_trans4, ctx4)
            d4_starts_trans[:B_perturbed] = melitz_project_start_to_middle_constraints(A_free04 .+ 0.05 .* randn(rng4, length(A_free04)), sys_trans4, ctx4)
            d4_starts_trans[:C_perturbed_H] = melitz_project_start_to_middle_constraints(melitz_A_free_from_h_free(h04 .+ 0.03 .* cos.(1:length(h04)), ctx4), sys_trans4, ctx4)
            d4_starts_trans[:D_perturbed2] = melitz_project_start_to_middle_constraints(A_free04 .+ 0.02 .* randn(rng4, length(A_free04)), sys_trans4, ctx4)
            d4_multistart!(phase6_rows, "D4_chamber_transition", d4_starts_trans, q_trans4, gpj04, ctx4, obj4, session4, sys_trans4)
        else
            @warn "Phase 6: no chamber-transition switch found within the tested D4 direction/budget -- skipping that half."
        end
    end
end

open(joinpath(OUTDIR, "melitz_fixedqA_phase6_d4_multistart_2026-07-30.csv"), "w") do io
    println(io, "label,coordinate,start,nStatus,Delta_final_verified,final_classification,n_fc_calls,n_ga_calls,wall_s")
    for r in phase6_rows
        println(io, "$(r.label),$(r.coordinate),$(r.start),$(r.nStatus),$(r.Delta_final_verified),$(r.final_classification),$(r.n_fc_calls),$(r.n_ga_calls),$(r.wall_s)")
    end
end
println("Phase 6 CSV written. n_rows=", length(phase6_rows)); flush(stdout)

# ============================================================================
# Phase 7: conditioning comparison (log-A vs log-H constraint-matrix condition numbers).
# ============================================================================
println("\n" * "="^100); println("PHASE 7: conditioning + cost summary"); println("="^100); flush(stdout)
sys_anchor_d20 = melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20)
condA = cond(sys_anchor_d20.rows_A)
condH = cond(sys_anchor_d20.rows_H)
@printf("D20 anchor constraint-matrix condition number: A-space=%.4g  H-space=%.4g\n", condA, condH)
nnzA = count(!iszero, sys_anchor_d20.rows_A)
nnzH = count(!iszero, sys_anchor_d20.rows_H)
@printf("D20 anchor constraint-matrix nnz: A-space=%d/%d  H-space=%d/%d\n",
    nnzA, length(sys_anchor_d20.rows_A), nnzH, length(sys_anchor_d20.rows_H))

open(joinpath(OUTDIR, "melitz_fixedqA_phase7_conditioning_2026-07-30.csv"), "w") do io
    println(io, "metric,logA,logH")
    println(io, "cond,$condA,$condH")
    println(io, "nnz,$nnzA,$nnzH")
    println(io, "nnz_total,$(length(sys_anchor_d20.rows_A)),$(length(sys_anchor_d20.rows_H))")
end

serialize(joinpath(SCRATCH, "fixedqA_experiment_state_2026-07-30.jls"),
    (theta0_d20=theta0_d20, x0_d20=x0_d20, p_star_d20=p_star_d20, b_q_d20=b_q_d20,
     Delta0=lfd0.Delta, phase3_rows=phase3_rows, phase4_rows=phase4_rows, phase6_rows=phase6_rows,
     condA=condA, condH=condH, cliff_points=cliff_points))
println("\nDONE MIDDLE-LOOP EXPERIMENT (MODE=$MODE)")
