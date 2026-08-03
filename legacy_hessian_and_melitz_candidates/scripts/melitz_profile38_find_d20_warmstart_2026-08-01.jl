# Attempt to find the FIRST feasible real-D20 KNITRO incumbent under the revised-gravity base
# (1b7a8ac) by re-embedding an OLD (pre-revision) converged incumbent's PHYSICAL (A,f,q)
# values -- the underlying economy (w,L,tau,lambda) did not change, only which cells feed the
# gravity regression/pivot selection did -- rather than cold-starting from the new calibration's
# own raw reduce-then-expand point (already shown, this session, to be a poor KNITRO cold start:
# nStatus=-300 / AboveEvaluationCap).
REPO2 = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profile-38-nongravity-nonfocal-2026-08-01"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using DelimitedFiles, Printf, LinearAlgebra, Serialization

melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)

mutable struct ChainPoint
    idx::Int; phase::Symbol; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    within_budget::Bool; best_start::Symbol; trigger_reason::Symbol
    A_free::Vector{Float64}; theta_free::Vector{Float64}
    f_full::Matrix{Float64}; q_full::Matrix{Float64}; dual_x::Vector{Float64}
    lfd_weights::Vector{Float64}; lfd_ok::Bool; lfd_Delta::Float64; nStatus::Int
    unique_A_points::Int; unique_inner_solves::Int; n_fc_calls::Int; n_ga_calls::Int
    wall_s::Float64; timestamp::String
end
mutable struct ChainState
    anchor_label::String; direction::Symbol; fingerprint::NamedTuple; points::Vector{ChainPoint}
    phase::Symbol; step_gt::Float64; bracket::Union{Nothing,NTuple{4,Float64}}
    n_evaluated::Int; n_polish::Int; n_accepted::Int; most_extreme_idx::Int
    n_nonimproving::Int; elapsed_wall_s::Float64; status::Symbol
    cur_A_free::Vector{Float64}; cur_q::Matrix{Float64}; cur_p_star::Vector{Float64}
    cur_g::Float64; cur_GT::Float64; prev_sys::Any
end

ckpt = joinpath(REPO2, "docs/key_results/production_delta0p5_2026-07-31/checkpoints/current_calibration_upper.jls")
old_state = deserialize(ckpt)
old = old_state.points[old_state.most_extreme_idx]
@assert old.classification == :FiniteSolved && old.lfd_ok
@printf("OLD incumbent: classification=%s GT=%.6f%% Delta=%.6e g=%.6f\n", old.classification, old.GT, old.Delta, old.g)
f_old = old.f_full
q_old = old.q_full
gamma_prime_j_old = exp(old.g)
flush(stdout)

real_dir = joinpath(REPO2, "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = resolve_country_index(countries, "fra")
D = length(countries)

observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
    p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)

obj, theta0 = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    outer_parameterization=:logcutoff,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=CappedEvaluation(10.0))
ctx = obj.γ
@printf("NEW ctx: D=%d target_country=%d W=%d n_theta=%d\n", ctx.D, ctx.target_country, size(obj.U,1), length(theta0))
@assert ctx.target_country == focal
flush(stdout)

# Invert (f_old, q_old) -> A_old under the NEW ctx's own (w,tau,expenditure,sigma) -- these are
# DATA-derived quantities (empirical lambda/L via melitz_solve_wages) unaffected by the gravity
# SAMPLE/theta_star revision, so reusing ctx (not the old, unavailable ctx) is valid.
sigma = ctx.sigma
markup = melitz_markup(sigma)
w = ctx.w; tau = ctx.tau; expenditure = ctx.expenditure
A_old = zeros(D, D)
for o in 1:D, d in 1:D
    log_f_od = log(f_old[o, d])
    q_od = q_old[o, d]
    a_od = (log_f_od - (sigma - 1) * q_od + (sigma - 1) * log(markup) + (sigma - 1) * log(w[o]) +
            (sigma - 1) * log(tau[o, d]) - log(expenditure[d]) + log(sigma) + log(w[o])) / (sigma - 1)
    A_old[o, d] = exp(a_od)
end
# round-trip sanity check: recompute q from (A_old, f_old) via the production formula and compare
q_check = melitz_baseline_cutoff(A_old, f_old, w, tau, expenditure, sigma)
max_qdev = maximum(abs.(log.(q_check) .- q_old))
@printf("Inversion round-trip check: max|log(recomputed zhat) - q_old| = %.3e (should be ~0)\n", max_qdev)
@assert max_qdev < 1e-6 "A inversion from (f_old,q_old) failed round-trip check"
flush(stdout)

theta1 = reduce_to_free_theta_logcutoff(A_old, f_old, gamma_prime_j_old, ctx)
_, _, gpj_check, _, q1_check = expand_free_theta_logcutoff(theta1, ctx)
@printf("Re-embedded theta1: |gamma_prime_j round-trip| dev = %.3e, |q round-trip| max dev = %.3e\n",
    abs(gpj_check - gamma_prime_j_old), maximum(abs.(q1_check .- q_old)))
flush(stdout)

session = MelitzInnerSession(obj, ctx, CappedEvaluation(10.0))

println("\n--- Attempt 1: solve_melitz_delta! at theta1, cold (warm_start_source=:neutral) ---")
t0 = time()
r1 = solve_melitz_delta!(session, theta1, CappedEvaluation(10.0); warm_start_source=:neutral)
@printf("solved in %.1fs -> %s\n", time() - t0, typeof(r1))
if r1 isa FiniteSolved
    @printf("Delta=%.6e nStatus=%d lfd_ok=%s\n", r1.Delta, r1.nStatus, true)
else
    println(r1)
end
flush(stdout)

if !(r1 isa FiniteSolved)
    println("\n--- Attempt 2: raw melitz_recover_lfd at theta1, warm-started from OLD dual_x ---")
    if length(old.dual_x) == obj.outer_constr_index
        obj.x = copy(old.dual_x)
        obj.use_cached_x = true
        t1 = time()
        lfd2 = melitz_recover_lfd(obj, theta1)
        @printf("solved in %.1fs -> Delta=%.6e nStatus=%d lfd_ok=%s\n", time() - t1, lfd2.Delta, lfd2.nStatus, lfd2.lfd_ok)
    else
        println("dimension mismatch: length(old.dual_x)=$(length(old.dual_x)) vs obj.outer_constr_index=$(obj.outer_constr_index) -- skipping")
    end
    flush(stdout)
end

serialize(joinpath(REPO2, "scratch_theta1_d20.jls"), (theta1=theta1, theta0=theta0, ctx_D=D, focal=focal))
println("\nDone.")
