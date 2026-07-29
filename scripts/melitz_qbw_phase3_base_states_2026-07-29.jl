# q-bandwidth convergence campaign (2026-07-29), Phase 3: fix the economic test points.
#
# Reuses the EXACT interpolate-g-from-existing-profile-CSV mechanism the 2026-07-28
# post-consolidation session's own `melitz_phase2_profile_recert_2026-07-28.jl` established
# (attributed, not re-derived) -- NOT a new gamma-profile campaign. For each target, builds
# the verified FiniteSolved (A,f,gamma_prime_j) point under the PRODUCTION `:logf`
# parameterization, then reconstructs the IDENTICAL economic point under `:logcutoff` via
# `reduce_to_free_theta_logcutoff` and round-trip-verifies it solves to the SAME DeltaStar.
#
# D4 (seed=29, W=20000): pareto/delta~0.1/delta~0.5 only -- delta~1.0/2.0 are NOT reachable in
# this fixed-A/f corridor (max finite profile Delta=0.572, established in
# docs/melitz_post_consolidation_validation_2026-07-28.md Phase 2); per the governing prompt's
# own priority order ("construct via the reliable one-dimensional gamma profile or
# continuation, NOT a new broad nuisance-parameter campaign"), these two targets are reported
# as unreachable rather than force-constructed via flexible-nuisance re-optimization (which
# would itself be exactly the "broader campaign" the prompt rules out as the fallback).
#
# real-D20 (seed=1, W=80000): all four of pareto/0.1/0.5/1.0/~2.0 are reachable (established in
# the same doc's Phase 2 table); ~2.0 lands at Delta=1.65061, disclosed as the closest
# achievable point rather than exactly 2.0.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random

const OUTDIR = joinpath(REPO, "docs", "key_results")
isdir(OUTDIR) || mkpath(OUTDIR)
CAP = 10.0
policy = CappedEvaluation(CAP)

function interp_g_at_target(profile, target)
    finite = filter(r -> r.classification == "FiniteSolved", profile)
    gs = [r.g for r in finite]; ds = [r.DeltaStar for r in finite]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    target == 0.0 && return gs[1], nothing   # pareto row itself (smallest |g|, base point)
    if target > maximum(ds)
        return nothing, maximum(ds)
    end
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    k === nothing && return nothing, maximum(ds)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    g = gs[k] + t * (gs[k+1] - gs[k])
    return g, nothing
end

"""
    build_base_state(label, build_fn, profile_csv, target; W, seed) -> NamedTuple or nothing

Builds the verified FiniteSolved point at `target` under BOTH :logf and :logcutoff, and
cross-checks their DeltaStar match to high precision (the round-trip validation the governing
prompt's own Phase 3 implicitly requires: the two coordinate systems must describe the SAME
economic point, not merely two independently-plausible ones).
"""
function build_base_state(label::AbstractString, build_fn, profile_csv::AbstractString, target::Real;
                           W::Int, seed::Int)
    profile = load_gamma_profile_csv(profile_csv)
    g, maxd = interp_g_at_target(profile, target)
    if g === nothing
        println("-- $label target=$target: NOT REACHABLE (max profile FiniteSolved Delta=$maxd) -- SKIPPED")
        return nothing
    end

    # :logf verified point
    d_logf = build_fn(; W=W, seed=seed, policy=policy)
    ctx_logf, obj_logf, theta0_logf = d_logf.ctx, d_logf.obj, d_logf.theta0
    theta_logf = copy(theta0_logf); theta_logf[1] = g
    session_logf = MelitzInnerSession(obj_logf, ctx_logf, policy)
    r_logf = solve_melitz_delta!(session_logf, theta_logf, policy; origin_block_screen=false, warm_start_source=:previous)
    if !(r_logf isa FiniteSolved)
        println("-- $label target=$target: :logf solve returned $(typeof(r_logf)) -- SKIPPED")
        return nothing
    end
    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta_logf, ctx_logf)

    # :logcutoff reconstruction of the IDENTICAL economic point
    data = build_fn == build_d4_fixture ?
        generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W) : nothing
    obj_q, theta0_q = if build_fn == build_d4_fixture
        build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free,
                                 policy=policy, outer_parameterization=:logcutoff)
    else
        d20_calib = _load_realD20_calib()
        build_melitz_psi_bundle_from_calibration(d20_calib; W=W, seed=seed,
            inner_loop_opt=joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt"),
            forbid_dense_fallback=true, policy=policy, outer_parameterization=:logcutoff)
    end
    ctx_q = obj_q.γ
    theta_q = reduce_to_free_theta_logcutoff(A, f, gamma_prime_j, ctx_q)
    obj_q.use_cached_x = false; obj_q.x .= NaN
    lfd_q = melitz_recover_lfd(obj_q, theta_q)
    if !lfd_q.lfd_ok
        println("-- $label target=$target: :logcutoff round-trip FAILED to verify -- SKIPPED")
        return nothing
    end
    delta_mismatch = abs(lfd_q.Delta - r_logf.Delta)
    @printf("-- %-22s target=%-5s g=%.6f  Delta_logf=%.8e  Delta_logcutoff=%.8e  |mismatch|=%.3e\n",
            label, string(target), g, r_logf.Delta, lfd_q.Delta, delta_mismatch)
    flush(stdout)
    @assert delta_mismatch < 1e-6 * max(1.0, abs(r_logf.Delta)) "round-trip Delta mismatch too large"

    return (label=label, target=target, g=g, W=W, seed=seed,
            Delta=r_logf.Delta, nStatus=r_logf.nStatus,
            theta_q=theta_q, x0_q=copy(lfd_q.dual_x))
end

# real-D20 calibration is expensive (Pareto MLE + wage GE) -- build ONCE, reuse across targets.
const _REALD20_CALIB_CACHE = Ref{Any}(nothing)
function _load_realD20_calib()
    if _REALD20_CALIB_CACHE[] === nothing
        real_dir = joinpath(REPO, "real_data", "noah_D20")
        lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
        LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
        tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
        countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
        focal = findfirst(==("fra"), countries)
        observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
        _REALD20_CALIB_CACHE[] = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
            p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    end
    return _REALD20_CALIB_CACHE[]
end

println("="^100)
println("Phase 3: D4 base states (seed=29, W=20000)")
println("="^100)
d4_states = Dict{Any,Any}()
for target in (0.0, 0.1, 0.5, 1.0, 2.0)
    r = build_base_state("D4_seed29_W20000", build_d4_fixture,
        joinpath(OUTDIR, "melitz_phase2_gamma_profile_d4_2026-07-28.csv"), target; W=20_000, seed=29)
    r !== nothing && (d4_states[target] = r)
end

println("\n" * "="^100)
println("Phase 3: real-D20 base states (seed=1, W=80000)")
println("="^100)
d20_states = Dict{Any,Any}()
for target in (0.0, 0.1, 0.5, 1.0, 2.0)
    r = build_base_state("realD20_seed1_W80000", build_realD20_fixture,
        joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"), target; W=80_000, seed=1)
    r !== nothing && (d20_states[target] = r)
end

# --- Phase 4 (partial): nested QMC prefix property, explicit test, not assumed ---
println("\n" * "="^100)
println("Phase 4: nested QMC W-prefix property check (pareto_draws, D4 theta_star)")
println("="^100)
theta_star4 = 6.8
z_small = pareto_draws(20_000, 4, theta_star4; seed=29, mode=:halton)
z_large = pareto_draws(1_280_000, 4, theta_star4; seed=29, mode=:halton)
prefix_ok = z_small == z_large[1:20_000, :]
println("W=20000 is an exact prefix of W=1,280,000 (same seed=29): ", prefix_ok)
@assert prefix_ok "nested QMC prefix property FAILED -- must not assume it"

z_seed2 = pareto_draws(20_000, 4, theta_star4; seed=41, mode=:halton)
println("Different seed (41 vs 29) gives a genuinely different sample: ", z_seed2 != z_small)

open(joinpath(OUTDIR, "melitz_qbw_phase3_base_states_2026-07-29.csv"), "w") do io
    println(io, "label,target,g,W,seed,Delta,nStatus,theta_q_dim")
    for (states, ) in [(d4_states,), (d20_states,)]
        for (target, r) in sort(collect(states); by=x -> x[1])
            println(io, join([r.label, r.target, r.g, r.W, r.seed, r.Delta, r.nStatus, length(r.theta_q)], ","))
        end
    end
end

# Persist the full theta_q vectors (needed by later phases) as simple delimited rows.
open(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"), "w") do io
    for (states,) in [(d4_states,), (d20_states,)]
        for (target, r) in sort(collect(states); by=x -> x[1])
            println(io, join(vcat([r.label, r.target, r.W, r.seed], r.theta_q), ","))
        end
    end
end

println("\nPhase 3 (+ partial Phase 4 prefix check) complete. CSVs written to docs/key_results/.")
flush(stdout)
