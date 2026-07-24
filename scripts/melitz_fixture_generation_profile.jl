# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 9: staged profiling of `generate_fake_melitz_data` (fake_data.jl). This is a
# DIAGNOSTIC-ONLY instrumented COPY of that function's logic (not a modification of the
# production function itself -- monkey-patching `melitz_solve_wages_ge` in place is unsafe
# in Julia here: a same-signature redefinition does not reliably shadow the more specific
# original method, so an accurate stage breakdown needs its own copy with timers inserted).
#
# Preliminary finding BEFORE running this (source read, `equilibrium.jl`): the governing
# prompt's own premise that the gravity projection is a JuMP/HiGHS QP is WRONG for this
# function -- `project_to_gravity_manifold`/`project_to_gravity_manifold_weighted` are
# ALREADY closed-form (`z_proj = z_raw - c*(dot(c,z_raw)+g0)/dot(c,c)`, weighted Lagrangian
# generalization), no JuMP/HiGHS dependency anywhere in equilibrium.jl or fake_data.jl. This
# script exists to find the REAL dominant cost instead of re-deriving an already-closed-form
# step.
#
# Usage: julia --project=. scripts/melitz_fixture_generation_profile.jl [D...]

using Printf
using Random: MersenneTwister
using Statistics: std
using Roots: find_zero, Bisection
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

function profiled_generate_fake_melitz_data(; D::Int=4, sigma::Float64=2.5, theta_star::Float64=6.8,
                                             target_country::Int=1, seed::Int=29, W::Int=20_000,
                                             draw_mode::Symbol=:halton,
                                             tau_offdiag_logrange::Tuple{Float64,Float64}=(0.05, 0.13),
                                             L_range::Tuple{Float64,Float64}=(0.95, 1.25),
                                             logA_noise_sd::Float64=0.04,
                                             logf_domestic_mean::Float64=-0.55, logf_export_mean::Float64=-0.30,
                                             logf_noise_sd::Float64=0.06,
                                             gamma_prime_bracket::Tuple{Float64,Float64}=(0.2, 1.5),
                                             min_participation_prob::Float64=0.01,
                                             ge_damping::Float64=0.1)
    rng = MersenneTwister(seed)
    j = target_country
    timings = Dict{Symbol,Float64}()
    ge_call_count = Ref(0)
    ge_iter_total = Ref(0)
    popx_call_count = Ref(0)
    inner_fixedpoint_iter_total = Ref(0)
    bisection_call_count = Ref(0)

    t0 = time()
    tau = ones(Float64, D, D)
    for o in 1:D, d in 1:D
        o == d && continue
        tau[o, d] = exp(tau_offdiag_logrange[1] + (tau_offdiag_logrange[2] - tau_offdiag_logrange[1]) * rand(rng))
    end
    L = L_range[1] .+ (L_range[2] - L_range[1]) .* rand(rng, D)
    timings[:raw_primitives] = time() - t0

    t0 = time()
    logA_raw = logA_noise_sd .* randn(rng, D, D)
    c_full = gravity_coefficient_vector(D, tau)
    a_proj_weights = vec([o == d ? 1000.0 : 1.0 for o in 1:D, d in 1:D])
    A = exp.(reshape(project_to_gravity_manifold_weighted(vec(logA_raw), c_full, 0.0, a_proj_weights), D, D))
    timings[:A_gravity_projection] = time() - t0

    t0 = time()
    jj_lin = od2lin(j, j, D)
    f_free_lin = [i for i in 1:D^2 if i != jj_lin]
    logf_raw = [lin2od(i, D)[1] == lin2od(i, D)[2] ? logf_domestic_mean + logf_noise_sd * randn(rng) :
                logf_export_mean + logf_noise_sd * randn(rng) for i in f_free_lin]
    f_proj_weights = [lin2od(i, D)[1] == lin2od(i, D)[2] ? 1000.0 : 1.0 for i in f_free_lin]
    timings[:f_raw_and_weights] = time() - t0

    function build_at(gamma_prime_target::Real)
        A_jj_guess = A[j, j]
        local f, f_jj, w, A_final, X, q, expenditure
        n_inner = 0
        for _ in 1:50
            n_inner += 1
            f_jj = derive_fjj_from_autarky_cutoff(gamma_prime_target, 1.0, 1.0, A_jj_guess, L[j], sigma)
            g0_f = c_full[jj_lin] * log(f_jj)
            logf_free_final = project_to_gravity_manifold_weighted(logf_raw, c_full[f_free_lin], g0_f, f_proj_weights)
            f = zeros(Float64, D, D)
            f[j, j] = f_jj
            for (k, i) in enumerate(f_free_lin)
                o, d = lin2od(i, D)
                f[o, d] = exp(logf_free_final[k])
            end
            ge_call_count[] += 1
            tge0 = time()
            w, A_final, ge_iters = melitz_solve_wages_ge(L, tau, A, f, sigma, theta_star; damping=ge_damping)
            timings[:wage_ge_solve] = get(timings, :wage_ge_solve, 0.0) + (time() - tge0)
            ge_iter_total[] += ge_iters
            popx_call_count[] += 2 * ge_iters  # population_X called twice per GE Jacobi iterate
            abs(A_final[j, j] - A_jj_guess) < 1e-14 * max(1.0, abs(A_jj_guess)) && break
            A_jj_guess = A_final[j, j]
        end
        inner_fixedpoint_iter_total[] += n_inner
        tpx0 = time()
        X, q, expenditure = population_X(w, L, tau, A_final, f, sigma, theta_star)
        timings[:population_trade_share] = get(timings, :population_trade_share, 0.0) + (time() - tpx0)
        popx_call_count[] += 1
        return (f=f, f_jj=f_jj, w=w, A=A_final, X=X, q=q, expenditure=expenditure)
    end

    function link_residual(gamma_prime_target::Real)
        bisection_call_count[] += 1
        r = build_at(gamma_prime_target)
        primitives_trial = MelitzPrimitives(D, sigma, theta_star, j, tau, r.w, r.A, r.f, gamma_prime_target)
        cf_trial = MelitzCounterfactual(j, 1.0, 1.0 * L[j], 1.0, 1.0 * L[j])
        return population_focal_link_residual(primitives_trial, r.X, r.q, r.f_jj, cf_trial)
    end
    t0 = time()
    gamma_prime_target = find_zero(link_residual, gamma_prime_bracket, Bisection(); xatol=1e-12)
    timings[:bisection_total_wallclock] = time() - t0   # includes wage_ge_solve/population_X time above

    t0 = time()
    r = build_at(gamma_prime_target)
    timings[:final_evaluation_at_solved_gamma] = time() - t0
    f, f_jj, w, A_final, X, zhat, expenditure = r.f, r.f_jj, r.w, r.A, r.X, r.q, r.expenditure

    t0 = time()
    primitives = MelitzPrimitives(D, sigma, theta_star, j, tau, w, A_final, f, gamma_prime_target)
    price_power = vec(sum(X, dims=1)) ./ expenditure
    eq = MelitzEquilibrium(expenditure, price_power, zhat, X)
    expenditure_prime = 1.0 * L[j]
    counterfactual = MelitzCounterfactual(j, 1.0, expenditure_prime, 1.0, expenditure_prime)

    min_cutoff = minimum(zhat)
    min_cutoff >= 1.0 || error("zhat >= 1 violated (min=$min_cutoff)")
    for o in 1:D, d in 1:D
        d == o && continue
        zhat[o, d] >= zhat[o, o] || error("export-selection violated")
    end
    min_prob = minimum(pareto_tail_prob(zhat[o, d], theta_star) for o in 1:D, d in 1:D)
    min_prob >= min_participation_prob || error("min participation probability $min_prob < $min_participation_prob")
    gravity_residual_A, gravity_residual_f = gravity_residuals(primitives)
    @assert abs(gravity_residual_A) < 1e-8
    @assert abs(gravity_residual_f) < 1e-8
    @assert maximum(abs.(w .* L .- vec(sum(X, dims=2)))) < 1e-8
    @assert maximum(abs.(price_power .- 1.0)) < 1e-8
    @assert abs(population_focal_link_residual(primitives, X, zhat, f_jj, counterfactual)) < 1e-8
    @assert std(log.(A_final)) > 0.01
    @assert std(log.(f)) > 0.01
    @assert std(withinTransform(A_final)) > 1e-4
    @assert std(withinTransform(f)) > 1e-4
    timings[:validation] = time() - t0

    t0 = time()
    z_draws = pareto_draws(W, D, theta_star; seed=seed + 1, mode=draw_mode)
    timings[:pareto_draws] = time() - t0

    data = MelitzSyntheticData(primitives, eq, counterfactual, L, z_draws, seed)
    return data, (; timings, ge_call_count=ge_call_count[], ge_iter_total=ge_iter_total[],
        popx_call_count=popx_call_count[], inner_fixedpoint_iter_total=inner_fixedpoint_iter_total[],
        bisection_call_count=bisection_call_count[])
end

function run_profile(D; W=20_000, timeout_s=300.0,
                      min_participation_prob=(D <= 4 ? 0.01 : 0.002))
    println("="^70)
    @printf("D=%d, W=%d, min_participation_prob=%.4f (timeout guard %.0fs)\n", D, W,
        min_participation_prob, timeout_s)
    D > 4 && println("  NOTE: min_participation_prob relaxed from the D=4 default 0.01 -- ",
        "matches the prior continuation session's own documented finding that the DEFAULT ",
        "calibration is D=4-specific and fails this gate at larger D (structural/timing ",
        "diagnostic only, NOT an economic fixture claim -- see report Section F).")
    t_start = time()
    data, stats = profiled_generate_fake_melitz_data(; D=D, W=W, min_participation_prob=min_participation_prob)
    total = time() - t_start
    @printf("TOTAL wall = %.3fs\n", total)
    for (k, v) in sort(collect(stats.timings); by=x -> -x[2])
        @printf("  %-32s %8.3fs  (%.1f%%)\n", k, v, 100 * v / total)
    end
    @printf("  bisection_call_count (link_residual/build_at invocations) = %d\n", stats.bisection_call_count)
    @printf("  melitz_solve_wages_ge call count                          = %d\n", stats.ge_call_count)
    @printf("  melitz_solve_wages_ge mean Jacobi iters/call               = %.1f\n",
        stats.ge_iter_total / max(stats.ge_call_count, 1))
    @printf("  population_X call count (2x per GE iter + 1 per build_at)  = %d\n", stats.popx_call_count)
    @printf("  inner A[j,j] fixed-point iterations (total across bisection) = %d\n", stats.inner_fixedpoint_iter_total)
    return total, stats
end

if abspath(PROGRAM_FILE) == @__FILE__
    Ds = isempty(ARGS) ? [4, 10, 20] : parse.(Int, ARGS)
    results = Dict{Int,Any}()
    for D in Ds
        results[D] = run_profile(D)
    end
    println("="^70)
    println("Summary (TOTAL wall by D):")
    for D in Ds
        @printf("  D=%2d: %.3fs\n", D, results[D][1])
    end
end
