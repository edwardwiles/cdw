# A_q separation and gradient diagnostics session (2026-07-29). Phase 1: mechanical audit
# of the EXISTING :logcutoff (q = log zhat) implementation for strict A/q block separation.
# No KNITRO solve is performed here -- pure economic-state reconstruction, safe to run
# concurrently with the Phase 0 test-suite baseline.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Random, LinearAlgebra, Printf

function build_ctx(; D=4, seed=29, W=20_000)
    data = generate_fake_melitz_data(; D=D, seed=seed, W=W)
    obj, theta_free = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
    return obj.γ, data, obj, theta_free
end

println("=== Phase 1: strict A/q block separation audit (:logcutoff) ===")
ctx, data, obj, theta0 = build_ctx()
D = ctx.D
rng = MersenneTwister(777)

@assert length(theta0) == melitz_free_dim_logcutoff(ctx)
nA = D^2 - 1

A0, f0, g0j, fjj0, q0 = expand_free_theta_logcutoff(theta0, ctx)

function phase1_a_perturbation(theta0, ctx, q0, nA, rng)
    max_q_move = 0.0
    for k in 1:nA
        theta_p = copy(theta0)
        theta_p[1+k] += 1e-3 * (0.5 + rand(rng))
        Ap, fp, gj, fjj, qp = expand_free_theta_logcutoff(theta_p, ctx)
        dq = qp .- q0
        max_q_move = max(max_q_move, maximum(abs, dq))
    end
    return max_q_move
end

function phase1_diagnose_sa(theta0, ctx)
    s_a_vals = Float64[]
    for k in 1:5
        theta_p = copy(theta0)
        theta_p[1+k] += 1e-2
        Ap, _, _, _, _ = expand_free_theta_logcutoff(theta_p, ctx)
        s_a = dot(ctx.c_full, vec(log.(Ap)))
        push!(s_a_vals, s_a)
        @printf("  coordinate %d perturbed by 1e-2: dot(c_full, vec(logA)) = %.3e\n", k, s_a)
    end
    return s_a_vals
end

function phase1_q_perturbation(theta0, ctx, A0, nA, nq, rng)
    max_A_move = 0.0
    for k in 1:nq
        theta_p = copy(theta0)
        theta_p[1+nA+k] += 1e-3 * (0.5 + rand(rng))
        Ap, fp, gj, fjj, qp = expand_free_theta_logcutoff(theta_p, ctx)
        dA = vec(log.(Ap)) .- vec(log.(A0))
        max_A_move = max(max_A_move, maximum(abs, dA))
    end
    return max_A_move
end

println("\n-- A-coordinate perturbation test (CURRENT code, as-is) --")
max_q_move = phase1_a_perturbation(theta0, ctx, q0, nA, rng)
@printf("max |q movement| under single free-A perturbations (1e-3 scale), CURRENT code: %.3e\n", max_q_move)
println("(nonzero here means A perturbations move at least one q cell -- confirms NOT strictly separated in the current code as literally written)")

println("\n-- Diagnosing the mechanism: is dot(c_full, vec(logA)) truly zero by construction? --")
phase1_diagnose_sa(theta0, ctx)
println("(if all ~1e-16, the A-gravity restriction holds EXACTLY by pivot construction for ANY A_free -- the q-pivot's use of this term is a machine-precision-noise channel, not real economic coupling)")

println("\n-- q-coordinate perturbation test (CURRENT code) --")
nq = length(theta0) - 1 - nA
max_A_move = phase1_q_perturbation(theta0, ctx, A0, nA, nq, rng)
@printf("max |logA movement| under single free-q perturbations (1e-3 scale): %.3e\n", max_A_move)

println("\nDone.")
