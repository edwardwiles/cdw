# ============================================================================
# Compares the CURRENT method (blind first CC solve) against the GRAVITY-SEEDED
# variant (first CC solve augmented with the linearized gravity moment from the
# previous F) across a short CHAIN of theta evaluations with increasing
# perturbation size, mimicking what an outer KNITRO search explores as it moves
# away from the calibrated point (real production runs at larger divergence
# budgets delta=10/20 hit genuinely degenerate R_mean (~1e48-1e107) and
# gravity-infeasible KNITRO-own points -- see trade_robustness_modular's
# 60ca63d commit message. This experiment tests whether gravity-seeding the
# first solve reduces that drift risk, using theta-perturbation size as a
# proxy for how far the real outer search explores.
#
# Both methods share the SAME warm-started umat (inversion warm start) at each
# step -- the ONLY difference under test is whether the first CC solve for p
# is blind (D+1 focal-only moments) or seeded with the previous F's linearized
# gravity moment (D+2 moments, using the chain's OWN previous (p, umat)).
#
#   DVAL=10 julia --project=. full_aod_diag/gravity_seeded_initial_solve/compare_drift.jl
# ============================================================================

include(joinpath(@__DIR__, "setup_and_variant.jl"))
using Printf

function make_chain(nsteps::Int, scale::Float64; seed::Int = 1)
    rng = Random.MersenneTwister(seed)
    θs = Vector{Vector{Float64}}(undef, nsteps + 1)
    θs[1] = copy(θr0)
    for k in 1:nsteps
        θnext = copy(θs[k])
        # perturb the free A-column (theta[4:3+D]) and gamma'_focal (theta[3]) -- the outer
        # solve's own free parameters (mu is frozen) -- with growing amplitude, a proxy for how
        # far a large-delta outer search explores away from the calibrated point.
        amp = scale * k
        # log-normal (always-positive) multiplicative perturbation -- Acol/gamma'_focal must stay
        # positive (they enter focal_u via a fractional power); a naive (1+amp*randn) perturbation
        # can go negative and hit a DomainError, which is a bug in the perturbation, not a finding.
        θnext[4:3+D] .*= exp.(amp .* randn(rng, D))
        θnext[3] *= exp(0.3 * amp * randn(rng))
        θs[k+1] = θnext
    end
    return θs
end

function run_chain(θs, method::Symbol)
    n = length(θs)
    rows = NamedTuple[]
    warm = nothing; warm_p = nothing
    for (i, θ) in enumerate(θs)
        r = if method == :baseline
            seq_gravcol_baseline(θ; warm = warm)
        else
            seq_gravcol_gravityseed(θ; warm = warm, warm_p = warm_p)
        end
        push!(rows, (step = i - 1, R0 = r.R0, R_final = r.R, n_iters = r.n_iters, ok = r.ok, div_p = r.div_p))
        warm = r.umat; warm_p = r.p
    end
    return rows
end

for (scale_label, scale) in (("moderate", 0.15), ("large", 0.40))
    println("\n" * "="^80)
    println("CHAIN scale=$scale_label ($scale per-step amplitude)")
    println("="^80)
    θs = make_chain(4, scale)
    rows_b = run_chain(θs, :baseline)
    rows_g = run_chain(θs, :gravityseed)
    @printf("%-5s | %-14s %-14s %-8s %-6s %-10s | %-14s %-14s %-8s %-6s %-10s\n",
            "step", "R0(baseline)", "Rfinal(base)", "iters", "ok", "div_p", "R0(seeded)", "Rfinal(seed)", "iters", "ok", "div_p")
    for (rb, rg) in zip(rows_b, rows_g)
        @printf("%-5d | %-14.4e %-14.4e %-8d %-6s %-10.4e | %-14.4e %-14.4e %-8d %-6s %-10.4e\n",
                rb.step, rb.R0, rb.R_final, rb.n_iters, rb.ok, rb.div_p,
                rg.R0, rg.R_final, rg.n_iters, rg.ok, rg.div_p)
    end
end
