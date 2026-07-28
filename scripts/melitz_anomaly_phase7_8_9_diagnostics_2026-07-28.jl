using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
mkpath(OUTDIR)
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

d20 = build_realD20_fixture()
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D; nA = D^2 - 1

# Properly wire the cap this time (this session's own Phase 5 fix): the fixture's obj.lower_limit
# is -Inf by construction (the confirmed root cause); set it explicitly to match the
# delta_evaluation_cap used throughout this script.
CAP = 10.0
obj.lower_limit = -CAP

profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]

function g_for_target(target)
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    return gs[k] + t * (gs[k+1] - gs[k])
end

g_interior = g_for_target(0.5)
g_nearbudget = g_for_target(1.0)
println("g_interior (DeltaStar~0.5) = ", g_interior)
println("g_nearbudget (DeltaStar~1) = ", g_nearbudget)
flush(stdout)

theta_interior = copy(theta0); theta_interior[1] = g_interior
theta_nearbudget = copy(theta0); theta_nearbudget[1] = g_nearbudget

h_fd = 1e-4
direct_gradient_fn = make_melitz_gradient_delta_direct_parallel(h_fd)

function gt_pct(g)
    100.0 * melitz_welfare_metrics_from_g(g, ctx).gains_from_trade
end

rows7 = NamedTuple[]
rows8 = NamedTuple[]
n_solves = Ref(0)

function run_point(label, theta_pt)
    println("\n" * "="^100)
    println("POINT: $label  (theta[1]=g=$(theta_pt[1]))")
    println("="^100)
    bank = MelitzDualBank(8)
    r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=CAP, bank=bank)
    n_solves[] += 1
    @assert r0 isa FiniteSolved "base point $label not FiniteSolved: $(typeof(r0))"
    Delta0 = r0.Delta
    x0 = copy(r0.x)
    GT0 = gt_pct(theta_pt[1])
    println("  Delta0=$Delta0  GT0=$(GT0)pp  ||x0||=$(norm(x0))")
    flush(stdout)

    snap = base_active_mask(theta_pt, ctx, obj)

    g_grad = zeros(n)
    direct_gradient_fn(g_grad, theta_pt, ctx, obj, x0)
    g_grad ./= 1e10   # convert d(1e10*Delta)/dtheta -> dDelta/dtheta

    # direction 1: pure gamma (welfare-improving). Determine sign empirically via a tiny probe
    # on GT itself (gains_from_trade is a pure function of g=theta[1]).
    sign1 = gt_pct(theta_pt[1] + 1e-6) > GT0 ? 1.0 : -1.0
    v1 = zeros(n); v1[1] = sign1

    # direction 2: normalized technology (logA) block, random unit vector (seeded, deterministic),
    # oriented to (locally) DECREASE predicted DeltaStar per the assembled gradient.
    rawA = zeros(n); rawA[2:1+nA] .= randn(MersenneTwister(2001 + Int(round(1000*theta_pt[1]))), nA)
    rawA ./= norm(rawA)
    v2 = (dot(g_grad, rawA) > 0 ? -1.0 : 1.0) .* rawA

    # direction 3: normalized participation/fixed-cost (logf) block, random unit vector,
    # same DeltaStar-decreasing sign convention.
    rawF = zeros(n); rawF[2+nA:end] .= randn(MersenneTwister(3001 + Int(round(1000*theta_pt[1]))), n - 1 - nA)
    rawF ./= norm(rawF)
    v3 = (dot(g_grad, rawF) > 0 ? -1.0 : 1.0) .* rawF

    # direction 4: mixed gamma+participation, literally the production gradient's own
    # restriction to those coordinates (technology/A-block zeroed out), negated (descent on
    # predicted DeltaStar), normalized.
    v4raw = zeros(n); v4raw[1] = -g_grad[1]; v4raw[2+nA:end] .= -g_grad[2+nA:end]
    v4 = v4raw ./ norm(v4raw)

    directions = [("pure_gamma", v1, false), ("technology_block", v2, false),
                  ("participation_block", v3, true), ("mixed_gamma_participation", v4, true)]
    steps = (1e-6, 1e-5, 1e-4)

    for (dname, v, has_secant) in directions
        for step in steps
            theta_new = theta_pt .+ step .* v
            predicted_A = dot(g_grad, step .* v)   # assembled coordinatewise gradient prediction

            predicted_B = NaN
            if has_secant
                melitz_bundle_prepare_at_theta!(obj, theta_new)
                localc = zeros(1)
                obj(x0, constr=localc)   # FIXED dual x0, moved moment matrix -- one-shot block secant
                Delta_fixed_new = localc[1] / 1e10
                predicted_B = Delta_fixed_new - Delta0
            end

            bank_i = MelitzDualBank(8)
            r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank_i)
            n_solves[] += 1
            reoptimized = r_new isa FiniteSolved ? (r_new.Delta - Delta0) : NaN
            nsw, _, nauk = count_switches(snap, theta_new, ctx, obj)
            dGT = gt_pct(theta_new[1]) - GT0

            push!(rows7, (point=label, direction=dname, step=step, dGT_pp=dGT,
                pred_dDelta_assembled_gradient=predicted_A, pred_dDelta_block_secant=predicted_B,
                reoptimized_dDelta=reoptimized, n_switches=nsw, n_autarky_switches=nauk,
                classification=string(typeof(r_new))))
            @printf("  [%-26s step=%.0e] dGT=%+.4fpp  A=%+.4e  B=%s  reopt=%+.4e  nsw=%d  class=%s\n",
                dname, step, dGT, predicted_A, has_secant ? @sprintf("%+.4e", predicted_B) : "NA",
                reoptimized, nsw, typeof(r_new))
            flush(stdout)

            # Phase 8: intensive/switching decomposition, ONLY for participation/mixed
            # (has_secant) directions -- reuses the SAME trial, no new solves.
            if has_secant
                intensive = predicted_A            # coordinatewise-assembled (small-h FD), active-set-frozen proxy
                switching = predicted_B - predicted_A   # residual: what the full block secant captures beyond the frozen-active-set prediction
                push!(rows8, (point=label, direction=dname, step=step,
                    total_fixed_dual_change=predicted_B, intensive_component=intensive,
                    switching_component=switching,
                    intensive_plus_switching_check=intensive + switching,
                    reoptimized_dDelta=reoptimized, n_switches=nsw))
            end
        end
    end
    return (theta=theta_pt, Delta0=Delta0, x0=x0, GT0=GT0, snap=snap, g_grad=g_grad)
end

state_interior = run_point("interior_Delta0.5", theta_interior)
state_nearbudget = run_point("nearbudget_Delta1", theta_nearbudget)

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
end
write_csv(joinpath(OUTDIR, "melitz_phase7_participation_gradient_diagnostic_2026-07-28.csv"), rows7)
write_csv(joinpath(OUTDIR, "melitz_phase8_intensive_switching_decomposition_realD20_2026-07-28.csv"), rows8)

println("\nTotal fully-reoptimized inner solves this script: ", n_solves[])
flush(stdout)

# ============================================================================
# Phase 9: composite (xi = theta_star*logA + beta*logf) vs cutoff/decomposition direction.
# ============================================================================
println("\n" * "="^100)
println("PHASE 9: composite-vs-cutoff analytic diagnostic")
println("="^100)

# Verify the active beta convention from production code: melitz_expand_theta's own f-block
# parameterization (outer_parameterization=:logf per build_melitz_psi_bundle_from_calibration).
println("ctx.theta_star = ", ctx.theta_star)
theta_star = ctx.theta_star
flush(stdout)

rows9 = NamedTuple[]

function phase9_point(label, theta_pt, Delta0, x0, snap)
    println("\n--- $label ---")
    # xi_od = theta_star*logA_od + beta*logf_od  with beta=1 (theta_star already scales A;
    # f enters the profit-cutoff condition with unit elasticity in this codebase's own
    # f-parameterization -- verified via melitz_expand_theta's f_jj/f_od direct (unscaled) use).
    beta = 1.0
    xi = zeros(n); xi[2:1+nA] .= theta_star; xi[2+nA:end] .= beta   # d(xi)/d(logA), d(xi)/d(logf) coefficients
    # Composite direction: move logA and logf so that xi=theta_star*logA+beta*logf stays
    # approximately FIXED while displacing the A/f split -- i.e. move ALONG the composite's
    # own null direction in (logA,logf) space at one coordinate pair, or more simply here (a
    # deliberately minimal diagnostic, not a full reparameterization): move logA UP and logf
    # DOWN in the ratio that cancels in xi (a random single A/f coordinate PAIR shared across
    # a random subset, normalized) -- approximates "changes the composite while approximately
    # holding cutoffs fixed" is the ORTHOGONAL case; we build both explicitly below.
    rng = MersenneTwister(9001)
    dirA = zeros(nA); dirA .= randn(rng, nA); dirA ./= norm(dirA)
    dirF = zeros(n - 1 - nA); dirF .= randn(rng, n - 1 - nA); dirF ./= norm(dirF)

    # "Composite" direction: change A and f TOGETHER, in the SAME sign as an INCREASE in xi
    # (theta_star*dlogA + beta*dlogf > 0 uniformly), holding the cutoff-relevant combination
    # roughly fixed by using a shared, common-sign random pattern on A oriented consistently.
    v_composite = zeros(n)
    v_composite[2:1+nA] .= dirA
    v_composite[2+nA:end] .= dirF
    v_composite ./= norm(v_composite)

    # "Cutoff/decomposition" direction: change logA and logf in OPPOSITE proportion so that
    # xi = theta_star*logA + beta*logf is approximately held fixed for a matched coordinate
    # subset (the D free A cells, D free f cells sharing the same random pattern, but logf
    # scaled by -theta_star/beta so the two contributions cancel in xi to first order).
    v_cutoff = zeros(n)
    v_cutoff[2:1+nA] .= dirA
    v_cutoff[2+nA:end] .= -(theta_star / beta) .* dirA[1:min(length(dirA), n - 1 - nA)]
    if length(dirA) < n - 1 - nA
        v_cutoff[2+nA+length(dirA):end] .= 0.0
    end
    xi_change_check = dot(xi[2:1+nA], v_cutoff[2:1+nA]) + dot(xi[2+nA:end], v_cutoff[2+nA:end])
    println("  ||xi-change along v_cutoff|| (should be near 0) = ", abs(xi_change_check))
    v_cutoff ./= norm(v_cutoff)

    step = 1e-5
    for (dname, v) in (("composite_xi", v_composite), ("cutoff_decomposition", v_cutoff))
        for sign in (1.0, -1.0)
            theta_new = theta_pt .+ sign * step .* v
            melitz_bundle_prepare_at_theta!(obj, theta_new)
            localc = zeros(1)
            obj(x0, constr=localc)
            Delta_fixed_new = localc[1] / 1e10
            total_fixed_dual = Delta_fixed_new - Delta0

            bank_i = MelitzDualBank(8)
            r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank_i)
            n_solves[] += 1
            reoptimized = r_new isa FiniteSolved ? (r_new.Delta - Delta0) : NaN
            nsw, _, nauk = count_switches(snap, theta_new, ctx, obj)
            dGT = gt_pct(theta_new[1]) - GT0_for(theta_pt)

            push!(rows9, (point=label, direction=dname, sign=sign, step=step, dGT_pp=dGT,
                total_fixed_dual_change=total_fixed_dual, reoptimized_dDelta=reoptimized,
                n_switches=nsw, classification=string(typeof(r_new))))
            @printf("  [%-20s sign=%+.0f] dGT=%+.4fpp  fixed_dual_change=%+.4e  reopt=%+.4e  nsw=%d  class=%s\n",
                dname, sign, dGT, total_fixed_dual, reoptimized, nsw, typeof(r_new))
            flush(stdout)
        end
    end
end
GT0_for(theta_pt) = gt_pct(theta_pt[1])

phase9_point("interior_Delta0.5", state_interior.theta, state_interior.Delta0, state_interior.x0, state_interior.snap)
phase9_point("nearbudget_Delta1", state_nearbudget.theta, state_nearbudget.Delta0, state_nearbudget.x0, state_nearbudget.snap)

write_csv(joinpath(OUTDIR, "melitz_phase9_composite_cutoff_diagnostic_realD20_2026-07-28.csv"), rows9)

println("\nTotal fully-reoptimized inner solves (Phase 7+8+9 combined): ", n_solves[])
println("\nDONE Phase 7/8/9.")
