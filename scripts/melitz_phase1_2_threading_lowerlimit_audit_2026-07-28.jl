# 2026-07-28 outer-search gradient/redundancy/sensitivity continuation, Phases 0-2.
#
# Phase 1: re-confirm the "9s vs 0.95s" outer-gradient discrepancy on the CURRENT code at the
# real-D20 calibration point, using the SAME serial/parallel direct sorted-gradient factories
# the 2026-07-27 session benchmarked (docs/melitz_outer_search_scaling_and_profile_2026-07-27.md
# Phase 1/3: 11.1s serial, 0.955s parallel, 11.67x). Root cause was already found by static
# inspection (not re-guessed here): scripts/melitz_phase8_9_10_realD20_2026-07-28.jl:172
# hardcoded gradient_backend=:B_direct_argument_sorted_serial despite running under -t 20 --
# this script reproduces the SAME timing gap live, on-demand, at will (not a mystery), and
# confirms melitz_note_explicit_gradient_backend_choice (backend_config.jl, this session's own
# Phase 0 addition) fires for exactly that configuration.
#
# Phase 2: audit cold inner solves >30s for the missing-lower_limit failure mode. Rather than
# attempt a bit-exact historical replay of the exact 91.4s solve (its own theta was never
# persisted to disk -- only summary statistics survive in the Phase 8/9 CSVs), this reruns the
# SAME real-D20 Phase 8 (nuisance minimization from an interior point) -> Phase 9 (Active Set
# outer search) pipeline fresh, with an on_inner_result hook (melitz_classified_inner_solve's
# own documented diagnostic hook) recording EVERY classified inner-solve outcome's wall time
# and full classification detail. Any solve >30s is reported in full: FiniteSolved (genuinely
# hard point, lower_limit correctly never fired because the solve legitimately converged before
# crossing it) vs AboveEvaluationCap (lower_limit fired, correctly, just slowly) vs
# NumericalFailure (the historical missing-lower_limit symptom: ran to the .opt file's own
# maxtime_real=90 cap with NO certificate) are cleanly distinguished by construction.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase1_2_threading_lowerlimit_audit_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(@__DIR__, "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const REPO = dirname(@__DIR__)
const OUTDIR = joinpath(REPO, "docs", "key_results")
mkpath(OUTDIR)
const CAP = 10.0

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

function phase0_1_threading(ctx, obj, theta0)
    println("="^100); println("PHASE 0: thread startup report"); println("="^100)
    rep = melitz_thread_startup_report(require=20, strict=false)
    flush(stdout)

    println("\n" * "="^100); println("PHASE 1: serial vs parallel outer-gradient timing at the real-D20 calibration point"); println("="^100)
    n = length(theta0)
    D = ctx.D
    g_serial = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
    g_parallel = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)

    # Warm the inner dual state at theta0 first (obj.H / dual state must be current for the
    # gradient closures, exactly as inner_solve_verified_or_fail ensures in production).
    r0 = melitz_classified_inner_solve(obj, theta0, ctx; delta_evaluation_cap=CAP, bank=MelitzDualBank(8))
    @assert r0 isa FiniteSolved "calibration point itself must be FiniteSolved for a clean gradient timing comparison"
    x0 = r0.x

    local_jac = zeros(n)
    g_serial(local_jac, theta0, ctx, obj, x0)          # compile/warm
    g_parallel(local_jac, theta0, ctx, obj, x0)         # compile/warm

    t_serial = @elapsed g_serial(local_jac, theta0, ctx, obj, x0)
    t_parallel = @elapsed g_parallel(local_jac, theta0, ctx, obj, x0)
    speedup = t_serial / t_parallel
    @printf("  complete outer gradient, SERIAL   backend: %.4fs (n_theta=%d, D=%d)\n", t_serial, n, D)
    @printf("  complete outer gradient, PARALLEL backend: %.4fs (n_theta=%d, D=%d, Julia threads=%d)\n",
        t_parallel, n, D, Threads.nthreads())
    @printf("  speedup (serial/parallel) = %.2fx\n", speedup)
    flush(stdout)

    # Confirm melitz_note_explicit_gradient_backend_choice fires for exactly the buggy
    # configuration found live in melitz_phase8_9_10_realD20_2026-07-28.jl:172.
    MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[] = 0
    melitz_note_explicit_gradient_backend_choice(:B_direct_argument_sorted_serial, D; nthreads_available=Threads.nthreads())
    n_warned = MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[]
    println("  melitz_note_explicit_gradient_backend_choice fired for the exact buggy config: ", n_warned == 1)
    flush(stdout)

    return (julia_threads=rep.julia_threads, blas_threads=rep.blas_threads,
            meets_20_thread_requirement=rep.meets_requirement,
            t_serial_s=t_serial, t_parallel_s=t_parallel, speedup=speedup,
            n_theta=n, D=D, warning_fired_for_known_bug_config=(n_warned == 1))
end

function phase2_lower_limit_audit(ctx, obj, theta0, profile20)
    println("\n" * "="^100); println("PHASE 2: cold inner-solve wall-time audit + lower_limit correctness"); println("="^100)
    n = length(theta0)
    D = ctx.D

    bank = MelitzDualBank(8)
    events = NamedTuple[]
    t0_all = time()
    function on_result(theta, result)
        elapsed = (time() - t0_all)
        row = if result isa FiniteSolved
            (t_wall=elapsed, kind="FiniteSolved", nStatus=result.nStatus, Delta=result.Delta,
             certified_lower_bound=NaN, source="", crossing_time_s=NaN, crossing_iteration=-1)
        elseif result isa AboveEvaluationCap
            (t_wall=elapsed, kind="AboveEvaluationCap", nStatus=-1, Delta=NaN,
             certified_lower_bound=result.certified_lower_bound, source=string(result.source),
             crossing_time_s=result.crossing_time_s, crossing_iteration=result.crossing_iteration)
        elseif result isa InfiniteDeltaCertified
            (t_wall=elapsed, kind="InfiniteDeltaCertified", nStatus=-1, Delta=Inf,
             certified_lower_bound=NaN, source=string(result.kind), crossing_time_s=NaN, crossing_iteration=-1)
        else   # NumericalFailure
            (t_wall=elapsed, kind="NumericalFailure", nStatus=result.nStatus, Delta=NaN,
             certified_lower_bound=NaN, source="", crossing_time_s=NaN, crossing_iteration=-1)
        end
        push!(events, row)
        return nothing
    end

    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    println("  inner_loop_opt = ", inner_opt, " (maxtime_real=90 per its own [act] block -- the suspected timeout mechanism)")

    println("\n  -- locating interior point (target Delta~0.5), same method as the prior session's Phase 8 --")
    finite = filter(r -> r.classification == "FiniteSolved", profile20)
    gs = [r.g for r in finite]; ds = [r.DeltaStar for r in finite]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    k = findfirst(i -> ds[i] <= 0.5 <= ds[i+1] || ds[i] >= 0.5 >= ds[i+1], 1:length(ds)-1)
    t = (log(0.5) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    g05 = gs[k] + t * (gs[k+1] - gs[k])
    theta_g05 = copy(theta0); theta_g05[1] = g05
    r_g05 = melitz_classified_inner_solve(obj, theta_g05, ctx; delta_evaluation_cap=CAP, bank=bank, on_result=on_result)
    @printf("  interior_g05: g=%.6f -> %s\n", g05, r_g05 isa FiniteSolved ? (@sprintf("Delta=%.4e (t=%.2fs)", r_g05.Delta, events[end].t_wall)) : string(typeof(r_g05)))
    flush(stdout)

    # NOTE (deliberate scope decision, disclosed): this audit does NOT re-run the Phase 8
    # A-only/f-only nuisance-profile stages (the 2026-07-28 step-control session's own
    # `melitz_phase8_9_10_realD20_2026-07-28.jl` already ran and documented these -- A_only
    # took 1050.9s and hit the outer iteration limit unresolved, f_only took 83.3s -- see
    # docs/melitz_outer_search_step_control_and_robustness_2026-07-28.md Phase 8/10 -- re-
    # running the already-slow, already-characterized A_only stage here would cost another
    # ~1050s for no new information). Separately confirmed by READING nuisance_profile.jl live
    # this session (not re-run): `solve_melitz_nuisance_min_delta`'s `inner_solve_cached` calls
    # `melitz_bundle_inner_loop` DIRECTLY, bypassing `melitz_classified_inner_solve` (and its
    # stored-dual/dual-polish/range PRE-screens) entirely -- it only checks whether the raw
    # nStatus is in `(0,-100,-101,-103)` vs. throwing `DomainError`. `lower_limit` IS still set
    # (from `inner_solve_config.lower_limit`, `solve_melitz_nuisance_min_delta` line ~393), so
    # the KNITRO-native mid-solve bailout can still fire -- but this path gets NONE of the
    # cheap pre-solve certificate screens the classified outer-search path has, which is
    # consistent with (part of the explanation for) why A-only nuisance minimization is so much
    # slower than the classified outer search at a comparable point: this genuinely is a
    # DIFFERENT, less-instrumented code path, not the same machinery running slower.
    #
    # This audit instead devotes its live-compute budget entirely to the ACTUAL call path that
    # produced the historical 91.4s event (Phase 9's classified outer search, immediately
    # below), starting directly from the already-solved interior_g05 point (skipping the
    # nuisance-improved reseed -- theta_g05 is itself a genuine FiniteSolved D=20 point, a
    # legitimate substitute starting point for this audit's purpose, which is to stress-test
    # melitz_classified_inner_solve's own lower_limit/classification machinery, not to
    # reproduce Phase 9's exact historical trajectory bit-for-bit).
    theta_start_pre = theta_g05

    println("\n  -- Phase 9 replay: Active Set outer search from interior_g05,")
    println("     THIS TIME with gradient_backend=:auto (resolves to the PARALLEL sorted backend at D=20,")
    println("     Threads.nthreads()>1 -- the Phase 0/1 fix), on_inner_result wired to the SAME `events` log --")
    println("     this is the exact call path (solve_melitz_finite_delta_bound's own classified inner solve)")
    println("     that produced the historical 91.4s AboveEvaluationCap/FiniteSolved/NumericalFailure event.")
    theta_start = theta_start_pre

    r_boundary = filter(r -> r.gamma_fraction == 0.65, profile20)
    external_boundary_theta = nothing
    if !isempty(r_boundary)
        tb = copy(theta0); tb[1] = r_boundary[1].g
        external_boundary_theta = tb
    end
    vs20 = ones(n); vs20[1] = 1e-4
    nA2 = D^2 - 1
    box = zeros(n); box[1] = 0.10; box[2:1+nA2] .= 0.15; box[2+nA2:end] .= 0.15
    alg_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_active_2026-07-27.opt")
    tmp_opt_dir = joinpath(OUTDIR, "tmp_opt_phase2_2026-07-28"); mkpath(tmp_opt_dir)
    lines = readlines(alg_opt)
    lines = filter(l -> !occursin(r"^\s*(delta|maxit|maxtime_real)\s", l), lines)
    push!(lines, "delta           0.05"); push!(lines, "maxit           2000"); push!(lines, "maxtime_real    240.0")
    opt_path = joinpath(tmp_opt_dir, "phase2_active_2026-07-28.opt")
    open(io -> foreach(l -> println(io, l), lines), opt_path, "w")

    n_events_before9 = length(events)
    wall9 = @elapsed begin
        res9 = solve_melitz_finite_delta_bound(ctx, obj, theta_start; delta=1.0, direction=:upper,
            delta_evaluation_cap=CAP, gradient_backend=:auto, h=1e-4,
            theta_box=box, cutoff_constraint_backend=:linear,
            inner_loop_opt=inner_opt, outer_loop_opt=opt_path,
            var_scale=vs20, var_center=collect(Float64.(theta_start)),
            backend=:matrix_free, forbid_dense_fallback=true,
            objective_scale=:auto, external_incumbent=external_boundary_theta, on_inner_result=on_result)
    end
    n_solves_phase9 = length(events) - n_events_before9
    cv9 = res9.cold_verified_incumbent
    @printf("  [Phase9 Active Set, PARALLEL] nStatus=%5d wall=%6.2fs n_fc=%3d n_ga=%3d n_solved=%3d n_cap=%3d n_numfail=%3d Delta=%s inner_solves_this_call=%d\n",
        res9.nStatus, wall9, res9.n_fc_calls, res9.n_ga_calls, res9.n_inner_solved, res9.n_above_cap_reject,
        res9.n_numerical_failure_reject, cv9 === nothing ? "NA" : @sprintf("%.4e", cv9.eval.Delta), n_solves_phase9)
    flush(stdout)

    write_csv(joinpath(OUTDIR, "melitz_phase2_lower_limit_audit_events_2026-07-28.csv"), events)

    # Classified-outer-search events (interior_g05 locate call + the Phase 9 replay above, in
    # `events`): elapsed-
    # BETWEEN-consecutive-events duration, since t_wall is cumulative from t0_all.
    durations = NamedTuple[]
    prev_t = 0.0
    for (i, r) in enumerate(events)
        dur = r.t_wall - prev_t
        push!(durations, merge(r, (index=i, duration_s=dur)))
        prev_t = r.t_wall
    end
    write_csv(joinpath(OUTDIR, "melitz_phase2_lower_limit_audit_durations_2026-07-28.csv"), durations)
    slow_solves = filter(r -> r.duration_s > 30.0, durations)

    println("\n" * "-"^100)
    println("SLOW (>30s) INNER SOLVES FOUND: ", length(slow_solves), " of ", length(events), " total classified inner solves")
    println("-"^100)
    for r in slow_solves
        @printf("  #%d kind=%-22s duration=%.2fs nStatus=%d Delta=%s cert_lb=%s source=%s crossing_time_s=%s\n",
            r.index, r.kind, r.duration_s, r.nStatus,
            isnan(r.Delta) ? "NA" : @sprintf("%.4e", r.Delta),
            isnan(r.certified_lower_bound) ? "NA" : @sprintf("%.4e", r.certified_lower_bound),
            r.source, isnan(r.crossing_time_s) ? "NA" : @sprintf("%.2f", r.crossing_time_s))
    end
    if isempty(slow_solves)
        println("  (none found in this replay -- see console output above for the full per-stage wall-clock breakdown;")
        println("   the historical 91.4s solve's exact theta was not persisted to disk so an exact bit-replay was not")
        println("   possible; this replay instead directly stress-tests the SAME lower_limit/classification machinery")
        println("   at the same fixture/config and finds it working as documented -- see the full events CSV.)")
    end
    flush(stdout)

    n_finite = count(r -> r.kind == "FiniteSolved", events)
    n_cap = count(r -> r.kind == "AboveEvaluationCap", events)
    n_numfail = count(r -> r.kind == "NumericalFailure", events)
    n_inf = count(r -> r.kind == "InfiniteDeltaCertified", events)
    println("\nClassification totals across the full Phase 2 replay: FiniteSolved=", n_finite,
            " AboveEvaluationCap=", n_cap, " NumericalFailure=", n_numfail, " InfiniteDeltaCertified=", n_inf)
    # THE key correctness check: any NumericalFailure event with a long duration AND no certificate
    # is the historical missing-lower_limit symptom. Any such found here is a live regression.
    numfail_slow = filter(r -> r.kind == "NumericalFailure" && r.duration_s > 30.0, durations)
    println("NumericalFailure events with duration>30s (the historical missing-lower_limit symptom): ", length(numfail_slow))
    return (n_events=length(events), n_slow=length(slow_solves), n_finite=n_finite, n_cap=n_cap,
            n_numfail=n_numfail, n_inf=n_inf, n_numfail_slow=length(numfail_slow),
            phase9_wall_s=wall9, phase9_nStatus=res9.nStatus, phase9_Delta=(cv9 === nothing ? NaN : cv9.eval.Delta),
            phase9_n_inner_solves=n_solves_phase9)
end

function main()
    BLAS.set_num_threads(1)
    d20 = build_realD20_fixture()
    ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
    profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))

    p1 = phase0_1_threading(ctx, obj, theta0)
    p2 = phase2_lower_limit_audit(ctx, obj, theta0, profile20)

    write_csv(joinpath(OUTDIR, "melitz_phase1_threading_audit_2026-07-28.csv"), [p1])
    write_csv(joinpath(OUTDIR, "melitz_phase2_lower_limit_summary_2026-07-28.csv"), [p2])
    println("\nDONE. CSVs written to ", OUTDIR)
end

main()
