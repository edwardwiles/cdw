# 2026-07-28 outer-search gradient/redundancy/sensitivity continuation, Phases 9-10.
#
# Phase 9: real-D20 nested block search (gamma-only / gamma+technology / gamma+participation
# / gamma+technology+participation) from the SAME interior profile point, SQP,
# objective_scale=:auto, g-radius=0.5 (the 2026-07-28 step-control session's own Phase 3
# finding -- the ONLY configuration that session found reaching genuine nStatus=0 convergence
# at real D=20 -- reused, not re-derived), SEPARATE tight nuisance radii for technology/
# participation (not one shared 0.15 box for all 797 nuisance coordinates, per the governing
# prompt's explicit instruction). A small staged radius comparison ({1e-4,3e-4,1e-3}), not a
# large grid: the primary 4-model comparison fixes both nuisance radii at the middle value
# (3e-4), then a light follow-up sweeps ONE radius at a time holding the other fixed.
#
# Phase 10: nuisance minimization at the SAME two interior points (Delta~0.5,~0.8, matching
# the 2026-07-28 step-control session's own Phase 8 points) in A/f coordinates vs a
# composite/cutoff-flavored diagnostic basis. Scope disclosure: rather than build a genuine
# re-parameterized KNITRO registration for a hand-derived xi=theta_star*logA+beta*logf / q=
# log(cutoff) analytic coordinate change (a nontrivial new derivation this session's own time
# budget does not support doing safely), this uses the EMPIRICAL basis Phase 7 already
# produced (the right-singular-vectors of the local moment-response Jacobian, i.e. the
# data-driven "near-null" and "steep" directions in log-A/log-f space) as the diagnostic
# alternate coordinate system -- the model-free version of exactly the same underlying idea
# (does moving along a redundant/near-null direction buy more DeltaStar slack per unit raw
# movement than moving along an arbitrary A/f direction). Compared via the SAME directional-
# probe-plus-full-reoptimization machinery Phase 6/8 already use, not a full re-parameterized
# outer KNITRO search -- disclosed, not silently substituted for the literal prompt wording.
#
# Usage: julia --project=. -t 20 scripts/melitz_phase9_10_realD20_nested_and_basis_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(@__DIR__, "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const REPO = dirname(@__DIR__)
const OUTDIR = joinpath(REPO, "docs", "key_results")
mkpath(OUTDIR)
const TMPOPT = joinpath(OUTDIR, "tmp_opt_phase9_10_2026-07-28")
mkpath(TMPOPT)
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

function make_opt(base, tag; delta, maxit, maxtime_real)
    lines = readlines(base)
    lines = filter(l -> !occursin(r"^\s*(delta|maxit|maxtime_real)\s", l), lines)
    push!(lines, @sprintf("delta           %.6g", delta))
    push!(lines, @sprintf("maxit           %d", maxit))
    push!(lines, @sprintf("maxtime_real    %.1f", maxtime_real))
    path = joinpath(TMPOPT, tag * ".opt")
    open(io -> foreach(l -> println(io, l), lines), path, "w")
    return path
end

function block_box(n, D, g_radius, A_radius, f_radius)
    nA = D^2 - 1
    b = zeros(n)
    b[1] = g_radius
    b[2:1+nA] .= A_radius
    b[2+nA:end] .= f_radius
    return b
end

function movement_norms(theta_final, theta0, D)
    nA = D^2 - 1
    dlogA = norm(theta_final[2:1+nA] - theta0[2:1+nA])
    dlogf = norm(theta_final[2+nA:end] - theta0[2+nA:end])
    return dlogA, dlogf
end

# ============================================================================
# Phase 9
# ============================================================================

function phase9_run(ctx, obj, theta_start, sqp_base, inner_opt, blockname, box; maxtime_real=180.0)
    n = length(theta_start)
    D = ctx.D
    vs = ones(n); vs[1] = 1e-4
    opt = make_opt(sqp_base, "phase9_$(blockname)_$(hash(box))"; delta=0.05, maxit=1000, maxtime_real=maxtime_real)

    MELITZ_PROFILE[] = true
    melitz_profile_reset!()
    t0 = time()
    res = solve_melitz_finite_delta_bound(ctx, obj, theta_start; delta=1.0, direction=:upper,
        delta_evaluation_cap=CAP, gradient_backend=:auto, theta_box=box,
        cutoff_constraint_backend=:linear, inner_loop_opt=inner_opt, outer_loop_opt=opt,
        var_scale=vs, var_center=collect(Float64.(theta_start)),
        backend=:matrix_free, forbid_dense_fallback=true, objective_scale=:auto)
    wall = time() - t0
    prof = melitz_profile_summary()
    gradient_time = sum(r.total_s for r in prof if r.category == :ga_divergence_gradient; init=0.0)
    # melitz_record_seconds_outcome! names categories Symbol(category, :_, outcome), e.g.
    # :inner_solve_warm_success/:inner_solve_cache_hit/:inner_solve_above_evaluation_cap/... --
    # aggregate every :inner_solve_* bucket, not a plain :inner_solve match (which would
    # silently match nothing and always report 0.0).
    inner_time = sum(r.total_s for r in prof if startswith(string(r.category), "inner_solve"); init=0.0)
    MELITZ_PROFILE[] = false

    cv = res.cold_verified_incumbent
    if cv === nothing
        return (block=blockname, nStatus=res.nStatus, wall=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
            n_inner_solved=res.n_inner_solved, n_above_cap=res.n_above_cap_reject,
            n_numfail=res.n_numerical_failure_reject, dg=0.0, dlogA=0.0, dlogf=0.0,
            gamma_prime=NaN, kappa_ratio=NaN, GT=NaN, DeltaStar=NaN,
            gradient_time=gradient_time, inner_time=inner_time, native_knitro_time=(wall - gradient_time - inner_time))
    end
    theta_final = cv.eval.theta_free
    dg = theta_final[1] - theta_start[1]
    dlogA, dlogf = movement_norms(theta_final, theta_start, D)
    wm = melitz_welfare_metrics_from_g(theta_final[1], ctx)
    return (block=blockname, nStatus=res.nStatus, wall=wall, n_fc=res.n_fc_calls, n_ga=res.n_ga_calls,
        n_inner_solved=res.n_inner_solved, n_above_cap=res.n_above_cap_reject,
        n_numfail=res.n_numerical_failure_reject, dg=dg, dlogA=dlogA, dlogf=dlogf,
        gamma_prime=wm.gamma_prime, kappa_ratio=wm.kappa_ratio, GT=wm.gains_from_trade, DeltaStar=cv.eval.Delta,
        gradient_time=gradient_time, inner_time=inner_time, native_knitro_time=(wall - gradient_time - inner_time))
end

function phase9_main(ctx, obj, theta_start, D, n, sqp_base, inner_opt)
    println("="^100); println("PHASE 9: nested D20 block search from the interior point"); println("="^100)
    rows = NamedTuple[]
    tech_r = 3e-4; part_r = 3e-4
    box_gamma = block_box(n, D, 0.5, 0.0, 0.0)
    box_tech = block_box(n, D, 0.5, tech_r, 0.0)
    box_part = block_box(n, D, 0.5, 0.0, part_r)
    box_full = block_box(n, D, 0.5, tech_r, part_r)
    for (name, box) in (("gamma_only", box_gamma), ("gamma_technology", box_tech),
                         ("gamma_participation", box_part), ("gamma_technology_participation", box_full))
        r = phase9_run(ctx, obj, theta_start, sqp_base, inner_opt, name, box)
        push!(rows, merge((tech_radius=tech_r, part_radius=part_r), r))
        @printf("  [%-30s] nStatus=%5d wall=%6.1fs dg=%+.5f dlogA=%.3e dlogf=%.3e Delta=%s\n",
            name, r.nStatus, r.wall, r.dg, r.dlogA, r.dlogf, isnan(r.DeltaStar) ? "NA" : @sprintf("%.4e", r.DeltaStar))
        flush(stdout)
        write_csv(joinpath(OUTDIR, "melitz_phase9_realD20_nested_block_search_2026-07-28.csv"), rows)
    end

    println("\n  -- light radius sensitivity (staged, not a grid): sweep technology radius at fixed participation=3e-4, then vice versa --")
    for tr in (1e-4, 1e-3)
        box = block_box(n, D, 0.5, tr, part_r)
        r = phase9_run(ctx, obj, theta_start, sqp_base, inner_opt, "gamma_technology_participation", box; maxtime_real=120.0)
        push!(rows, merge((tech_radius=tr, part_radius=part_r), r))
        @printf("  [tech_radius=%.0e] nStatus=%5d wall=%6.1fs dg=%+.5f Delta=%s\n", tr, r.nStatus, r.wall, r.dg,
            isnan(r.DeltaStar) ? "NA" : @sprintf("%.4e", r.DeltaStar))
        flush(stdout)
    end
    for pr in (1e-4, 1e-3)
        box = block_box(n, D, 0.5, tech_r, pr)
        r = phase9_run(ctx, obj, theta_start, sqp_base, inner_opt, "gamma_technology_participation", box; maxtime_real=120.0)
        push!(rows, merge((tech_radius=tech_r, part_radius=pr), r))
        @printf("  [part_radius=%.0e] nStatus=%5d wall=%6.1fs dg=%+.5f Delta=%s\n", pr, r.nStatus, r.wall, r.dg,
            isnan(r.DeltaStar) ? "NA" : @sprintf("%.4e", r.DeltaStar))
        flush(stdout)
    end
    write_csv(joinpath(OUTDIR, "melitz_phase9_realD20_nested_block_search_2026-07-28.csv"), rows)
    return rows
end

# ============================================================================
# Phase 10 (scoped, see file header)
# ============================================================================

function phase10_main(ctx, obj, theta0, D, n, profile20, inner_opt, svd_basis)
    println("\n" * "="^100); println("PHASE 10 (scoped): A/f vs empirical-SVD-basis nuisance directions at two interior points"); println("="^100)
    finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
    gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    rows = NamedTuple[]
    for target in (0.5, 0.8)
        k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
        t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
        g_pt = gs[k] + t * (gs[k+1] - gs[k])
        theta_pt = copy(theta0); theta_pt[1] = g_pt
        bank = MelitzDualBank(8)
        r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=CAP, bank=bank)
        r0 isa FiniteSolved || (println("  SKIP target=$target: base point not FiniteSolved"); continue)
        Delta0 = r0.Delta
        nA = D^2 - 1

        # A/f-coordinate directions: 3 random unit directions in the raw (A,f) subspace.
        af_dirs = [begin
            v = zeros(n); v[2:end] .= randn(MersenneTwister(1000 + i), n - 1); v ./ norm(v)
        end for i in 1:3]

        # empirical-SVD-basis directions: the smallest (most redundant) and largest (steepest)
        # singular directions from Phase 7's Jacobian, embedded into the full n-length theta
        # vector (Phase 7's Jacobian columns are indexed 1:n exactly like theta -- direct reuse,
        # no re-derivation).
        svd_dirs = [svd_basis[:, 1], svd_basis[:, end]]   # smallest sv direction, largest sv direction
        svd_names = ["svd_near_null", "svd_steepest"]

        step_norm = 0.05
        for (label, dvec) in vcat([("af_random_$i", d) for (i, d) in enumerate(af_dirs)],
                                    collect(zip(svd_names, svd_dirs)))
            for sign in (1.0, -1.0)
                theta_new = theta_pt .+ sign * step_norm .* dvec
                r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank)
                finite_new = r_new isa FiniteSolved
                push!(rows, (target_delta=target, g=g_pt, direction=label, sign=sign, step_norm=step_norm,
                    Delta0=Delta0, finite_after_step=finite_new,
                    Delta_after_step=(finite_new ? r_new.Delta : NaN),
                    delta_reduction=(finite_new ? Delta0 - r_new.Delta : NaN), status=string(typeof(r_new))))
            end
        end
        write_csv(joinpath(OUTDIR, "melitz_phase10_realD20_coordinate_basis_comparison_2026-07-28.csv"), rows)
        flush(stdout)
    end
    return rows
end

function main()
    BLAS.set_num_threads(1)
    d20 = build_realD20_fixture()
    ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
    n = length(theta0); D = ctx.D
    sqp_base = joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt")
    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))

    finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
    gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    k = findfirst(i -> ds[i] <= 0.5 <= ds[i+1] || ds[i] >= 0.5 >= ds[i+1], 1:length(ds)-1)
    t = (log(0.5) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    g_start = gs[k] + t * (gs[k+1] - gs[k])
    theta_start = copy(theta0); theta_start[1] = g_start
    println("Phase 9 starting point: g=", g_start, " (target Delta~0.5)")

    phase9_main(ctx, obj, theta_start, D, n, sqp_base, inner_opt)

    println("\nBuilding a LIGHTER D20 empirical basis for Phase 10.")
    println("Scope disclosure: the full n=$n-coordinate x W*num_moments Jacobian (Phase 7's D4")
    println("construction, scaled to D20, would need $n calls to melitz_moment_directional_derivative,")
    println("each an O(D^2*W) SERIAL loop (not the optimized parallel gradient backend) -- too slow")
    println("for this session's budget. Instead: a random subset of n_sub A/f coordinates (g excluded,")
    println("matching Phase 7's own A/f-only redundancy scope) is used to build a REDUCED empirical")
    println("Jacobian -- still exact for the sampled coordinates, just lower-resolution than a full SVD.")
    n_sub = min(80, n - 1)
    sub_coords = sort(randperm(MersenneTwister(4242), n - 1)[1:n_sub] .+ 1)   # coordinates 2:n only (A/f block)
    Wsub = 4000
    idxsub = 1:Wsub
    Jsub = zeros(Wsub * ctx.moment_layout.num_moments, n_sub)
    ei = zeros(n)
    for (jj, kk) in enumerate(sub_coords)
        ei[kk] = 1.0
        dG = melitz_moment_directional_derivative(theta_start, ei, ctx, obj)
        Jsub[:, jj] .= vec(dG[idxsub, :])
        ei[kk] = 0.0
    end
    Usvd = svd(Jsub)
    # Usvd.V is n_sub x n_sub, singular values DESCENDING -- embed the extreme (smallest/
    # largest singular value) right-singular vectors back into full n-length theta-space
    # vectors (zero at every coordinate NOT in sub_coords).
    v_near_null = zeros(n); v_near_null[sub_coords] .= Usvd.V[:, end]
    v_steepest = zeros(n); v_steepest[sub_coords] .= Usvd.V[:, 1]
    svd_basis = zeros(n, 2)
    svd_basis[:, 1] .= v_near_null ./ norm(v_near_null)
    svd_basis[:, end] .= v_steepest ./ norm(v_steepest)
    write_csv(joinpath(OUTDIR, "melitz_phase10_realD20_svd_summary_2026-07-28.csv"),
        [(index=i, singular_value=s, n_sub=n_sub, n_sub_coordinates_of=n - 1) for (i, s) in enumerate(Usvd.S)])

    phase10_main(ctx, obj, theta0, D, n, profile20, inner_opt, svd_basis)

    println("\nDONE. CSVs written to ", OUTDIR)
end

main()
