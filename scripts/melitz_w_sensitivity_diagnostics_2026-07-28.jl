# User-directed diagnostic (2026-07-28 outer-search gamma-profile session): how does the
# real-D20 fixture behave as W scales from the production default (80,000) up toward
# 2,560,000 (32x)? Motivating concern (user's own hypothesis, stated directly): what matters
# economically is the behavior PAST each cell's cutoff, but only a shrinking fraction of raw
# draws land there -- so the EFFECTIVE support size near the participation margin may be much
# smaller than W itself, and may not grow proportionally with W, meaning finite-draw/
# conditioning issues could persist or worsen even as W grows nominally huge.
#
# Reports, at each W on a doubling ladder, WITHOUT ever materializing a dense G (no SVD-based
# conditioning diagnostic here -- that would need a dense W x 401 matrix + a dense QR/SVD,
# genuinely expensive at W=2.56M and not necessary: the cheap min_active_draw_count /
# cell_participation_diagnostics scans below answer the user's actual question directly):
#   - min raw active-draw count across all D^2 cells, and the worst (o,d) cell
#   - LFD-reweighted EFFECTIVE active count at that same worst cell (post-solve)
#   - a fixed low-probability benchmark cell (the (3,14)/bra->kor cell the 2026-07-24 report
#     flagged as the worst cell at W<=20,000) tracked across the whole ladder
#   - fresh cold inner-solve wall-clock/nStatus/Delta (the actual production matrix-free path)
#   - full 798-coordinate outer-gradient wall-clock + bytes, BOTH the current default
#     (:B_direct_argument_sorted_parallel, full W-length copy per coordinate) and the
#     alternative (:B_direct_argument_touched_row_parallel, touches only the crossing-active
#     rows) -- directly answers whether touched-row becomes worth adopting as W grows
#   - process RSS (from /proc/self/status) before/after each W step, so memory growth is
#     visible in the log without relying on an external `ps` check alone
#
# SAFETY: prints progress with flush(stdout) after every step (early-checkin friendly); a
# W step that takes > MAX_STEP_SECONDS aborts the ladder rather than proceeding to a larger,
# even slower W blind. OPENBLAS_NUM_THREADS/OMP_NUM_THREADS must be 1 (hard-cap convention);
# Julia threads=20 (production convention) for the gradient-backend comparison.
#
# Usage: julia --project=. -t 20 scripts/melitz_w_sensitivity_diagnostics_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra, Statistics
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
const W_LADDER = [80_000, 160_000, 320_000, 640_000, 1_280_000, 2_560_000]
const MAX_STEP_SECONDS = 600.0   # abort the ladder (not the process) if one W step exceeds this

function rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                return parse(Float64, split(line)[2]) / 1024
            end
        end
    catch
    end
    return NaN
end

function main()
    println("Threads.nthreads() = ", Threads.nthreads(), "  BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())
    real_dir = joinpath(REPO, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")

    rows = NamedTuple[]
    for W in W_LADDER
        println("\n" * "="^100); @printf("W = %d  (RSS before = %.1f MB)\n", W, rss_mb()); println("="^100)
        flush(stdout)
        t_step0 = time()
        BLAS.set_num_threads(1)

        z_draws = pareto_draws(W, calib.D, calib.theta_star; seed=calib.seed)
        p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_parallel)
        D = ctx20.D

        # --- cheap, no-dense-G support diagnostics ---
        min_count, worst_cell = min_active_draw_count(p20, eq20, z_draws)
        bra_kor = (D >= 14) ? (3, 14) : worst_cell
        bra_kor_count = count(>(eq20.cutoff[bra_kor[1], bra_kor[2]]), @view z_draws[:, bra_kor[1]])
        @printf("  support: min_active_count=%d at cell %s (raw prob~%.2e); (3,14) bra->kor count=%d (raw prob~%.2e)\n",
            min_count, worst_cell, min_count / W, bra_kor_count, bra_kor_count / W)
        flush(stdout)

        # --- production matrix-free bundle + cold inner solve ---
        op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
        obj = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
            outer_constr_index=ctx20.moment_layout.num_moments + 1,
            lower_limit=-10.0, inner_loop_opt=inner_opt, outer_loop_opt=ctx20.outer_loop_opt,
            hessian_backend=:structured_parallel)
        theta0 = melitz_reduce_theta(p20, ctx20)
        n = length(theta0); nA = D^2 - 1

        t_solve = @elapsed r0 = evaluate_melitz_delta(theta0, ctx20, obj; cold=true, store_G=false)
        @printf("  cold inner solve: wall=%.3fs  nStatus=%d  Delta=%.6e  verified=%s\n",
            t_solve, r0.nStatus, r0.Delta, r0.verified)
        flush(stdout)

        eff_active = NaN
        if r0.verified
            cpd = cell_participation_diagnostics(p20, eq20, z_draws; weights=r0.weights)
            eff_active = cpd.effective_active[worst_cell[1], worst_cell[2]]
            @printf("  LFD-effective active count at worst cell %s: %.3f (raw was %d)\n", worst_cell, eff_active, min_count)
        end
        flush(stdout)

        elapsed_so_far = time() - t_step0
        if elapsed_so_far > MAX_STEP_SECONDS
            @printf("  ABORT LADDER: step already took %.1fs (> %.1fs cap) before gradient timing -- stopping here.\n", elapsed_so_far, MAX_STEP_SECONDS)
            push!(rows, (W=W, min_active_count=min_count, worst_cell_o=worst_cell[1], worst_cell_d=worst_cell[2],
                worst_cell_effective_active=eff_active, bra_kor_count=bra_kor_count,
                cold_solve_wall_s=t_solve, cold_nStatus=r0.nStatus, cold_Delta=r0.Delta,
                grad_sorted_wall_s=NaN, grad_touched_wall_s=NaN, grad_sorted_bytes=-1, grad_touched_bytes=-1,
                rss_mb=rss_mb(), aborted_before_gradient=true))
            flush(stdout)
            break
        end

        # --- full outer-gradient timing: sorted vs touched-row, 20 threads ---
        BLAS.set_num_threads(1)   # ambient Julia threads=20 does the parallelism; BLAS stays 1 (hard cap)
        x0 = r0.dual_x
        gsorted_p = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
        gtouched_p = make_melitz_gradient_delta_direct_touched_row_parallel(1e-4)
        g1 = zeros(n); g2 = zeros(n)
        gsorted_p(g1, theta0, ctx20, obj, x0)    # warmup / JIT
        gtouched_p(g2, theta0, ctx20, obj, x0)
        maxrel = maximum(abs.(g1 .- g2) ./ max.(abs.(g1), 1.0))

        t_sorted = @elapsed gsorted_p(g1, theta0, ctx20, obj, x0)
        t_touched = @elapsed gtouched_p(g2, theta0, ctx20, obj, x0)
        b_sorted = @allocated gsorted_p(g1, theta0, ctx20, obj, x0)
        b_touched = @allocated gtouched_p(g2, theta0, ctx20, obj, x0)
        @printf("  outer gradient (798 coords): sorted=%.3fs (%d B)  touched_row=%.3fs (%d B)  speedup=%.3fx  maxreldiff=%.2e\n",
            t_sorted, b_sorted, t_touched, b_touched, t_sorted / t_touched, maxrel)
        flush(stdout)

        rss_after = rss_mb()
        @printf("  RSS after this W step = %.1f MB\n", rss_after)
        push!(rows, (W=W, min_active_count=min_count, worst_cell_o=worst_cell[1], worst_cell_d=worst_cell[2],
            worst_cell_effective_active=eff_active, bra_kor_count=bra_kor_count,
            cold_solve_wall_s=t_solve, cold_nStatus=r0.nStatus, cold_Delta=r0.Delta,
            grad_sorted_wall_s=t_sorted, grad_touched_wall_s=t_touched,
            grad_sorted_bytes=b_sorted, grad_touched_bytes=b_touched,
            rss_mb=rss_after, aborted_before_gradient=false))

        # Explicitly drop references and GC before the next (larger) W -- keeps peak RSS to
        # roughly one W-step's worth of buffers rather than accumulating across the ladder.
        obj = nothing; op20 = nothing; ctx20 = nothing; z_draws = nothing; g1 = nothing; g2 = nothing
        GC.gc()
        @printf("  RSS after GC = %.1f MB  (step wall=%.1fs)\n", rss_mb(), time() - t_step0)
        flush(stdout)
    end

    outfile = joinpath(OUTDIR, "melitz_w_sensitivity_2026-07-28.csv")
    open(outfile, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
    BLAS.set_num_threads(1)
    println("\nDONE. CSV written to ", outfile)
    return rows
end

main()
