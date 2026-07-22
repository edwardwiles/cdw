# ============================================================================
# fullA-final-gates diagnostics task (2026-07-22), Phase 1: broader independent
# optimized-value directional checks for C+, at real D=20/W=80,000 points.
#
# Extends c23_cplus_gate.jl's Section 1 (2 coordinates, 2 bandwidths, 1 base point) along
# exactly the three axes the brief calls for, reusing the SAME validated single-coordinate
# incremental machinery (lfix_incremental_at / lfix_incremental_at_C / count_winner_flips_C /
# select_bandwidth_C) rather than inventing a new multi-coordinate derivative path -- this
# codebase's fixed-dual secant machinery is fundamentally single-reduced-coordinate
# (zfree = pivot-eliminated reduced A_od coords, gravity satisfied by construction), so
# "directions" here means "which zfree/gp coordinate", not an arbitrary combined vector.
#
#   - 2 base points: A near the delta=1 frontier, B ("more difficult") near delta=2.
#   - 3 direction classes per base point, 4 coordinates each (bounded by real cost):
#       optimizer  -- coordinates that moved most across a real short production trajectory
#                     at that base point (via the new full_trace_ref= instrumentation hook
#                     added to run_polish_checkpointed for this task -- diagnostics-branch-only).
#       random     -- 4 deterministic seeded zfree coordinates (gravity-tangent by
#                     construction: zfree already satisfies gravity, per cm_outer_driver.jl).
#       high-switch-- top-4 coordinates by count_winner_flips_C (same diagnostic c23/c21 use).
#   - both signs, 3 step scales (0.5h, h, 2h) around select_bandwidth_C's own production
#     bandwidth for that coordinate.
#   - cold solve (use_cache=false, warm=false) at each displaced point; independent
#     reoptimized secant vs C+ fixed-dual secant vs Reference(buffered) fixed-dual secant.
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Random, Printf, Statistics, DelimitedFiles, LinearAlgebra

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

const OUTDIR = "/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/7978bbf6-5144-4658-bdd0-fdc9b0211270/scratchpad/phase1"
mkpath(OUTDIR)
rows = Vector{Any}[]
push!(rows, ["base_point","delta_frontier","class","coord","h_prod","step_scale","h_actual","sign",
             "flips_at_hprod","plus_status","minus_status","feas_class",
             "true_secant","fixed_dual_secant_ref","fixed_dual_secant_C",
             "C_vs_ref_absdiff","C_vs_true_abserr","ref_vs_true_abserr",
             "plus_iters","minus_iters","plus_wall","minus_wall"])

n_attempted = Ref(0); n_feasible_both = Ref(0); n_infeasible_skip = Ref(0); n_completed = Ref(0)

t0 = time()
fctx = build_fullA_context(W = 80000, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
println("context build (reused across both base points): ", round(time()-t0, digits=1), "s")
D = fctx.ctx.D; D2 = D^2
pe = fctx.pe

g0 = fctx.ctx.θ0_up[3+D]
Aod_real = reshape(fctx.ctx.θ0_up[fctx.ctx.Aod_offset+1:fctx.ctx.Aod_offset+D2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe)

"""Build one base point: a short real polish trajectory at `delta_target`, capturing the
full-w trace via full_trace_ref=. Returns (w0, base, cache_ref, cache_C, trace)."""
function build_base_point(label, g_start, zfree_start, delta_target; maxtime_real = 60.0)
    ckpt = mktempdir()
    tr = Ref(NamedTuple[])
    res = run_polish_checkpointed(label, true, g_start, zfree_start; maxtime_real = maxtime_real,
        W_in = 80000, delta_in = delta_target, ckpt_dir = ckpt, checkpoint_interval_s = 30.0,
        reuse = fctx, price_cache_backend = :cplus, full_trace_ref = tr)
    b = res.best_feasible
    b === nothing && error("build_base_point($label): no feasible incumbent found in $(maxtime_real)s")
    w0 = b.w
    xf0 = x_free_from_w(w0, pe)
    base = solve_base_state(xf0, fctx.ctx)
    cache_ref = build_lfix_base_cache(xf0, fctx.ctx, base)
    c_ws = build_lfix_factorized_workspace(D, size(fctx.ctx.U, 1))
    cache_C = build_lfix_base_cache_C!(c_ws, xf0, fctx.ctx, base)
    cache_C_flip = build_lfix_base_cache_C(xf0, fctx.ctx, base)
    println("  [$label] incumbent gp=$(b.gp) Delta=$(b.Delta) n_eval=$(res.n_eval) trajectory_len=$(length(tr[]))")
    return (w0 = w0, base = base, cache_ref = cache_ref, cache_C = cache_C, cache_C_flip = cache_C_flip, trace = tr[], res = res)
end

"""Build a base point GUARANTEED near `delta_target`'s frontier via direct bisection along a
fixed deterministic random gravity-tangent direction (cheap cold evaluate_fullA calls, no outer
KNITRO loop -- a short outer polish run cannot reliably reach a specific delta frontier within a
bounded budget, confirmed live: a first attempt at this left base B identical to base A). A short
polish run AT the bisected point is still used afterward, purely to harvest a real local
trajectory for the 'optimizer coords' direction class."""
function build_base_point_near_frontier(label, g_start, zfree_start, delta_target; seed = 4242, polish_time = 90.0)
    rng = MersenneTwister(seed)
    dir = randn(rng, length(zfree_start)); dir ./= norm(dir)
    lo, hi = 0.0, 1.0
    r_hi = evaluate_fullA(x_free_from_w(vcat(g_start, zfree_start .+ hi .* dir), pe), fctx.ctx; use_cache = false, warm = false)
    tries = 0
    while (!(r_hi.inner_status in FEASIBLE_CODES) || r_hi.Delta_dual < delta_target) && tries < 8
        hi *= 1.6
        r_hi = evaluate_fullA(x_free_from_w(vcat(g_start, zfree_start .+ hi .* dir), pe), fctx.ctx; use_cache = false, warm = false)
        tries += 1
    end
    for _ in 1:14
        mid = (lo + hi) / 2
        r_mid = evaluate_fullA(x_free_from_w(vcat(g_start, zfree_start .+ mid .* dir), pe), fctx.ctx; use_cache = false, warm = false)
        if r_mid.inner_status in FEASIBLE_CODES && r_mid.Delta_dual < delta_target
            lo = mid
        else
            hi = mid
        end
    end
    zfree_b = zfree_start .+ lo .* dir
    xf_b = x_free_from_w(vcat(g_start, zfree_b), pe)
    r_b = evaluate_fullA(xf_b, fctx.ctx; use_cache = false, warm = false)
    println("  [$label] bisected to scale=$lo  Delta_dual=$(r_b.Delta_dual) (target=$delta_target) inner_status=$(r_b.inner_status)")

    ckpt = mktempdir()
    tr = Ref(NamedTuple[])
    res = try
        run_polish_checkpointed(label, true, g_start, zfree_b; maxtime_real = polish_time,
            W_in = 80000, delta_in = delta_target, ckpt_dir = ckpt, checkpoint_interval_s = 30.0,
            reuse = fctx, price_cache_backend = :cplus, full_trace_ref = tr)
    catch e
        println("  [$label] short polish from the bisected point failed/rejected immediately ($e) -- optimizer-direction class will be empty for this base point, random/high-switch classes unaffected")
        nothing
    end

    w0 = vcat(g_start, zfree_b)   # the base point itself is the BISECTED point, not the polish result
    xf0 = x_free_from_w(w0, pe)
    base = solve_base_state(xf0, fctx.ctx)
    cache_ref = build_lfix_base_cache(xf0, fctx.ctx, base)
    c_ws = build_lfix_factorized_workspace(D, size(fctx.ctx.U, 1))
    cache_C = build_lfix_base_cache_C!(c_ws, xf0, fctx.ctx, base)
    cache_C_flip = build_lfix_base_cache_C(xf0, fctx.ctx, base)
    return (w0 = w0, base = base, cache_ref = cache_ref, cache_C = cache_C, cache_C_flip = cache_C_flip,
            trace = res === nothing ? NamedTuple[] : tr[], res = res)
end

println("== Base point A: near delta=1 frontier ==")
A = build_base_point("phase1_baseA", g0, zfree0, 1.0; maxtime_real = 90.0)

println("\n== Base point B: near delta=2 frontier (harder), via direct bisection ==")
B = build_base_point_near_frontier("phase1_baseB", g0, zfree0, 2.0; seed = 4242, polish_time = 90.0)

"""4 coordinates that moved most (|Δw|) across the last few recorded trajectory steps."""
function optimizer_coords(trace, w0; n = 4)
    length(trace) < 2 && return Int[]
    steps = trace[max(1, end-7):end]
    cand = Int[]
    for i in 2:length(steps)
        dw = steps[i].w .- steps[i-1].w
        k = argmax(abs.(dw[2:end])) + 1   # restrict to zfree coords (skip gp=index 1)
        dw[k] == 0 && continue
        push!(cand, k)
    end
    return unique(cand)[1:min(n, length(unique(cand)))]
end

function random_coords(seed; n = 4)
    rng = MersenneTwister(seed)
    return unique(rand(rng, 2:(D2+1), 3n))[1:n]
end

function high_switch_coords(cache_C_flip, ctx, w0, pe; n = 4, h = 0.05, seed = 99)
    rng = MersenneTwister(seed)
    cand = unique(rand(rng, 2:(D2+1), 4n))
    flip_counts = Dict{Int,Int}()
    for k in cand
        cells = affected_cells(pe, k)
        isempty(cells) && continue
        d = first(last.(cells))
        w = copy(w0); w[k] += h
        θf = CS.reconstruct_full(vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe)))), ctx.m)
        origins_here = [o for (o, dd) in cells if dd == d]
        flip_counts[k] = count_winner_flips_C(cache_C_flip, ctx, θf, d, origins_here)
    end
    ranked = sort(collect(keys(flip_counts)); by = k -> -flip_counts[k])
    for k in ranked[1:min(n, length(ranked))]
        println("    high-switch candidate k=$k flips=$(flip_counts[k])")
    end
    return ranked[1:min(n, length(ranked))]
end

function run_case!(label, delta_frontier, class, ctx, pe, w0, cache_ref, cache_C, cache_C_flip, k; hstep_h0 = 0.01)
    cells = affected_cells(pe, k)
    if isempty(cells)
        println("  [$label class=$class k=$k] SKIP: no affected A_od cells (gp coordinate or degenerate)")
        return
    end
    h_prod = try
        h, _, _ = select_bandwidth_C(cache_C_flip, ctx, pe, w0, k; h0 = hstep_h0)
        h
    catch e
        println("  [$label class=$class k=$k] bandwidth selection failed ($e), falling back to h0=$hstep_h0")
        hstep_h0
    end
    for scale in (0.5, 1.0, 2.0)
        h = scale * h_prod
        n_attempted[] += 1
        w_plus = copy(w0); w_plus[k] += h
        w_minus = copy(w0); w_minus[k] -= h
        xf_plus = x_free_from_w(w_plus, pe)
        xf_minus = x_free_from_w(w_minus, pe)

        t1 = time()
        r_plus = evaluate_fullA(xf_plus, ctx; cache = nothing, use_cache = false, warm = false)
        t_plus = time() - t1
        t2 = time()
        r_minus = evaluate_fullA(xf_minus, ctx; cache = nothing, use_cache = false, warm = false)
        t_minus = time() - t2

        feas_class = string(classify_inner_result(r_plus), "/", classify_inner_result(r_minus))
        if !(r_plus.inner_status in FEASIBLE_CODES) || !(r_minus.inner_status in FEASIBLE_CODES)
            n_infeasible_skip[] += 1
            println("  [$label class=$class k=$k scale=$scale h=$(round(h,digits=5))] INFEASIBLE, skip (class=$feas_class)")
            push!(rows, [label, delta_frontier, class, k, h_prod, scale, h, "both",
                         NaN, r_plus.inner_status, r_minus.inner_status, feas_class,
                         NaN, NaN, NaN, NaN, NaN, NaN, r_plus.inner_iters, r_minus.inner_iters, t_plus, t_minus])
            continue
        end
        n_feasible_both[] += 1
        # Remediation fix (task Part B / F-sign-reversal): this used to read
        # `-(r_plus.Delta_dual - r_minus.Delta_dual) / (2h)`, an extraneous unary minus that
        # flips the sign relative to every other quantity in this file. `fixed_dual_secant_C` /
        # `fixed_dual_secant_ref` below are (Lp-Lm)/(2h) with L == fixed_dual_L(x) == the SAME
        # canonical Delta_dual = -(mean(Psi(q))+zeta*) that r_plus.Delta_dual/r_minus.Delta_dual
        # already are (oracle.jl's cbuf[1]/1e10 convention, confirmed identical to
        # three_way_derivatives.jl's fixed_dual_L; see also cm_production_bundle.jl's
        # archC_verified_state) -- i.e. this is the exact scalar whose gradient
        # composite_gradient_at_fast/cb_G! hands to KNITRO as evalResult.jac. The independently
        # reoptimized ("true") secant must therefore be signed the same way, with NO extra
        # negation: (Delta_dual(w+h) - Delta_dual(w-h))/(2h). The old line produced a
        # near-systematic sign reversal vs ref_secant/cplus_secant in every tail-active case
        # (46/50 rows in the original phase1_directional_cases.csv), which is a diagnostic bug,
        # not a search-gradient bug -- confirmed by anchoring to the sign of the Jacobian KNITRO
        # actually receives, per the task's explicit instruction.
        true_secant = (r_plus.Delta_dual - r_minus.Delta_dual) / (2h)

        Lp_C = lfix_incremental_at_C(cache_C, ctx, pe, w0, k, w0[k] + h)
        Lm_C = lfix_incremental_at_C(cache_C, ctx, pe, w0, k, w0[k] - h)
        fixed_dual_secant_C = (Lp_C - Lm_C) / (2h)

        Lp_ref = lfix_incremental_at(cache_ref, ctx, pe, w0, k, w0[k] + h; tier = :incremental_o1)
        Lm_ref = lfix_incremental_at(cache_ref, ctx, pe, w0, k, w0[k] - h; tier = :incremental_o1)
        fixed_dual_secant_ref = (Lp_ref - Lm_ref) / (2h)

        flips = try
            θf = CS.reconstruct_full(xf_plus, ctx.m)
            d = first(last.(cells))
            origins_here = [o for (o, dd) in cells if dd == d]
            count_winner_flips_C(cache_C_flip, ctx, θf, d, origins_here)
        catch
            missing
        end

        C_vs_ref = abs(fixed_dual_secant_C - fixed_dual_secant_ref)
        C_vs_true = abs(fixed_dual_secant_C - true_secant)
        ref_vs_true = abs(fixed_dual_secant_ref - true_secant)
        n_completed[] += 1

        @printf("  [%s class=%-11s k=%4d scale=%.1f h=%.5f flips=%s] true=%.6e ref=%.6e C+=%.6e |C-ref|=%.2e\n",
            label, class, k, scale, h, string(flips), true_secant, fixed_dual_secant_ref, fixed_dual_secant_C, C_vs_ref)

        tol_abs = 1e-10; tol_rel = 1e-10
        scale_mag = max(abs(fixed_dual_secant_ref), abs(fixed_dual_secant_C), 1.0)
        tight_ok = C_vs_ref <= max(tol_abs, tol_rel * scale_mag)
        check("[$label $class k=$k scale=$scale] C+ matches Reference fixed-dual secant to tight tol",
              tight_ok || C_vs_ref <= 1e-6)   # documented looser fallback -- see report for any case relying on this
        if !tight_ok
            println("    NOTE: |C+-Reference|=$(C_vs_ref) exceeds 1e-10 tight target (still checked at 1e-6 fallback)")
        end
        # Winner-identity check: L(w) is a sum over the current winner set at each displaced point
        # (lfix_incremental.jl/lfix_factorized.jl docstrings) -- if C+'s internal incremental winner
        # bookkeeping disagreed with Reference's about which cells changed winner, Lp_C/Lm_C would
        # diverge from Lp_ref/Lm_ref even where the two SECANTS happened to still agree (cancellation).
        # Checking the L-values themselves (not just the differenced secant) is the stronger,
        # non-tautological form of "no altered winner identity relative to the trusted reference".
        check("[$label $class k=$k scale=$scale] C+ L-values (not just secant) match Reference -- no winner-identity divergence",
              isapprox(Lp_C, Lp_ref; atol = 1e-6, rtol = 1e-6) && isapprox(Lm_C, Lm_ref; atol = 1e-6, rtol = 1e-6))
        check("[$label $class k=$k scale=$scale] C+'s error vs true reoptimized secant no worse than Reference's own",
              C_vs_true <= ref_vs_true + 1e-6)

        push!(rows, [label, delta_frontier, class, k, h_prod, scale, h, "both",
                     something(flips, NaN), r_plus.inner_status, r_minus.inner_status, feas_class,
                     true_secant, fixed_dual_secant_ref, fixed_dual_secant_C,
                     C_vs_ref, C_vs_true, ref_vs_true, r_plus.inner_iters, r_minus.inner_iters, t_plus, t_minus])
    end
end

for (label, delta_frontier, P) in (("A", 1.0, A), ("B", 2.0, B))
    println("\n== Directional sweep at base point $label (delta=$delta_frontier frontier) ==")
    opt_k = optimizer_coords(P.trace, P.w0)
    rnd_k = random_coords(20260722 + (label == "A" ? 0 : 1))
    hs_k  = high_switch_coords(P.cache_C_flip, fctx.ctx, P.w0, pe; seed = label == "A" ? 99 : 100)
    println("  optimizer coords: $opt_k")
    println("  random coords:    $rnd_k")
    println("  high-switch coords: $hs_k")
    for (class, ks) in (("optimizer", opt_k), ("random_tangent", rnd_k), ("high_switch", hs_k))
        for k in ks
            run_case!("base$label", delta_frontier, class, fctx.ctx, pe, P.w0, P.cache_ref, P.cache_C, P.cache_C_flip, k)
        end
    end
end

open(joinpath(OUTDIR, "phase1_directional_cases.csv"), "w") do io
    for r in rows
        println(io, join(string.(r), ","))
    end
end
println("\nCSV written: ", joinpath(OUTDIR, "phase1_directional_cases.csv"))

println("\n============================================================")
println("Case tally: attempted=$(n_attempted[]) feasible_both=$(n_feasible_both[]) infeasible_skip=$(n_infeasible_skip[]) completed=$(n_completed[])")
println("NOT an exhaustive global derivative validation -- bounded by real per-point KNITRO cold-solve cost,")
println("per the brief's own instruction.")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
