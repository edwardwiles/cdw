# ============================================================================
# Continuation 9, Phase 6: production-scale A-block L_fix derivative validation
# at real D=20/W=80,000. Direct follow-up to the OPEN finding in
# docs/fullA_nested_w_continuation_c8.md sec 4 (D=4, synthetic data): the fast
# `lfix_composite` gradient's A-block-ONLY sub-vector shows real, unresolved
# sign disagreement (cosine 0.17-0.59, sign agreement 73-93%) against a slow
# finite-difference reference, even though the FULL vector (dominated by
# gamma'_focal) agrees almost perfectly (cosine >=0.9998). This has never been
# tested at D=20 or on real data -- that is this task.
#
# Methodology: "gravity-tangent A-only direction" = a random unit vector in
# the pivot-reduced z_free (A-block) coordinates, EXACT gravity-feasible by
# construction of pivot_expand (not a first-order approximation -- see
# gravity_elimination.jl / the W80k doc's own Point 2 finding, reused here
# rather than re-derived). 20 such directions (deterministic seed), reused
# across all 3 test points. For each direction/point/bandwidth, compare:
#   - the fast gradient's own PREDICTED directional derivative
#     dot(g_fast[A-block], v) -- read off from ONE full 400-coord
#     composite_gradient_at_fast call per point (not 20 separate partial
#     calls -- g_fast[2:end] already contains every A-block coordinate's
#     partial derivative, so a linear combination along any direction is
#     free once g_fast is in hand).
#   - the "slow, trusted" optimized-value central/left/right secants of
#     Delta_dual along that direction, via evaluate_fullA (fully re-solved,
#     the same "trusted reference" oracle used throughout this investigation,
#     matching docs/fullA_D20_W80k_microbenchmark.md sec 3F's own directional-
#     secant methodology, extended here from 5 directions with no fast-
#     gradient comparison to 20 directions WITH one).
#   - winner-switch counts / CC-weighted switch mass for the +-h perturbed
#     points, via the EXISTING winner_switching.jl::switch_stats +
#     winners.jl::compute_winners machinery (reused, not reimplemented --
#     count_winner_flips itself only applies to single/double-origin
#     PER-COORDINATE perturbations, so switch_stats+compute_winners is the
#     correct existing sibling tool for an arbitrary multi-coordinate
#     direction; documented here explicitly per this investigation's
#     "attribute reused code clearly" convention).
#
# Fast-gradient production default: per docs/fullA_D20_bandwidth_optimization_report.md
# sec 4's explicit recommendation, h_mode=:cached wrapped in BandwidthCachePolicy
# (NOT h_mode=:quantile, which is numerically fine but delivers little wall-
# clock win; NOT h_mode=:fixed, which has no per-coordinate adaptivity). The
# underlying Dict-write thread-safety bug that report flagged (sec 3F) was
# already fixed by the coordinating session (commit a6ed25e, ReentrantLock),
# confirmed present in this worktree below before trusting threaded=true. A
# FRESH BandwidthCachePolicy per point (all-miss on first use) is used, since
# each of our 3 points is evaluated as a one-shot "first outer iterate" call,
# exactly the case a real production driver hits when the outer loop lands on
# a genuinely new point (as opposed to nearby repeated calls, which is what
# the cache's warm-hit path is FOR and is not this task's scope).
#
# Cost management: per point, 1 base Delta(w0) solve (reused for all one-
# sided secants) + 1 steepest-descent-direction sanity probe (2 h x 2 signs
# = 4 solves) + 20 directions x 2 bandwidths x 2 signs = 80 solves. Total
# ~85 evaluate_fullA calls/point x 3 points = ~255 solves, each ~1.8-4s warm
# at this scale (docs/fullA_D20_W80k_microbenchmark.md sec 3A/3B) -- budgeted
# at roughly 15-20 minutes of pure compute, consistent with the task's own
# 10-16 minute estimate plus compute_winners/switch_stats overhead.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using Statistics, Printf, Dates, Random, LinearAlgebra

@assert isdefined(Main, :ReentrantLock)
# thread-safety-fix presence check (commit a6ed25e): grep the source directly
# rather than trust a comment, matching this investigation's "verify before
# causal claims" discipline.
let src = read(joinpath(@__DIR__, "composite_gradient_fast.jl"), String)
    @assert occursin("bandwidth_cache_lock", src) && occursin("ReentrantLock()", src) "composite_gradient_fast.jl: expected ReentrantLock guard around the h_mode=:cached Dict (commit a6ed25e) not found -- ABORT, threaded=true would be unsafe."
end
let src = read(joinpath(@__DIR__, "context_real_d20.jl"), String)
    @assert occursin("needs_outer_moment_jacobian::Bool = false", src) "context_real_d20.jl: needs_outer_moment_jacobian default is not false -- ABORT before building a D=20/W=80000 context (109GB jac_h risk, see docs/fullA_D20_W80k_microbenchmark.md sec 0)."
end
println("Safety checks passed: ReentrantLock guard present, needs_outer_moment_jacobian defaults to false.")
flush(stdout)

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase6_gradval_d20_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase6_gradval_d20.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end

const FEASIBLE_CODES = (0, -100, -101, -103)

# ============================================================================
# STEP 0: memory sanity check BEFORE scaling to the full run (mandatory per
# this investigation's standing safety protocol).
# ============================================================================
logprint("\n", "="^90); logprint("STEP 0: context build + memory sanity check"); logprint("="^90)
t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
vmhwm_setup = vmhwm_kb()
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s   VmHWM = ", vmhwm_setup,
         " KB = ", round(vmhwm_setup / 1e6, digits = 2), " GB")
if vmhwm_setup > 5_000_000
    logprint("MEMCHECK FAILED -- VmHWM over 5GB after setup alone. ABORTING before any further compute.")
    close(LOGIO)
    error("memcheck failed")
end
logprint("MEMCHECK PASS (< 5GB). Proceeding.")

D = ctx.D; D2 = D^2
n_free = 1 + D2
logprint("D=", D, " D2=", D2, " n_free=", n_free)

pe = build_pivot_elimination(ctx)
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
logprint("gamma'_focal (natural theta) = ", gp0, "  bounds=[", ctx.bounds.γp_lo, ", ", ctx.bounds.γp_hi, "]")

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
zfree_nat = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)

# ============================================================================
# STEP 1: the 3 outer points (calibration / upper / lower), matching the W80k
# doc's own Point 1 / Point 3 / Point 4 exactly (same offsets, same "A held
# at natural theta" construction).
# ============================================================================
logprint("\n", "="^90); logprint("STEP 1: the 3 outer points"); logprint("="^90)

r1 = evaluate_fullA(xf_nat, ctx; warm = false)
logprint("Point 1 (calibration): inner_status=", r1.inner_status, " Delta_dual=", r1.Delta_dual)
@assert r1.inner_status in FEASIBLE_CODES "calibration point infeasible -- contradicts every prior benchmark in this session"

gp_up = clamp(gp0 * 1.01, ctx.bounds.γp_lo, ctx.bounds.γp_hi)
xf3 = vcat(gp_up, xf_nat[2:end])
r3 = evaluate_fullA(xf3, ctx; warm = true)
logprint("Point 3 (upper, gp0*1.01=", gp_up, "): inner_status=", r3.inner_status, " Delta_dual=", r3.Delta_dual)
@assert r3.inner_status in FEASIBLE_CODES "upper-branch point infeasible at offset 0.01 -- W80k doc found this feasible, re-check"

gp_dn = clamp(gp0 * 0.99, ctx.bounds.γp_lo, ctx.bounds.γp_hi)
xf4 = vcat(gp_dn, xf_nat[2:end])
r4 = evaluate_fullA(xf4, ctx; warm = true)
logprint("Point 4 (lower, gp0*0.99=", gp_dn, "): inner_status=", r4.inner_status, " Delta_dual=", r4.Delta_dual)
@assert r4.inner_status in FEASIBLE_CODES "lower-branch point infeasible at offset 0.01 -- W80k doc found this feasible, re-check"

points = [
    (label = "calibration", xf = xf_nat, gp = gp0),
    (label = "upper_gp1.01", xf = xf3, gp = gp_up),
    (label = "lower_gp0.99", xf = xf4, gp = gp_dn),
]

# ============================================================================
# STEP 2: 20 deterministic random gravity-tangent A-only directions (fixed
# seed, reused identically across all 3 points for direct comparability).
# ============================================================================
const N_DIRS = 20
const H_LIST = [0.02, 0.005]
Random.seed!(20260719)
dirs = Vector{Vector{Float64}}()
for i in 1:N_DIRS
    v = randn(D2 - 1)
    v ./= norm(v)
    push!(dirs, v)
end
logprint("\nBuilt ", N_DIRS, " deterministic unit directions (seed=20260719) in the ", D2 - 1, "-dim pivot-reduced A-block.")

cossim(a, b) = dot(a, b) / max(norm(a) * norm(b), 1e-300)

DIR_CSV_COLS = ["point", "dir_idx", "h", "pred_dot", "delta0", "delta_plus", "delta_minus",
                "central_secant", "right_secant", "left_secant",
                "sign_match_central", "sign_match_right", "sign_match_left",
                "n_switches_plus", "cc_mass_plus", "n_switches_minus", "cc_mass_minus",
                "wall_plus_s", "wall_minus_s"]
DIR_CSV_PATH = joinpath(OUTDIR, "direction_secants.csv")
open(DIR_CSV_PATH, "w") do io
    println(io, join(DIR_CSV_COLS, ","))
end

STEEP_CSV_COLS = ["point", "h", "pred_slope", "delta0", "delta_plus", "delta_minus",
                  "central_secant", "right_secant", "left_secant", "actual_decreases"]
STEEP_CSV_PATH = joinpath(OUTDIR, "steepest_descent_probe.csv")
open(STEEP_CSV_PATH, "w") do io
    println(io, join(STEEP_CSV_COLS, ","))
end

SUMMARY_ROWS = NamedTuple[]

signmatch(a, b) = (a == 0.0 || b == 0.0) ? missing : (sign(a) == sign(b))

# ============================================================================
# STEP 3: main loop over the 3 points
# ============================================================================
for pt in points
    logprint("\n", "="^90); logprint("POINT: ", pt.label); logprint("="^90)
    w0 = vcat(pt.gp, pivot_reduce(log.(reshape(pt.xf[2:end], D, D)), pe))
    Delta_of_w(w) = evaluate_fullA(x_free_from_w(w), ctx; warm = true).Delta_dual

    local t0 = time()
    Δ0 = Delta_of_w(w0)
    logprint("  Delta_dual(w0) = ", Δ0, "  wall=", round(time() - t0, digits = 2), "s")

    θ_full0 = CS.reconstruct_full(pt.xf, ctx.m)
    winner0, _, _ = compute_winners(θ_full0, ctx)

    local t0 = time()
    base0 = solve_base_state(pt.xf, ctx)
    logprint("  solve_base_state wall=", round(time() - t0, digits = 2), "s  inner_status=", base0.inner_status)

    # ---- fast gradient: production default h_mode=:cached + fresh BandwidthCachePolicy ----
    policy = BandwidthCachePolicy()
    maybe_invalidate!(policy, w0)
    local t0 = time()
    g_fast, meta_fast = composite_gradient_at_fast(pt.xf, ctx, pe; base = base0, threaded = true,
                                                     h_mode = :cached, bandwidth_cache = policy.cache,
                                                     multi_method = :top3)
    wall_fast = time() - t0
    record_hits!(policy, meta_fast.cache_hits)
    logprint("  fast gradient (h_mode=:cached, threaded=true): wall=", round(wall_fast, digits = 2), "s",
             "  cache_hits=", count(meta_fast.cache_hits), "/", D2 - 1,
             "  gamma_component=", g_fast[1])

    g_Ablock = g_fast[2:end]
    @assert length(g_Ablock) == D2 - 1 == length(dirs[1])
    logprint("  ||g_Ablock|| = ", norm(g_Ablock), "  max|g_Ablock| = ", maximum(abs.(g_Ablock)))

    # ---- steepest-descent-direction sanity probe: does moving along -g_Ablock/||.|| actually
    #      decrease (increase, per find_smallest) Delta_dual, matching the sign the gradient predicts? ----
    v_sd = -g_Ablock ./ norm(g_Ablock)
    for h in H_LIST
        wp = copy(w0); wp[2:end] .+= h .* v_sd
        wm = copy(w0); wm[2:end] .-= h .* v_sd
        Δp = Delta_of_w(wp); Δm = Delta_of_w(wm)
        pred_slope = dot(g_Ablock, v_sd)   # == -norm(g_Ablock) by construction; predicted: Delta decreases along +v_sd
        central = (Δp - Δm) / (2h)
        right = (Δp - Δ0) / h
        left = (Δ0 - Δm) / h
        actual_decreases = Δp < Δ0
        logprint(@sprintf("  [steepest,h=%.3f] pred_slope=%.6e central=%.6e right=%.6e left=%.6e Delta0=%.6f Delta+=%.6f actual_decreases=%s",
                           h, pred_slope, central, right, left, Δ0, Δp, actual_decreases))
        open(STEEP_CSV_PATH, "a") do io
            println(io, join([pt.label, h, pred_slope, Δ0, Δp, Δm, central, right, left, actual_decreases], ","))
        end
    end

    # ---- 20 random directions x 2 bandwidths ----
    preds = zeros(N_DIRS); centrals = zeros(N_DIRS, length(H_LIST))
    for (di, v) in enumerate(dirs)
        pred = dot(g_Ablock, v)
        preds[di] = pred
        for (hi, h) in enumerate(H_LIST)
            wp = copy(w0); wp[2:end] .+= h .* v
            wm = copy(w0); wm[2:end] .-= h .* v
            local t0 = time(); Δp = Delta_of_w(wp); wall_p = time() - t0
            local t0 = time(); Δm = Delta_of_w(wm); wall_m = time() - t0
            central = (Δp - Δm) / (2h)
            right = (Δp - Δ0) / h
            left = (Δ0 - Δm) / h
            centrals[di, hi] = central

            θ_full_p = CS.reconstruct_full(x_free_from_w(wp), ctx.m)
            θ_full_m = CS.reconstruct_full(x_free_from_w(wm), ctx.m)
            winner_p, _, _ = compute_winners(θ_full_p, ctx)
            winner_m, _, _ = compute_winners(θ_full_m, ctx)
            stats_p = switch_stats(winner0, winner_p; m_weights = base0.m_star)
            stats_m = switch_stats(winner0, winner_m; m_weights = base0.m_star)

            sm_c = signmatch(pred, central); sm_r = signmatch(pred, right); sm_l = signmatch(pred, left)

            open(DIR_CSV_PATH, "a") do io
                println(io, join([pt.label, di, h, pred, Δ0, Δp, Δm, central, right, left,
                                   sm_c, sm_r, sm_l,
                                   stats_p.n_switches, stats_p.cc_weighted_mass,
                                   stats_m.n_switches, stats_m.cc_weighted_mass,
                                   wall_p, wall_m], ","))
            end
            @printf("    dir %2d h=%.3f pred=%+.4e central=%+.4e sign_match=%s  switches(+/-)=%d/%d\n",
                    di, h, pred, central, sm_c, stats_p.n_switches, stats_m.n_switches)
            flush(stdout)
        end
    end

    for (hi, h) in enumerate(H_LIST)
        actual = centrals[:, hi]
        cs = cossim(preds, actual)
        n_valid = count(x -> !ismissing(x), signmatch.(preds, actual))
        n_match = count(x -> x === true, signmatch.(preds, actual))
        sign_frac = n_valid > 0 ? n_match / n_valid : NaN
        mean_abs_err = mean(abs.(preds .- actual))
        median_abs_err = median(abs.(preds .- actual))
        logprint(@sprintf("  SUMMARY point=%s h=%.3f  direction-sample cosine(pred,central)=%.4f  sign_agree=%d/%d (%.1f%%)  mean_abs_err=%.3e  median_abs_err=%.3e",
                           pt.label, h, cs, n_match, n_valid, 100sign_frac, mean_abs_err, median_abs_err))
        push!(SUMMARY_ROWS, (point = pt.label, h = h, n_dirs = N_DIRS, cosine_pred_vs_central = cs,
                              sign_agree_frac = sign_frac, n_sign_match = n_match, n_sign_valid = n_valid,
                              mean_abs_err = mean_abs_err, median_abs_err = median_abs_err,
                              norm_g_Ablock = norm(g_Ablock), wall_fast_s = wall_fast,
                              cache_hits = count(meta_fast.cache_hits), Delta0 = Δ0))
    end
    flush(stdout)
end

SUMMARY_CSV_PATH = joinpath(OUTDIR, "summary.csv")
open(SUMMARY_CSV_PATH, "w") do io
    cols = collect(string.(keys(SUMMARY_ROWS[1])))
    println(io, join(cols, ","))
    for r in SUMMARY_ROWS
        println(io, join([getfield(r, Symbol(c)) for c in cols], ","))
    end
end

logprint("\n", "="^90); logprint("ALL SUMMARY ROWS"); logprint("="^90)
for r in SUMMARY_ROWS
    logprint(r)
end

logprint("\nWrote: ", DIR_CSV_PATH)
logprint("Wrote: ", STEEP_CSV_PATH)
logprint("Wrote: ", SUMMARY_CSV_PATH)
logprint("Final VmHWM = ", vmhwm_kb(), " KB = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("DONE at ", now())
close(LOGIO)
