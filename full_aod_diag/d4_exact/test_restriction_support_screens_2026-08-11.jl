# Do cheap SUPPORT screens catch the points that fail? (2026-08-11)
#
# The 8h OZC-CROSS run attempted 246 inner solves for 100 usable evaluations: 146 were rejected
# (nStatus outside the accepted set) AFTER paying for a full solve. The existing pre-solve screen
# cannot catch these -- `pairwise_certificate` (infeasibility_screen.jl) tests
# `M[o,k] = max_s(B[s,o]-B[s,k])` against the A_od block ONLY, i.e. "can origin o ever be the
# argmin into d"; it never sees nu, delta, or the restriction block. Its 0% hit rate near the
# delta=1 boundary is therefore EXPECTED, not a defect: the A_od configuration is fine there, and
# what fails is the MOMENT problem.
#
# Two candidate screens, both hard SUPPORT certificates (never heuristics -- a false rejection would
# silently bias a bound whose validity rests on every incumbent being genuinely feasible):
#
#   A (marginal). The LFD weights are a normalised reweighting of the draws, so each restriction
#     target must lie in the convex hull of the corresponding feature over the draws:
#         nu_{o,k} in [min_s z_{s,o}^k, max_s z_{s,o}^k],  z = U^(-mu).
#     For the ORIGIN-specific layout this is per-(o,k) and needs no intersection across origins.
#     Campaign-constant (z does not depend on theta). NOTE the production box is
#     `meanzc_default_nu_bounds` = (log(lo/4), log(hi*4)) -- deliberately FOUR TIMES wider than the
#     exact interval on each side, so KNITRO may legally propose provably-infeasible nu.
#
#   B (cross-moment). The cross families additionally impose
#         E[z_o^{k1} z_p^{k2}] = nu_{o,k1} * nu_{p,k2},
#     so that PRODUCT must lie in [min_s, max_s] of the pairwise product over draws. This is NOT
#     implied by A, it is exactly the constraint the cross grid adds over the diagonal family, and
#     it is campaign-constant too.
#
# THE TEST. Screens are useful only if they never fire on a point that would have SUCCEEDED. So this
# records every ATTEMPTED evaluation and its true outcome -- via RESTRICTION_SUPPORT_DIAG, because
# the driver's own trace contains only the successes (reject_point throws before `n_eval[] += 1`) --
# and reports a confusion matrix. FALSE POSITIVES MUST BE ZERO.
#
# Usage: julia --project=. -t 10 .../test_restriction_support_screens_2026-08-11.jl [budget_s]
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Serialization, Statistics
lp(xs...) = (println(xs...); flush(stdout))

const BUDGET = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1200.0
const KK = 3; const KSTAR = 2; const W = 100_000; const DELTA = 1.0
const CKPT = joinpath(_D4E, "..", "..", "results", "ozc_cross_production_smoke_2026-08-09",
                      "K3_W100000_sobol_randomized_upper8h_2026-08-10", "ozc_cross_K3_W100000_latest.jls")

lp("="^104)
lp("RESTRICTION SUPPORT SCREENS: would A / B have caught the points that actually failed?")
lp("="^104)

ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D; mu = ctx.μHat
layout = OriginByPowerCrossLayout(D, KK, KK)
prs = packed_pair_index(D)
levels = cross_pair_level_index(KK)
lp("D=", D, "  mu=", mu, "  n_eta=", n_eta(layout), "  pairs=", length(prs), "  cross levels=", length(levels))

lp("\nprecomputing support ranges (one-time, campaign-constant)...")
t_pre = @elapsed begin
    Z = [frechet_power_feature(ctx.U, k, mu) for k in 1:KK]          # Z[k][s,o] = z_{s,o}^k
    Alo = [minimum(@view Z[k][:, o]) for o in 1:D, k in 1:KK]
    Ahi = [maximum(@view Z[k][:, o]) for o in 1:D, k in 1:KK]
    Blo = Array{Float64}(undef, length(prs), length(levels))
    Bhi = Array{Float64}(undef, length(prs), length(levels))
    for (klin, kk) in enumerate(levels)
        k1, k2 = kk
        Zk1 = Z[k1]; Zk2 = Z[k2]
        Threads.@threads for ip in 1:length(prs)
            o, p = prs[ip]
            lo = Inf; hi = -Inf
            @inbounds for s in 1:W
                v = Zk1[s, o] * Zk2[s, p]
                v < lo && (lo = v); v > hi && (hi = v)
            end
            Blo[ip, klin] = lo; Bhi[ip, klin] = hi
        end
    end
end
@printf("  done in %.1f s  (paid ONCE per campaign, not per point)\n", t_pre)
@printf("  screen A: %d per-(o,k) intervals;  screen B: %d per-(pair,level) intervals\n",
        D * KK, length(prs) * length(levels))

function screen_A_fires(nu)
    @inbounds for k in 1:KK, o in 1:D
        v = nu[target_index(layout, o, k)]
        (v < Alo[o, k] || v > Ahi[o, k]) && return true
    end
    return false
end
function screen_B_fires(nu)
    @inbounds for (klin, kk) in enumerate(levels)
        k1, k2 = kk
        for (ip, op) in enumerate(prs)
            o, p = op
            v = nu[target_index(layout, o, k1)] * nu[target_index(layout, p, k2)]
            (v < Blo[ip, klin] || v > Bhi[ip, klin]) && return true
        end
    end
    return false
end
nu_probe = ones(n_eta(layout))
screen_A_fires(nu_probe); screen_B_fires(nu_probe)
t_A = @elapsed(for _ in 1:100; screen_A_fires(nu_probe); end)
t_B = @elapsed(for _ in 1:100; screen_B_fires(nu_probe); end)
@printf("  per-point screen cost: A %.2f us   B %.2f us   (a full inner solve is SECONDS)\n",
        1e6 * t_A / 100, 1e6 * t_B / 100)

lp("\nproduction eta box vs the exact support interval (this is what lets bad nu be proposed):")
for k in 1:KK
    lo, hi = nu_feasible_interval(ctx.U, k; μ = mu)
    @printf("  k=%d  exact [%.6g, %.6g]   production box [%.6g, %.6g]   (widened x4 each side)\n",
            k, lo, hi, lo / 4, hi * 4)
end

ck = load_cm_checkpoint_v10(CKPT); bf = ck.best_feasible
w0 = collect(Float64, bf.w)
lp("\nstarting from the 8h run's own incumbent: gp=", bf.gp, "  Delta=", bf.Delta)
lp("  (17 of 21 attempts failed there in the earlier profile -- high signal per unit wall-clock)")
OUT = joinpath(_D4E, "..", "..", "results", "support_screen_study_2026-08-11")
rm(OUT; force = true, recursive = true); mkpath(OUT)
RESTRICTION_SUPPORT_DIAG[] = Any[]
t0 = time()
result = run_originzc_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :sobol_randomized,
    draw_seed = 20260719, distribution_restriction = :origin_specific_moments_zero_covariance,
    K_mean = KK, K_pair = KK, power_target_layout = :origin_by_power_cross,
    originzc_profiled_level = KSTAR, inner_lower_limit = -10.0,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
    ckpt_dir = OUT, run_id = "support_study", label = "support_study",
    checkpoint_interval_s = 1e9, maxtime_real = BUDGET, verbose = true)
wall = time() - t0
rec = RESTRICTION_SUPPORT_DIAG[]; RESTRICTION_SUPPORT_DIAG[] = nothing
# Persist BEFORE analysing. The first version of this script lost a 23-minute driver run to a
# Julia soft-scope bug in the tally loop below: the recorded attempts existed only in memory and
# died with the process. Expensive data gets written out before anything can throw.
serialize(joinpath(OUT, "attempts.jls"), rec)
lp("  recorded attempts serialized -> ", joinpath(OUT, "attempts.jls"))

lp("\n", "="^104)
@printf("RECORDED %d attempted inner solves in %.0f s   (driver n_eval=%d => %d rejected)\n",
        length(rec), wall, result.n_eval, length(rec) - result.n_eval)
lp("="^104)

"Tallied inside a function: at top level Julia's soft scope makes `nA += 1` create a NEW local, so the accumulators silently reset (and then error). This is the bug that cost the first run."
function tally(rec, screen_A_fires, screen_B_fires)
    nA = nB = fpA = fpB = tpA = tpB = nfail = 0
    for r in rec
        a = screen_A_fires(r.nu); b = screen_B_fires(r.nu)
        a && (nA += 1); b && (nB += 1)
        r.ok || (nfail += 1)
        if r.ok
            a && (fpA += 1); b && (fpB += 1)
        else
            a && (tpA += 1); b && (tpB += 1)
        end
    end
    return (nA = nA, nB = nB, fpA = fpA, fpB = fpB, tpA = tpA, tpB = tpB, nfail = nfail)
end
t = tally(rec, screen_A_fires, screen_B_fires)
nok = length(rec) - t.nfail
@printf("  attempts=%d   succeeded=%d   failed=%d\n", length(rec), nok, t.nfail)
@printf("  %-28s %8s %14s %16s\n", "screen", "fired", "TRUE POS", "FALSE POS")
@printf("  %-28s %8d %8d/%-5d %10d   <-- must be 0\n", "A (marginal support)", t.nA, t.tpA, t.nfail, t.fpA)
@printf("  %-28s %8d %8d/%-5d %10d   <-- must be 0\n", "B (cross-moment support)", t.nB, t.tpB, t.nfail, t.fpB)
either_tp = count(r -> !r.ok && (screen_A_fires(r.nu) || screen_B_fires(r.nu)), rec)
either_fp = count(r ->  r.ok && (screen_A_fires(r.nu) || screen_B_fires(r.nu)), rec)
@printf("  %-28s %8s %8d/%-5d %10d\n", "A or B", "", either_tp, t.nfail, either_fp)
lp()
t.nfail > 0 && @printf("  => would avoid %.1f%% of failed solves, at ~%.0f us per point.\n",
                       100 * either_tp / t.nfail, 1e6 * (t_A + t_B) / 100)
(t.fpA == 0 && t.fpB == 0) ? lp("  FALSE POSITIVES: none.") :
                             lp("  *** FALSE POSITIVES PRESENT -- indicates a WIRING bug in the screen, not a flaw in the certificate.")
flush(stdout)
