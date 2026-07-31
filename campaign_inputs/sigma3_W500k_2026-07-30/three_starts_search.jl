# ============================================================================
# three_starts_search.jl (sigma3/W500k five-family production campaign, 2026-07-30)
#
# Adapted from full_aod_diag/d4_exact/common_five_starts_search.jl (five-family bounds
# campaign, 2026-07-28) -- same search machinery, retargeted at this campaign's
# configuration: REAL_DATA_DIR=immutable snapshot, W=500,000, sigma=3.0,
# exclude_diagonal_gravity=true, meanzc/originzc K_mean=K_pair=2, start_seed=20260730
# (per campaign brief), N_STARTS_POOL accepted candidates searched, then the 2
# best-separated (max-min pairwise scaled distance) selected as Starts 2-3 alongside
# the calibration point (Start 1) -- see SELECT_BY_DISTANCE step near the bottom, which
# is genuinely NEW relative to the source script (that script just took the first
# N_STARTS-1 valid perturbations found; this campaign's brief explicitly requires
# maximizing minimum pairwise distance, not first-found).
#
# Finds N_STARTS_POOL economic starting points -- the genuine calibrated point plus
# deterministic seeded perturbations of it -- that are SIMULTANEOUSLY valid for
# ALL FIVE optimization families used by the campaign:
#
#   1. unrestricted        (evaluate_fullA_screened_ranged, fast_range_screen.jl)
#   2. flexible_cm         (cm_production_value_verified_screened)
#   3. common_frechet      (cm_frechet_production_value_verified_screened)
#   4. cm_meanzc           (cm_meanzc_production_value_verified_screened)
#   5. origin_zc           (cm_originzc_production_value_verified_screened)
#
# The (A_od, gp) portion of an accepted start is IDENTICAL across all five
# families by construction: one w = [gp; a_nonpivot] vector in the production
# transformed-A coordinate (:powered_aspace, cm_aspace_coordinate.jl -- the
# production default of run_cm_upper_checkpointed/run_originzc_upper_checkpointed
# since the five-family finish task) is decoded ONCE per candidate into the
# natural free vector `xf = x_free_from_w(vcat(gp, cm_z_from_a(a,...)), pe)`,
# and that single `xf` is handed to all five families.
#
# Family-specific restriction coordinates (meanzc nu, origin-ZC nu) use their
# standard calibrated/implied initialization, held FIXED across all starts --
# only the shared economic (A, gp) block is perturbed, per the campaign spec.
#
# This is deliberately NOT an outer optimization: each family check is exactly
# the same single-point "screens + one inner Delta* solve + verification" call
# the real KNITRO driver's own callback makes, so a start accepted here is one
# the production driver can actually start from.
#
# Rejection policy: a candidate that fails ANY family is rejected for ALL five
# (the campaign needs a common start, not a per-family one).
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t 8 \
#       common_five_starts_search.jl [W] [max_candidates] [max_per_radius] [outdir]
# ============================================================================

const _D4E = normpath(joinpath(@__DIR__, "..", "..", "full_aod_diag", "d4_exact"))
ENV["REAL_DATA_DIR"] = joinpath(@__DIR__, "data_snapshot")

# Include list = the five smoke_delta1_*.jl scripts' own restricted-family list
# (verbatim, same order) MINUS cm_checkpoint.jl / cm_originzc_checkpoint.jl (this
# script never runs a checkpointed outer driver, and a concurrent task owns those
# two files), PLUS the unrestricted family's screening/gradient files.
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          # threaded_cross_hessian.jl/zc_gram_blas_candidates.jl/hcz_drawchunk_candidate_2026-07-29.jl
          # (2026-07-30 fix): missing from the 2026-07-28-era source script's own include list --
          # cm_hessian_architectures.jl (already included above) references HCZ_PREP_BACKEND_DEFAULT
          # as a default kwarg value, which is only ever resolved at CALL time (Julia re-evaluates
          # default-argument expressions per call), so this omission didn't fail until the first
          # real CM-family inner solve actually ran -- confirmed live 2026-07-30 (UndefVarError).
          "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          # 2026-07-28 fix: the *_lookup_production.jl files are NOT self-included by cm_screen_
          # bridge.jl/cm_frechet_cplus.jl/etc. even though the *_verified_screened wrappers call
          # into them at runtime (e.g. cm_frechet_production_value_verified_screened ->
          # inner_loop_internal_cmfrechetlookup_production) -- this is what the earlier real
          # UndefVarError at W=100,000 was. cm_meanzc_lookup_production.jl/cm_originzc_lookup_
          # production.jl self-include their own _lookup_kernels.jl; cm_frechet's does not, hence
          # cm_frechet_lookup_kernels.jl is listed explicitly too.
          "cm_lookup_production.jl", "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "cm_meanzc_lookup_production.jl", "cm_originzc_lookup_production.jl",
          # ---- unrestricted-family additions (not in the CM smoke list) ----
          "lfix_buffer_reuse.jl",          # -> composite_gradient_at_fast_buffered (unrestricted gradient)
          "bandwidth_cache_policy.jl",
          "fast_range_screen.jl"]          # -> build_ranged_screen_context / evaluate_fullA_screened_ranged
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics, Random, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

# ---------------------------------------------------------------------------
# Campaign configuration (task spec)
# ---------------------------------------------------------------------------
const W               = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500_000
const MAX_CANDIDATES  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 60
const MAX_PER_RADIUS  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
const OUTDIR          = length(ARGS) >= 4 ? ARGS[4] : @__DIR__

const DELTA           = 0.01
const FIND_SMALLEST   = true
const DRAW_DESIGN     = :sobol_randomized
const DRAW_SEED       = 20260719
const START_SEED      = 20260730   # campaign brief: perturbation seed = 20260730 (distinct from the shared Sobol seed 20260719)
const DEST_SAMPLE     = :exclude_row
const EXCLUDE_DIAGONAL_GRAVITY = true
const SIGMA           = 3.0
# search for a POOL of accepted candidates beyond the calibration point, then select the 2
# best-separated (max-min pairwise scaled distance) as Starts 2-3 -- campaign brief requires
# "maximize minimum pairwise scaled distance", not "first 2 found" (see SELECT_BY_DISTANCE below).
const N_STARTS_POOL   = 6
const N_STARTS_FINAL  = 3
const RADIUS0         = 0.01
const FEAS_TOL        = 1e-6      # this codebase's own convention: `Delta <= delta + 1e-6`
                                  # (cm_outer_driver.jl:110, cm_checkpoint.jl:1078,
                                  #  cm_originzc_checkpoint.jl:725, cm_cold_verify.jl:95,
                                  #  originzc_cold_verify.jl:104). No named constant exists.
const CM_L            = 50
const MEANZC_K        = 2   # confirmed exact intended meaning, see K_MEAN_K_PAIR_AUDIT.md
const ORIGINZC_K      = 2

const STAMP = "sigma3_W500k_2026-07-30"
const JSON_PATH = joinpath(OUTDIR, "THREE_STARTS_POOL_MANIFEST_$(STAMP).json")
const CSV_PATH  = joinpath(OUTDIR, "THREE_STARTS_POOL_MANIFEST_$(STAMP).csv")
const LOG_PATH  = joinpath(OUTDIR, "THREE_STARTS_SEARCH_LOG_$(STAMP).md")
const FINAL_JSON_PATH = joinpath(OUTDIR, "start_manifest.json")   # the campaign's frozen 3-start manifest (calibration + 2 selected)

peak_rss_mb() = (for line in eachline("/proc/self/status"); startswith(line, "VmHWM:") && return parse(Float64, split(line)[2]) / 1024; end; NaN)

lp("="^100)
lp("COMMON FIVE-FAMILY STARTS SEARCH -- ", Dates.now())
lp("  W=", W, " delta=", DELTA, " find_smallest=", FIND_SMALLEST,
   " draw_design=:", DRAW_DESIGN, " draw_seed=", DRAW_SEED, " destination_sample=:", DEST_SAMPLE)
lp("  start_seed=", START_SEED, " radius0=", RADIUS0, " n_starts=", N_STARTS_POOL,
   " max_candidates=", MAX_CANDIDATES, " max_per_radius=", MAX_PER_RADIUS)
lp("  julia threads=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())
lp("  outdir=", OUTDIR)
lp("="^100)

# ---------------------------------------------------------------------------
# Shared context + calibration point (verbatim smoke_delta1_*.jl construction)
# ---------------------------------------------------------------------------
t_ctx = time()
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
                            draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
                            destination_sample = DEST_SAMPLE,
                            exclude_diagonal_gravity = EXCLUDE_DIAGONAL_GRAVITY, σHat = SIGMA)
pe = build_pivot_elimination(ctx)
const D = ctx.D
const Ddest = ctx.D_dest
lp(">> ctx built in ", round(time() - t_ctx, digits = 1), "s: D=", D, " D_dest=", Ddest,
   " W=", size(ctx.U, 1), " draw checksums uniform=", ctx.draw_meta.checksum_uniform,
   " transformed=", ctx.draw_meta.checksum_transformed)

theta0 = cm_fixed_theta(ctx)
xy0 = precompute_cm_aspace_xy(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
gp_calib = x_free_calib[1]
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe)
const w_calib = vcat(gp_calib, a_calib)     # production transformed-A outer coordinate
const NW = length(w_calib)
lp(">> calibration point: gp=", gp_calib, "  |w|=", NW,
   "  gp bounds=[", ctx.bounds.γp_lo, ", ", ctx.bounds.γp_hi, "]")

"Decode a transformed-A outer vector w=[gp; a_nonpivot] into (xf, zfree) exactly as
 cm_checkpoint.jl::xf_from_w_econ/zfree_from_w_econ do under A_coordinate_mode=:powered_aspace."
function decode_w(w)
    zfree = cm_z_from_a(w[2:end], theta0, xy0, pe)
    return x_free_from_w(vcat(w[1], zfree), pe), zfree
end

# W-independent invariant: decoding the calibration w must reproduce ctx.θ0_up's own free
# vector to machine precision (cm_a_from_z/cm_z_from_a and pivot_reduce/pivot_expand are exact
# inverses). This is a pure coordinate-algebra check -- it does NOT depend on W, the draws, or
# any solver, so it is the one correctness signal that is meaningful at any scale.
let (xf_rt, _) = decode_w(w_calib)
    rel = maximum(abs.(xf_rt .- x_free_calib) ./ max.(abs.(x_free_calib), 1e-300))
    lp(">> coordinate round-trip (w -> z -> x_free) max rel err vs ctx.θ0_up free block = ", rel)
    rel < 1e-10 || error("common_five_starts_search: calibration coordinate round-trip failed " *
                          "(max rel err = $rel) -- the transformed-A decode path is wrong, refusing to search.")
end

# ---------------------------------------------------------------------------
# Per-family production contexts (built ONCE, reused for every candidate)
# ---------------------------------------------------------------------------
const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[CM_L]

lp(">> building family contexts ...")
t0 = time(); rsc = build_ranged_screen_context(ctx)
lp("   [unrestricted] RangedScreenContext in ", round(time() - t0, digits = 1), "s envelope=",
   rsc.envelope === nothing ? "UNSUPPORTED ($(rsc.unsupported_reason))" : "supported")

t0 = time(); pcx_flexcm = with_screen_counters(build_cm_production_context(ctx, CS;
        L = CM_L, contrasts = :orthonormal, probs = PROBS_L50))
lp("   [flexible_cm] ", round(time() - t0, digits = 1), "s  rss=", round(peak_rss_mb(), digits = 0), "MB")

t0 = time(); pcx_frechet = with_screen_counters(build_cm_frechet_production_context(ctx, CS;
        L = CM_L, contrasts = :orthonormal, probs = PROBS_L50, cm_hessian_backend = :structured))
lp("   [common_frechet] ", round(time() - t0, digits = 1), "s  rss=", round(peak_rss_mb(), digits = 0), "MB")

t0 = time(); pcx_meanzc = with_screen_counters(build_cm_meanzc_production_context(ctx, CS;
        L = CM_L, K_mean = MEANZC_K, K_pair = MEANZC_K, contrasts = :orthonormal, probs = PROBS_L50))
lp("   [cm_meanzc] ", round(time() - t0, digits = 1), "s  rss=", round(peak_rss_mb(), digits = 0), "MB")

const OZ_LAYOUT = OriginByPowerLayout(D, ORIGINZC_K, ORIGINZC_K)
t0 = time(); pcx_originzc = with_screen_counters(build_originzc_production_context(ctx, CS, OZ_LAYOUT))
lp("   [origin_zc] ", round(time() - t0, digits = 1), "s  n_eta=", n_eta(OZ_LAYOUT),
   "  rss=", round(peak_rss_mb(), digits = 0), "MB")

# ---- family-specific restriction coordinates (fixed across all starts) ----
# cm_meanzc: theoretical Exp(1) moments E[U^k]=k!, exactly smoke_delta1_cmzc.jl's
#            `exp.(log.(Float64.(factorial.(1:K_mean))))`.
const NU_MEANZC = Float64.(factorial.(1:MEANZC_K))
# origin_zc: per-origin empirical draw moments, exactly smoke_delta1_originzc.jl /
#            originzc_release_gateB_d20.jl's own nu0 construction.
const NU_ORIGINZC = begin
    nu = Vector{Float64}(undef, n_eta(OZ_LAYOUT))
    for k in 1:ORIGINZC_K
        Uk = ctx.U .^ k
        for o in 1:D
            nu[target_index(OZ_LAYOUT, o, k)] = mean(@view Uk[:, o])
        end
    end
    nu
end
lp(">> nu_meanzc=", NU_MEANZC)
lp(">> nu_originzc: n=", length(NU_ORIGINZC), " min=", minimum(NU_ORIGINZC), " max=", maximum(NU_ORIGINZC))

# ---------------------------------------------------------------------------
# Per-family single-point check.  Returns a NamedTuple, never throws.
# ---------------------------------------------------------------------------
const FAMILIES = ["unrestricted", "flexible_cm", "common_frechet", "cm_meanzc", "origin_zc"]

fail(fam, why, t) = (family = fam, ok = false, why = why, Delta = NaN, inner_status = missing,
                     verified = false, primal_dual_gap = NaN, mean_m_resid = NaN,
                     max_abs_moment_kkt_resid = NaN, m_min = NaN, wall = t)

function pass_or_fail(fam, verify, t)
    Δ = get(verify, :Delta_dual, NaN)
    cls = classify_inner_result(verify)
    ok_verified = (cls == VerifiedSolved)
    # 2026-07-28 correction (per user): a start point only needs a FINITE Delta*, not
    # Delta*<=0.01+tol -- the outer optimizer is what's expected to move the point down to
    # (or below) the budget during the real campaign. Requiring the un-optimized starting
    # point to already satisfy the delta=0.01 budget was rejecting nearly every perturbation
    # for a reason that isn't actually part of the start-acceptance criterion.
    ok_feasible = isfinite(Δ)
    why = !ok_verified ? "not_verified_success($(cls))" :
          !ok_feasible ? "Delta_nonfinite" : "ok"
    return (family = fam, ok = ok_verified && ok_feasible, why = why, Delta = Δ,
            inner_status = get(verify, :inner_status, missing), verified = ok_verified,
            primal_dual_gap = get(verify, :primal_dual_gap, NaN),
            mean_m_resid = get(verify, :mean_m_resid, NaN),
            max_abs_moment_kkt_resid = get(verify, :max_abs_moment_kkt_resid, NaN),
            m_min = get(verify, :m_min, NaN), wall = t)
end

function check_unrestricted(xf)
    t0 = time()
    try
        result, meta = evaluate_fullA_screened_ranged(xf, ctx, rsc;
            moment_representation = :compressed, use_cache = false)
        t = time() - t0
        ss = get(meta, :screen_status, :unknown)
        ss === :screen_passed || return (family = "unrestricted", ok = false,
            why = "screen_rejected($(ss))", Delta = get(result, :Delta_dual, NaN),
            inner_status = get(result, :inner_status, missing), verified = false,
            primal_dual_gap = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
            m_min = NaN, wall = t)
        return pass_or_fail("unrestricted", result, t)
    catch e
        return fail("unrestricted", "exception($(typeof(e)): $(sprint(showerror, e)[1:min(end,200)]))", time() - t0)
    end
end

function check_cm_family(fam, f)
    t0 = time()
    try
        _, _, verify = f()
        return pass_or_fail(fam, verify, time() - t0)
    catch e
        t = time() - t0
        e isa CMExpectedSolveFailure &&
            return fail(fam, "screen_or_solve_rejected($(sprint(showerror, e)[1:min(end,200)]))", t)
        return fail(fam, "exception($(typeof(e)): $(sprint(showerror, e)[1:min(end,200)]))", t)
    end
end

check_flexcm(xf)   = check_cm_family("flexible_cm",    () -> cm_production_value_verified_screened(xf, pcx_flexcm; counters = pcx_flexcm.screen_counters))
check_frechet(xf)  = check_cm_family("common_frechet", () -> cm_frechet_production_value_verified_screened(xf, pcx_frechet; counters = pcx_frechet.screen_counters))
check_meanzc(xf)   = check_cm_family("cm_meanzc",      () -> cm_meanzc_production_value_verified_screened(xf, NU_MEANZC, pcx_meanzc; counters = pcx_meanzc.screen_counters))
check_originzc(xf) = check_cm_family("origin_zc",      () -> cm_originzc_production_value_verified_screened(xf, NU_ORIGINZC, pcx_originzc; counters = pcx_originzc.screen_counters))

"Evaluate all five families at `xf`, short-circuiting on the first failure."
function check_all_families(xf; label = "")
    out = Dict{String,Any}()
    for (fam, chk) in (("unrestricted", check_unrestricted), ("flexible_cm", check_flexcm),
                       ("common_frechet", check_frechet), ("cm_meanzc", check_meanzc),
                       ("origin_zc", check_originzc))
        r = chk(xf)
        out[fam] = r
        @printf("      %-16s ok=%-5s Delta=%-12s why=%s (%.1fs)\n", fam, r.ok,
                isfinite(r.Delta) ? @sprintf("%.6e", r.Delta) : "NaN", r.why, r.wall)
        flush(stdout)
        r.ok || return (ok = false, failing = fam, results = out)
    end
    return (ok = true, failing = "", results = out)
end

# ---------------------------------------------------------------------------
# Post-acceptance gradient finiteness confirmation (acceptance criterion 3).
# Run only for ACCEPTED starts -- too expensive to run per rejected candidate.
# ---------------------------------------------------------------------------
function gradient_checks(xf)
    bw() = Dict{Int,Float64}()
    out = Dict{String,Any}()
    # unrestricted: same base-state -> composite_gradient_at_fast_buffered path
    # c10_d20_production_driver.jl::cb_G! uses.
    try
        r, m = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed, use_cache = false)
        base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        g, _ = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true,
                                                   h_mode = :cached, bandwidth_cache = bw())
        out["unrestricted"] = (finite = all(isfinite, g), norm = norm(g))
    catch e
        out["unrestricted"] = (finite = false, norm = NaN)
    end
    for (fam, gf) in (("flexible_cm",    () -> cm_production_gradient(xf, pcx_flexcm, ctx, pe; h_mode = :cached, bandwidth_cache = bw())),
                      ("common_frechet", () -> cm_frechet_production_gradient(xf, pcx_frechet, ctx, pe; h_mode = :cached, bandwidth_cache = bw())),
                      ("cm_meanzc",      () -> cm_meanzc_production_gradient(xf, NU_MEANZC, pcx_meanzc, ctx, pe; h_mode = :cached, bandwidth_cache = bw())),
                      ("origin_zc",      () -> cm_originzc_production_gradient(xf, NU_ORIGINZC, pcx_originzc, ctx, pe; h_mode = :cached, bandwidth_cache = bw())))
        try
            g, _ = gf()
            out[fam] = (finite = all(isfinite, g), norm = norm(g))
        catch e
            lp("      [grad ", fam, "] EXCEPTION ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 200)])
            out[fam] = (finite = false, norm = NaN)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
accepted = Vector{Any}()
attempts = Vector{Any}()     # full audit trail
n_cand = 0
t_search = time()

# ---- Start 1: the genuine calibration point ----
lp()
lp("-"^100)
lp("CANDIDATE 1  (label=calibration, radius=0.0) -- ctx.theta0_up's own calibrated A_od/gp")
lp("-"^100)
n_cand += 1
xf_calib, zfree_calib = decode_w(w_calib)
res_calib = check_all_families(xf_calib; label = "calibration")
push!(attempts, (idx = n_cand, label = "calibration", radius = 0.0, accepted = res_calib.ok,
                 failing = res_calib.failing, results = res_calib.results))
if !res_calib.ok
    lp()
    lp("!"^100)
    lp("FATAL: the genuine CALIBRATION point failed family '", res_calib.failing, "' (",
       res_calib.results[res_calib.failing].why, ").")
    lp("This is not a search failure -- it means the setup itself is wrong (context/coordinate/")
    lp("delta/W configuration), and no perturbation search should be run on top of it.")
    lp("!"^100)
    open(LOG_PATH, "w") do io
        println(io, "# Common five-family starts search -- ", STAMP, " (ABORTED)\n")
        println(io, "**FATAL**: the genuine calibration point (`ctx.θ0_up`'s own A_od/gp block) failed family `",
                res_calib.failing, "`: `", res_calib.results[res_calib.failing].why, "`.\n")
        println(io, "Configuration: W=", W, ", delta=", DELTA, ", draw_design=:", DRAW_DESIGN,
                ", draw_seed=", DRAW_SEED, ", destination_sample=:", DEST_SAMPLE, ".\n")
        println(io, "No perturbation search was run. See the run log for per-family detail.")
    end
    exit(2)
end
push!(accepted, (label = "start1_calibration", radius = 0.0, cand_idx = 1, w = copy(w_calib),
                 zfree = copy(zfree_calib), xf = copy(xf_calib), results = res_calib.results))
lp("  ==> ACCEPTED (start 1/", N_STARTS_POOL, ")")

# ---- Starts 2..N: deterministic seeded perturbations ----
Random.seed!(START_SEED)
radius = RADIUS0
n_at_radius = 0
radius_history = Vector{Any}()

while length(accepted) < N_STARTS_POOL && n_cand < MAX_CANDIDATES
    global radius, n_at_radius, n_cand
    if n_at_radius >= MAX_PER_RADIUS
        push!(radius_history, (radius = radius, tried = n_at_radius,
                               accepted = count(a -> a.radius == radius, accepted)))
        radius /= 2
        n_at_radius = 0
        lp()
        lp(">>> radius exhausted (", MAX_PER_RADIUS, " candidates), halving to radius=", radius)
    end
    n_cand += 1
    n_at_radius += 1
    wc = w_calib .+ radius .* (2 .* rand(NW) .- 1)      # audit_jach.jl:290 perturbation form
    lp()
    lp("-"^100)
    lp("CANDIDATE ", n_cand, "  (radius=", radius, ", draw #", n_at_radius, " at this radius)")
    lp("-"^100)
    gp_ok = ctx.bounds.γp_lo <= wc[1] <= ctx.bounds.γp_hi
    if !gp_ok
        lp("      gp=", wc[1], " outside theoretical bounds -- rejected before any solve")
        push!(attempts, (idx = n_cand, label = "perturbation", radius = radius, accepted = false,
                         failing = "gp_bounds", results = Dict{String,Any}()))
        continue
    end
    xfc, zfreec = decode_w(wc)
    rc = check_all_families(xfc)
    push!(attempts, (idx = n_cand, label = "perturbation", radius = radius, accepted = rc.ok,
                     failing = rc.failing, results = rc.results))
    if rc.ok
        push!(accepted, (label = "start$(length(accepted)+1)_perturbation", radius = radius,
                         cand_idx = n_cand, w = copy(wc), zfree = copy(zfreec), xf = copy(xfc),
                         results = rc.results))
        lp("  ==> ACCEPTED (start ", length(accepted), "/", N_STARTS_POOL, ")")
    else
        lp("  ==> REJECTED for ALL families (first failure: ", rc.failing, ")")
    end
end
push!(radius_history, (radius = radius, tried = n_at_radius,
                       accepted = count(a -> a.radius == radius, accepted)))
const SEARCH_WALL = time() - t_search

lp()
lp("="^100)
lp("SEARCH COMPLETE: ", length(accepted), "/", N_STARTS_POOL, " common starts accepted after ",
   n_cand, " candidate evaluations in ", round(SEARCH_WALL / 60, digits = 1), " min")
lp("="^100)

# ---------------------------------------------------------------------------
# SELECT_BY_DISTANCE: campaign brief requires the 2 non-calibration starts to "maximize
# minimum pairwise scaled distance" (to calibration AND to each other) -- NOT "first 2 found",
# which is all the source script (common_five_starts_search.jl) did. Distance = Euclidean in
# the shared w=[gp; a_nonpivot] transformed-A outer coordinate -- the same commensurate space
# the perturbation radius itself is drawn in (`w_calib .+ radius .* (2 .* rand(NW) .- 1)`
# above), so this is a like-for-like "scaled" distance, not raw natural-coordinate units.
# ---------------------------------------------------------------------------
function select_by_distance(pool, n_final)
    calib = pool[1]
    perturbations = pool[2:end]
    n_needed = n_final - 1
    dist(a, b) = norm(a.w .- b.w)
    if length(perturbations) < n_needed
        lp("!!! WARNING: only ", length(perturbations), " perturbations in the pool, need ", n_needed, " -- using all available")
        return vcat([calib], perturbations)
    elseif n_needed == 2
        best_pair = (1, 2); best_min_dist = -Inf
        for i in 1:length(perturbations), j in (i+1):length(perturbations)
            p1, p2 = perturbations[i], perturbations[j]
            mind = min(dist(calib, p1), dist(calib, p2), dist(p1, p2))
            if mind > best_min_dist
                best_min_dist = mind
                best_pair = (i, j)
            end
        end
        lp("  selected pair: pool indices ", best_pair, " of ", length(perturbations),
           " (min pairwise distance = ", best_min_dist, ")")
        return vcat([calib], [perturbations[best_pair[1]], perturbations[best_pair[2]]])
    else
        error("select_by_distance: only n_final=3 (n_needed=2) is implemented (greedy/exhaustive pair search); got n_final=$n_final")
    end
end

lp()
lp("="^100)
lp("SELECT_BY_DISTANCE: choosing ", N_STARTS_FINAL - 1, " best-separated perturbations from ",
   length(accepted) - 1, " candidates in the pool")
lp("="^100)
const POOL = accepted   # keep the full pool for the log/manifest audit trail
accepted = select_by_distance(POOL, N_STARTS_FINAL)
lp("  FINAL: ", length(accepted), " starts selected -- ", join([a.label for a in accepted], ", "))

# ---- gradient confirmation on the SELECTED starts ----
grads = Vector{Any}()
for (i, a) in enumerate(accepted)
    lp(">> gradient finiteness check, ", a.label)
    g = gradient_checks(a.xf)
    for fam in FAMILIES
        lp("      ", rpad(fam, 16), " finite=", g[fam].finite, " |g|=", g[fam].norm)
    end
    push!(grads, g)
end

# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
jesc(s) = "\"" * replace(string(s), "\\" => "\\\\", "\"" => "\\\"") * "\""
jnum(x) = (x isa Missing || (x isa AbstractFloat && !isfinite(x))) ? "null" : string(x)
jvec(v) = "[" * join((jnum(x) for x in v), ", ") * "]"
jmat(M) = "[" * join(("[" * join((jnum(x) for x in M[i, :]), ", ") * "]" for i in 1:size(M, 1)), ", ") * "]"

open(JSON_PATH, "w") do io
    println(io, "{")
    println(io, "  \"generated\": ", jesc(string(now())), ",")
    println(io, "  \"script\": \"full_aod_diag/d4_exact/common_five_starts_search.jl\",")
    println(io, "  \"config\": {")
    println(io, "    \"W\": ", W, ", \"delta\": ", DELTA, ", \"find_smallest\": ", FIND_SMALLEST, ",")
    println(io, "    \"draw_design\": ", jesc(DRAW_DESIGN), ", \"draw_seed\": ", DRAW_SEED, ",")
    println(io, "    \"destination_sample\": ", jesc(DEST_SAMPLE), ", \"D\": ", D, ", \"D_dest\": ", Ddest, ",")
    println(io, "    \"start_seed\": ", START_SEED, ", \"radius0\": ", RADIUS0,
                ", \"max_per_radius\": ", MAX_PER_RADIUS, ", \"max_candidates\": ", MAX_CANDIDATES, ",")
    println(io, "    \"A_coordinate_mode\": \"powered_aspace\", \"cm_L\": ", CM_L,
                ", \"cm_contrasts\": \"orthonormal\", \"meanzc_K_mean\": ", MEANZC_K,
                ", \"meanzc_K_pair\": ", MEANZC_K, ", \"originzc_K_mean\": ", ORIGINZC_K,
                ", \"originzc_K_pair\": ", ORIGINZC_K, ",")
    println(io, "    \"feasibility_tol\": ", FEAS_TOL, ", \"verified_success_tol\": {",
                "\"primal_dual_gap_tol\": ", DEFAULT_VERIFIED_SUCCESS_TOL.primal_dual_gap_tol,
                ", \"mean_m_resid_tol\": ", DEFAULT_VERIFIED_SUCCESS_TOL.mean_m_resid_tol,
                ", \"max_abs_moment_kkt_resid_tol\": ", DEFAULT_VERIFIED_SUCCESS_TOL.max_abs_moment_kkt_resid_tol, "},")
    println(io, "    \"draw_checksum_uniform\": ", jesc(ctx.draw_meta.checksum_uniform), ",")
    println(io, "    \"draw_checksum_transformed\": ", jesc(ctx.draw_meta.checksum_transformed))
    println(io, "  },")
    println(io, "  \"shared_extra_coordinates\": {")
    println(io, "    \"cm_meanzc_nu\": ", jvec(NU_MEANZC), ",")
    println(io, "    \"origin_zc_nu\": ", jvec(NU_ORIGINZC))
    println(io, "  },")
    println(io, "  \"n_candidates_evaluated\": ", n_cand, ",")
    println(io, "  \"n_accepted\": ", length(accepted), ",")
    println(io, "  \"search_wall_s\": ", round(SEARCH_WALL, digits = 1), ",")
    println(io, "  \"starts\": [")
    for (i, a) in enumerate(accepted)
        A = exp.(pivot_expand(a.zfree, pe))
        println(io, "    {")
        println(io, "      \"index\": ", i, ", \"label\": ", jesc(a.label), ",")
        println(io, "      \"radius\": ", a.radius, ", \"candidate_index\": ", a.cand_idx, ",")
        println(io, "      \"checksum_w_hash\": ", jesc(string(hash(a.w), base = 16)), ",")
        println(io, "      \"checksum_xf_hash\": ", jesc(string(hash(a.xf), base = 16)), ",")
        println(io, "      \"gp\": ", jnum(a.w[1]), ",")
        println(io, "      \"w_transformed_a\": ", jvec(a.w), ",")
        println(io, "      \"zfree_pivot_reduced_log\": ", jvec(a.zfree), ",")
        println(io, "      \"x_free_natural\": ", jvec(a.xf), ",")
        println(io, "      \"A_od\": ", jmat(A), ",")
        println(io, "      \"families\": {")
        for (k, fam) in enumerate(FAMILIES)
            r = a.results[fam]; g = grads[i][fam]
            print(io, "        ", jesc(fam), ": {\"ok\": ", r.ok, ", \"Delta_star\": ", jnum(r.Delta),
                  ", \"inner_status\": ", jnum(r.inner_status), ", \"verified_success\": ", r.verified,
                  ", \"primal_dual_gap\": ", jnum(r.primal_dual_gap),
                  ", \"mean_m_resid\": ", jnum(r.mean_m_resid),
                  ", \"max_abs_moment_kkt_resid\": ", jnum(r.max_abs_moment_kkt_resid),
                  ", \"m_min\": ", jnum(r.m_min), ", \"solve_wall_s\": ", jnum(round(r.wall, digits = 3)),
                  ", \"gradient_finite\": ", g.finite, ", \"gradient_norm\": ", jnum(g.norm), "}")
            println(io, k == length(FAMILIES) ? "" : ",")
        end
        println(io, "      }")
        println(io, i == length(accepted) ? "    }" : "    },")
    end
    println(io, "  ],")
    println(io, "  \"attempts\": [")
    for (k, at) in enumerate(attempts)
        print(io, "    {\"candidate_index\": ", at.idx, ", \"label\": ", jesc(at.label),
              ", \"radius\": ", at.radius, ", \"accepted\": ", at.accepted,
              ", \"first_failing_family\": ", jesc(at.failing), ", \"why\": ",
              jesc(haskey(at.results, at.failing) ? at.results[at.failing].why : at.failing), "}")
        println(io, k == length(attempts) ? "" : ",")
    end
    println(io, "  ]")
    println(io, "}")
end
lp(">> wrote ", JSON_PATH)

open(CSV_PATH, "w") do io
    println(io, "start_index,start_label,radius,candidate_index,checksum_w_hash,gp,family,ok,Delta_star,",
                "delta_budget,inner_status,verified_success,primal_dual_gap,mean_m_resid,",
                "max_abs_moment_kkt_resid,m_min,solve_wall_s,gradient_finite,gradient_norm,W,draw_design,draw_seed,start_seed")
    for (i, a) in enumerate(accepted), fam in FAMILIES
        r = a.results[fam]; g = grads[i][fam]
        println(io, i, ",", a.label, ",", a.radius, ",", a.cand_idx, ",", string(hash(a.w), base = 16), ",",
                a.w[1], ",", fam, ",", r.ok, ",", r.Delta, ",", DELTA, ",", r.inner_status, ",", r.verified, ",",
                r.primal_dual_gap, ",", r.mean_m_resid, ",", r.max_abs_moment_kkt_resid, ",", r.m_min, ",",
                round(r.wall, digits = 3), ",", g.finite, ",", g.norm, ",", W, ",", DRAW_DESIGN, ",",
                DRAW_SEED, ",", START_SEED)
    end
end
lp(">> wrote ", CSV_PATH)

open(LOG_PATH, "w") do io
    println(io, "# Common five-family starting points -- search log (", STAMP, ")\n")
    println(io, "Script: `full_aod_diag/d4_exact/common_five_starts_search.jl`  ")
    println(io, "Run: ", now(), "  |  wall ", round(SEARCH_WALL / 60, digits = 1), " min  |  peak RSS ",
            round(peak_rss_mb(), digits = 0), " MB\n")
    println(io, "## Configuration\n")
    println(io, "| key | value |")
    println(io, "|---|---|")
    for (k, v) in (("W", W), ("delta", DELTA), ("find_smallest", FIND_SMALLEST),
                   ("draw_design", DRAW_DESIGN), ("draw_seed", DRAW_SEED),
                   ("destination_sample", DEST_SAMPLE), ("D", D), ("D_dest", Ddest),
                   ("A_coordinate_mode", :powered_aspace), ("start_seed", START_SEED),
                   ("radius0", RADIUS0), ("max_per_radius", MAX_PER_RADIUS),
                   ("max_candidates", MAX_CANDIDATES), ("feasibility_tol", FEAS_TOL),
                   ("cm_L", CM_L), ("meanzc_K", MEANZC_K), ("originzc_K", ORIGINZC_K),
                   ("draw_checksum_uniform", ctx.draw_meta.checksum_uniform),
                   ("draw_checksum_transformed", ctx.draw_meta.checksum_transformed))
        println(io, "| `", k, "` | `", v, "` |")
    end
    println(io, "\n## Outcome\n")
    println(io, "**", length(accepted), " of ", N_STARTS_POOL, " common starts accepted** from ",
            n_cand, " candidate evaluations.\n")
    if length(accepted) < N_STARTS_POOL
        println(io, "> **INCOMPLETE**: fewer than ", N_STARTS_POOL, " starts were found within the budget ",
                "(max_candidates=", MAX_CANDIDATES, ", max_per_radius=", MAX_PER_RADIUS,
                ", final radius=", radius, "). No substitute starts were fabricated.\n")
    end
    println(io, "| start | label | radius | candidate # | gp | max Delta* over families |")
    println(io, "|---|---|---|---|---|---|")
    for (i, a) in enumerate(accepted)
        println(io, "| ", i, " | `", a.label, "` | ", a.radius, " | ", a.cand_idx, " | ",
                round(a.w[1], digits = 8), " | ",
                @sprintf("%.4e", maximum(a.results[f].Delta for f in FAMILIES)), " |")
    end
    println(io, "\n## Radius schedule\n")
    println(io, "| radius | candidates tried | accepted |")
    println(io, "|---|---|---|")
    for rh in radius_history
        println(io, "| ", rh.radius, " | ", rh.tried, " | ", rh.accepted, " |")
    end
    println(io, "\n## Candidate-by-candidate audit trail\n")
    println(io, "Rejection policy: a candidate failing ANY family is rejected for ALL five. ",
            "Families are evaluated in the order unrestricted, flexible_cm, common_frechet, ",
            "cm_meanzc, origin_zc and the check short-circuits at the first failure, so ",
            "`first failing family` is the first in that order to fail, not necessarily the only one.\n")
    println(io, "| candidate | kind | radius | accepted | first failing family | reason |")
    println(io, "|---|---|---|---|---|---|")
    for at in attempts
        why = haskey(at.results, at.failing) ? at.results[at.failing].why : at.failing
        println(io, "| ", at.idx, " | ", at.label, " | ", at.radius, " | ", at.accepted, " | ",
                isempty(at.failing) ? "-" : at.failing, " | `", why, "` |")
    end
    println(io, "\n## Per-family rejection counts\n")
    println(io, "| family | times it was the first failure |")
    println(io, "|---|---|")
    for fam in vcat(FAMILIES, ["gp_bounds"])
        println(io, "| ", fam, " | ", count(a -> a.failing == fam, attempts), " |")
    end
    println(io, "\n## Screen counters (cumulative over the whole search)\n")
    println(io, "| family | calls | pairwise hits | hard-winner hits | passed |")
    println(io, "|---|---|---|---|---|")
    for (fam, p) in (("flexible_cm", pcx_flexcm), ("common_frechet", pcx_frechet),
                     ("cm_meanzc", pcx_meanzc), ("origin_zc", pcx_originzc))
        nt = as_namedtuple(p.screen_counters)
        println(io, "| ", fam, " | ", nt.calls, " | ", nt.pairwise, " | ", nt.winner, " | ", nt.passed, " |")
    end
    println(io, "\n## Artifacts\n")
    println(io, "- `", basename(JSON_PATH), "` -- full manifest (coordinates, A_od, per-family Delta*/residuals/checksums)")
    println(io, "- `", basename(CSV_PATH), "` -- one row per (start, family)")
end
lp(">> wrote ", LOG_PATH)

cp(JSON_PATH, FINAL_JSON_PATH; force = true)
lp(">> wrote ", FINAL_JSON_PATH, " (canonical frozen 3-start manifest, copy of ", basename(JSON_PATH), ")")

lp()
lp("="^100)
lp("DONE: ", length(accepted), "/", N_STARTS_FINAL, " common starts selected (from a pool of ",
   length(POOL), "); peak RSS ", round(peak_rss_mb(), digits = 0), " MB")
lp("="^100)
length(accepted) == N_STARTS_FINAL || error("three_starts_search: only ", length(accepted),
    "/", N_STARTS_FINAL, " starts selected -- refusing to declare success")
