# production_all_hessian_audit_harness_2026-08-02.jl
#
# Canonical fresh-process audit harness for the production Hessian allocation/efficiency audit
# (PRODUCTION_HESSIAN_AUDIT_MASTER_2026-08-02.md). Covers all 5 families under IDENTICAL
# conventions: unrestricted | flexible_cm | common_frechet | origin_zc | cm_meanzc.
#
# Uses the canonical production include stack: campaign_cm_family_runner.jl's own list (covers
# flexible_cm/common_frechet/cm_meanzc/origin_zc) MERGED with campaign_unrestricted_runner.jl's
# own list (covers unrestricted) -- these are the two lists the 21fa6ec production merge commit
# itself documents as canonical; no manual/date-stamped candidate includes added beyond what each
# list already names.
#
# Scientific configuration (matches PRODUCTION_HESSIAN_AUDIT_SOURCE_SNAPSHOT_2026-08-02.md /
# the 3 real production driver functions' own current kwarg defaults, confirmed live by reading
# c10_d20_production_driver_unified.jl/cm_checkpoint.jl/cm_originzc_checkpoint.jl source):
#   sigma=3 (σHat=3.0), own-trade excluded (exclude_diagonal_gravity=true), Brazil->Korea excluded
#   (gravity_exclude_cells=default_gravity_exclude_cells_brazil_korea()), destination_sample=
#   :exclude_row, focal country=France (bi, AD_PARAMS default), randomized Sobol draws
#   (draw_design=:sobol_randomized, draw_seed=20260719 -- same constant campaign_cm_family_runner.jl
#   uses).
#
# Usage:
#   OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 julia --project=. -t 10 \
#       production_all_hessian_audit_harness_2026-08-02.jl
#   ENV vars: AUDIT_FAMILY (unrestricted|flexible_cm|common_frechet|origin_zc|cm_meanzc),
#             AUDIT_W (default 100000), AUDIT_NREPEAT (default 100), AUDIT_OUTDIR

const D4X = @__DIR__
cd(D4X)

# ---- merged canonical include stack (dedup of campaign_cm_family_runner.jl +
# campaign_unrestricted_runner.jl's own lists; order preserved from the CM list, unrestricted's
# 5 extra files appended at the end since none overlap) ----
const _CM_FAMILY_LIST = [
    "draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
    "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
    "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
    "hcz_drawchunk_candidate_2026-07-29.jl",
    "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl",
]
const _UNRESTRICTED_EXTRA_LIST = [
    "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
    "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl",
]
# Extra files needed directly by THIS harness (not transitively pulled by either list above,
# confirmed by iterative UndefVarError resolution against a real run -- see harness build log in
# PRODUCTION_HESSIAN_AUDIT_MASTER_2026-08-02.md section 4):
const _HARNESS_EXTRA_LIST = String[]

const CANONICAL_INCLUDE_LIST = unique(vcat(_CM_FAMILY_LIST, _UNRESTRICTED_EXTRA_LIST, _HARNESS_EXTRA_LIST))

for f in CANONICAL_INCLUDE_LIST
    include(joinpath(D4X, f))
end
include(joinpath(D4X, "json_lite.jl"))

using Printf, LinearAlgebra, Random, Statistics, Dates
using Base.Threads: nthreads

lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = Symbol(get(ENV, "AUDIT_FAMILY", "cm_meanzc"))
const W = parse(Int, get(ENV, "AUDIT_W", "100000"))
const N_REPEAT = parse(Int, get(ENV, "AUDIT_NREPEAT", "100"))
const OUTDIR = get(ENV, "AUDIT_OUTDIR", joinpath(D4X, "..", "..", "results", "production_hessian_audit_2026-08-02"))
mkpath(OUTDIR)

FAMILY in (:unrestricted, :flexible_cm, :common_frechet, :origin_zc, :cm_meanzc) ||
    error("production_all_hessian_audit_harness: unknown AUDIT_FAMILY=$FAMILY")

# ---- canonical scientific configuration (see file header) ----
const DRAW_DESIGN = :sobol_randomized
const DRAW_SEED = 20260719
const CM_L = 50
const K_ZC = 3   # live production K_mean/K_pair (task brief: "expected K=3", NOT the K=1 the
                 # campaign_cm_family_runner.jl "profile"-labelled default uses)

lp("="^100)
lp("PRODUCTION ALL-HESSIAN AUDIT HARNESS -- ", Dates.now(), "  family=", FAMILY, " W=", W,
   " n_repeat=", N_REPEAT, " pid=", getpid())
lp("julia_threads=", nthreads(), " BLAS_threads=", BLAS.get_num_threads())
lp("scientific config: sigma=3.0(sigmaHat) exclude_diagonal_gravity=true gravity_exclude_cells=Brazil->Korea",
   " destination_sample=:exclude_row draw_design=", DRAW_DESIGN, " draw_seed=", DRAW_SEED)
lp("="^100)

# ---- context construction (CONTEXT_SETUP) ----
t_ctx0 = time()
const GRAV_EXCLUDE = default_gravity_exclude_cells_brazil_korea()
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV_EXCLUDE, σHat = 3.0)
t_ctx_s = time() - t_ctx0
lp(">> context built in ", round(t_ctx_s, digits = 2), "s.  D=", ctx0.D, " D_dest=", ctx0.D_dest,
   " focal_country(bi)=", ctx0.bi, " sigma(σ)=", ctx0.σ, " W=", ctx0.W,
   " gravity_exclude_cells=", ctx0.gravity_exclude_cells, " exclude_diagonal_gravity=", ctx0.exclude_diagonal_gravity)

x_free_calib = ctx0.θ0_up[ctx0.free_idx]

# ---- per-family setup: returns (userParams, cb, x_state, n, hlen, cold_solve_s, backend_info) ----
function setup_unrestricted(ctx0, x_free_calib)
    ctx = build_unrestricted_operator_ctx(ctx0)   # :operator production default
    obj = ctx.obj
    θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
    t0 = time()
    K_hard, inner_x, nStatus, n_fg, n_hess, st = inner_loop_internal_compressed(obj, θ_full0, ctx)
    cold_solve_s = time() - t0
    nStatus in (0, -100, -101, -103) || error("setup_unrestricted: TRUE-COLD base solve failed, nStatus=$nStatus")
    x_state = collect(inner_x)
    n = length(x_state)
    hlen = n * (n + 1) ÷ 2
    cb = _callbackEvalH_inner_compressed!
    backend_info = (bundle_type = typeof(obj), core_hessian_backend = UNRESTRICTED_CORE_HESSIAN_BACKEND[])
    return (userParams = st, cb = cb, x_state = x_state, n = n, hlen = hlen,
            cold_solve_s = cold_solve_s, n_fg = n_fg, n_hess = n_hess, nStatus = nStatus, backend_info = backend_info)
end

function setup_flexible_cm(ctx0, x_free_calib)
    SNAPS = nested_grid_sequence([10, 20, 50])
    ctx_cm, aug, bins, cctx = build_cm_production_context(ctx0, CS; L = CM_L, contrasts = :orthonormal, probs = SNAPS[CM_L])
    t0 = time()
    base = archC_base_state(x_free_calib, ctx_cm, cctx)
    cold_solve_s = time() - t0
    x_state = vcat(base.ζstar, base.λstar)
    n = length(x_state)
    hlen = n * (n + 1) ÷ 2
    cb = archC_hess_cb_builder(cctx)
    backend_info = (core_hessian_backend = cctx.core_hessian_backend, cross_hessian_backend = cctx.cm_cross_hessian_backend,
                    use_threaded_bins = cctx.use_threaded_bins, inner_fg_backend = cctx.inner_fg_backend)
    return (userParams = ctx_cm.obj, cb = cb, x_state = x_state, n = n, hlen = hlen,
            cold_solve_s = cold_solve_s, n_fg = missing, n_hess = missing, nStatus = base.inner_status, backend_info = backend_info)
end

function setup_common_frechet(ctx0, x_free_calib)
    SNAPS = nested_grid_sequence([10, 20, 50])
    pcx = build_cm_frechet_production_context(ctx0, CS; L = CM_L, contrasts = :orthonormal, probs = SNAPS[CM_L], cm_hessian_backend = :structured)
    ctx_cm = pcx.ctx_cm; cctx = pcx.cctx; aug = pcx.aug
    t0 = time()
    base = archC_frechet_base_state(x_free_calib, ctx_cm, cctx, aug.level_targets)
    cold_solve_s = time() - t0
    x_state = vcat(base.ζstar, base.λstar)
    n = length(x_state)
    hlen = n * (n + 1) ÷ 2
    cb = archC_frechet_hess_cb_builder(cctx, aug.level_targets)
    backend_info = (core_hessian_backend = cctx.core_hessian_backend, cross_hessian_backend = cctx.cm_cross_hessian_backend,
                    use_threaded_bins = cctx.use_threaded_bins, inner_fg_backend = cctx.inner_fg_backend)
    return (userParams = ctx_cm.obj, cb = cb, x_state = x_state, n = n, hlen = hlen,
            cold_solve_s = cold_solve_s, n_fg = missing, n_hess = missing, nStatus = base.inner_status, backend_info = backend_info)
end

function setup_cm_meanzc(ctx0, x_free_calib)
    ctx_cm, aug, cctx, bins = build_cm_meanzc_production_context(ctx0, CS; L = CM_L, K_mean = K_ZC, K_pair = K_ZC,
        contrasts = :orthonormal, meanzc_basis = :direct)
    νvec = Float64.(factorial.(1:K_ZC))   # NOT ones(K) -- see gate5_single_solve_worker_2026-08-01.jl's
    # own note: ones(3) is infeasible (nStatus=-300) for K_mean=3 direct construction.
    t0 = time()
    base = archC_meanzc_base_state(x_free_calib, νvec, ctx_cm, cctx)
    cold_solve_s = time() - t0
    x_state = vcat(base.ζstar, base.λstar)
    n = length(x_state)
    hlen = n * (n + 1) ÷ 2
    cb = archC_hess_cb_builder(cctx)
    backend_info = (core_hessian_backend = cctx.core_hessian_backend, zc_gram_backend = cctx.zc_gram_backend,
                    hcz_prep_backend = cctx.hcz_prep_backend, zc_ez_backend = cctx.zc_ez_backend,
                    use_threaded_bins = cctx.use_threaded_bins)
    return (userParams = ctx_cm.obj, cb = cb, x_state = x_state, n = n, hlen = hlen,
            cold_solve_s = cold_solve_s, n_fg = missing, n_hess = missing, nStatus = base.inner_status, backend_info = backend_info)
end

function setup_origin_zc(ctx0, x_free_calib)
    layout = OriginByPowerLayout(ctx0.D, K_ZC, K_ZC)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for k in 1:K_ZC, o in 1:ctx0.D
        nu0[target_index(layout, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    pcx = build_originzc_production_context(ctx0, CS, layout; fg_backend = :operator, moment_representation = :operator)
    ctx_cm = pcx.ctx_cm; octx = pcx.octx
    t0 = time()
    base = archOZ_base_state(x_free_calib, nu0, ctx_cm)
    cold_solve_s = time() - t0
    x_state = vcat(base.ζstar, base.λstar)
    n = length(x_state)
    hlen = n * (n + 1) ÷ 2
    cb = archA_partitioned_hess_cb_builder(octx)
    backend_info = (core_hessian_backend = octx.core_hessian_backend, zc_gram_backend = octx.zc_gram_backend,
                    zc_ez_backend = octx.zc_ez_backend)
    return (userParams = ctx_cm.obj, cb = cb, x_state = x_state, n = n, hlen = hlen,
            cold_solve_s = cold_solve_s, n_fg = missing, n_hess = missing, nStatus = base.inner_status, backend_info = backend_info)
end

const SETUP_FNS = Dict(:unrestricted => setup_unrestricted, :flexible_cm => setup_flexible_cm,
    :common_frechet => setup_common_frechet, :cm_meanzc => setup_cm_meanzc, :origin_zc => setup_origin_zc)

lp(">> [TRUE_COLD_INNER_SOLVE] building family context + running FIRST (cold) inner solve for family=", FAMILY, " ...")
res = SETUP_FNS[FAMILY](ctx0, x_free_calib)
res.nStatus in (0, -100, -101, -103) || error("TRUE-COLD base solve failed for family=$FAMILY, nStatus=$(res.nStatus)")
lp(">> [TRUE_COLD_INNER_SOLVE] done.  cold_solve_s=", round(res.cold_solve_s, digits = 4),
   "  nStatus=", res.nStatus, "  n_fg=", res.n_fg, "  n_hess=", res.n_hess,
   "  n(dual dim)=", res.n, "  packed_hlen=", res.hlen)
lp(">> backend_info: ", res.backend_info)

# ---- FROZEN_STATE_CALLBACK: same verified dual/state, isolated Hessian callback ----
fake_req = (x = res.x_state,)
h = Vector{Float64}(undef, res.hlen)
fake_res = (hess = h,)

# warm-up (JIT) call -- excluded from all timing/allocation numbers per CONTEXT_SETUP/JIT convention
res.cb(nothing, nothing, fake_req, fake_res, res.userParams)
h_warm = copy(h)

# ONE frozen-state callback, timed + allocation-profiled
t_single = @elapsed res.cb(nothing, nothing, fake_req, fake_res, res.userParams)
b_single = @allocated res.cb(nothing, nothing, fake_req, fake_res, res.userParams)
h_single = copy(h)
lp(">> [FROZEN_STATE_CALLBACK single] t=", @sprintf("%.6f", t_single), "s  bytes=", b_single,
   "  max|h-h_warm|=", maximum(abs.(h_single .- h_warm)))

# N_REPEAT frozen-state callbacks (repeated, same state) -- per-call times AND total allocation
times_repeat = Vector{Float64}(undef, N_REPEAT)
for i in 1:N_REPEAT
    times_repeat[i] = @elapsed res.cb(nothing, nothing, fake_req, fake_res, res.userParams)
end
bytes_repeat_total = @allocated (for i in 1:N_REPEAT; res.cb(nothing, nothing, fake_req, fake_res, res.userParams); end)
h_final = copy(h)
max_drift = maximum(abs.(h_final .- h_warm))
lp(">> [FROZEN_STATE_CALLBACK x", N_REPEAT, "] mean_t=", @sprintf("%.6f", mean(times_repeat)),
   "s  median_t=", @sprintf("%.6f", median(times_repeat)), "s  min_t=", @sprintf("%.6f", minimum(times_repeat)),
   "s  max_t=", @sprintf("%.6f", maximum(times_repeat)), "s  total_bytes=", bytes_repeat_total,
   "  bytes_per_call=", round(bytes_repeat_total / N_REPEAT, digits = 1),
   "  max|h drift across repeats|=", max_drift)
max_drift < 1e-8 || lp("WARNING: repeated frozen-state callback outputs drifted by ", max_drift,
    " -- callback is NOT purely a function of (x_state, userParams); investigate before trusting repeat timing as directly comparable across calls.")

# ---- write result row ----
outfile = joinpath(OUTDIR, "$(FAMILY)_W$(W)_audit_harness_pid$(getpid()).csv")
open(outfile, "w") do io
    println(io, "family,W,K_zc,pid,julia_threads,blas_threads,ctx_build_s,cold_solve_s,n_fg,n_hess,nStatus,n_dual_dim,packed_hlen,",
        "t_single_s,bytes_single,mean_t_repeat_s,median_t_repeat_s,min_t_repeat_s,max_t_repeat_s,bytes_repeat_total,bytes_per_call,",
        "max_h_drift,backend_info,timestamp")
    println(io, "$FAMILY,$W,$K_ZC,$(getpid()),$(nthreads()),$(BLAS.get_num_threads()),$t_ctx_s,$(res.cold_solve_s),",
        "$(res.n_fg),$(res.n_hess),$(res.nStatus),$(res.n),$(res.hlen),",
        "$t_single,$b_single,$(mean(times_repeat)),$(median(times_repeat)),$(minimum(times_repeat)),$(maximum(times_repeat)),",
        "$bytes_repeat_total,$(bytes_repeat_total/N_REPEAT),$max_drift,\"$(res.backend_info)\",$(now())")
end
lp(">> wrote ", outfile)
lp("PRODUCTION ALL-HESSIAN AUDIT HARNESS COMPLETE -- family=", FAMILY)
lp("="^100)
