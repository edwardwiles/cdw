# ============================================================================
# Continuation 10, Sections 5+6: THE production D=20 driver.
#
# This is the file the coordinating session should use for Section 10's real
# frontier runs (see docs/fullA_D20_checkpoint_resume_report.md, which states
# this explicitly). Built as a direct extension of
# `c9_phase8_d20_pilot.jl`'s `profile_minimize`/`joint_polish` architecture
# (same production config: compressed moments, h_mode=:cached bandwidth
# policy, multi_method=:top3, SR1 outer Hessian -- unchanged, already gated
# PASS by Continuation 9 Phases 6/7/8). NOT a rewrite from scratch.
#
# Two things are new relative to that pilot driver:
#
#   SECTION 5 -- every evaluation goes through `evaluate_fullA_screened_ranged`
#   (fast_range_screen.jl, integration/fullA-fast-range-screen -- promoted to
#   THE production screening path; the underlying pairwise/witness/zero-winner
#   checks are infeasibility_screen.jl's, unchanged), reusing the ctx-level
#   `pairwise`/`witness` structures `context_real_d20.jl::d20_real_setup`
#   builds ONCE at context-construction time plus a per-ctx
#   `RangedScreenContext` (`rsc`, built once via `build_ranged_screen_context`
#   right after `ctx`) in the order: pairwise cert -> pre-winner envelope
#   cert -> witness -> destination winner-scan (fused zero-winner + winning-
#   range checks, single pass) -> general range-screen safety net (reusing
#   the already-built CompressedFactual) -> CC inner solve. A raw, unscreened
#   `evaluate_fullA_fast` call no longer appears anywhere in this driver's
#   hot path. See docs/fullA_fast_range_screen_production_integration.md for
#   the validation/benchmark backing this promotion (real D=20/W=80,000, 0
#   false positives across the recovered pathology catalogue, no measurable
#   overhead on warm feasible calls).
#
#   SECTION 6 -- checkpoint/resume. Every accepted KNITRO outer iterate (via
#   `KN_set_newpt_callback`), every new best-feasible point, every
#   `checkpoint_interval_s` of wall time, and the end of every profile/polish
#   stage triggers a checkpoint (`D20Checkpoint`, serialized via the stdlib
#   `Serialization` module -- no extra dependency). A checkpoint captures
#   everything askED for that this investigation's KNITRO.jl API surface
#   actually exposes: g, full+reduced log-A, the CC inner dual warm start
#   (`ctx.obj.x`), the current best feasible point, the FD bandwidth cache,
#   enough state to cheaply rebuild the SAME infeasibility-witness (the draw
#   seed -- see the note below on why this had to be introduced), the
#   branch/delta, and the draw seed. It does NOT capture KNITRO's internal
#   quasi-Newton (SR1/BFGS/L-BFGS) Hessian-approximation state -- the
#   KNITRO.jl/C API does not expose extracting or reinjecting that across
#   separate `KN_new()` instances (confirmed by inspecting `names(KNITRO,
#   all=true)` for anything hessian/state/restart-shaped -- nothing found
#   beyond `KN_get_hessian_values`, which reports the CURRENT exact-Hessian
#   evaluation for exact-Hessian modes, not the internal QN approximation
#   for hessopt=SR1/BFGS/L-BFGS this driver actually uses). A resumed run
#   restarts its outer Hessian approximation from KNITRO's own default
#   initialization -- same as any fresh warm-started solve.
#
# A REAL, non-obvious finding this task surfaced: `context_real_d20.jl`'s
# underlying `importData`/`genRands.jl::genExpRands!` draws the W Frechet
# simulation support via the GLOBAL Julia RNG (`rand!(U)`), UNSEEDED for the
# real-data (`fakeData==3`) path (only the synthetic `fakeData in (1,2)` paths
# call `Random.seed!`). This means `ctx.U` -- and therefore every Delta_dual/
# gravity/moment-residual number this whole investigation computes -- is NOT
# reproducible across separate Julia processes unless something seeds the RNG
# first. A checkpoint/resume driver by definition restarts in a NEW process,
# so THIS driver introduces the explicit `Random.seed!(draw_seed)` call
# (immediately before `d20_real_setup`) that the rest of this investigation's
# single-process scripts never needed, and records `draw_seed` in every
# checkpoint so a resume reconstructs the bit-identical `ctx.U` (and hence
# `ctx.pairwise`/`ctx.witness`/every downstream calculation). This is
# validated directly, not assumed -- see the bottom of this file / the
# checkpoint-resume report.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes infeasibility_screen.jl too (Section 5)
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))   # Continuation 10 Section 9: structured dense-materialize, used by compressed_live.jl / infeasibility_screen.jl
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "lfix_buffer_reuse.jl"))   # Continuation 11 Section 2: validated bit-identical vs composite_gradient_at_fast (0.0 diff, 5 points incl. trajectory test), ~1.1-1.9x faster; now the default gradient below
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))   # pre-winner envelope + fused winning-range + general safety-net screens -- THE production screening path via evaluate_fullA_screened_ranged, wired into screened_eval below (used by every cb_F!/cb_G!/cb_newpt! callback); see docs/fullA_fast_range_screen_production_integration.md
using KNITRO, Printf, Dates, Random, Statistics, Serialization
using LinearAlgebra: norm, dot

const FEASIBLE_CODES = (0, -100, -101, -103)
const SOLVER_STATE_NOTE = "KNITRO's internal quasi-Newton (SR1/BFGS/L-BFGS) Hessian-approximation " *
    "state is NOT exposed by the KNITRO.jl/C API for extraction+reinjection across separate " *
    "KN_new() instances -- checked via names(KNITRO, all=true); nothing hessian/state/restart-shaped " *
    "beyond KN_get_hessian_values (exact-Hessian-mode only, not applicable to this driver's SR1 " *
    "default). NOT checkpointed. A resumed run restarts its outer Hessian approximation from " *
    "KNITRO's own default initialization."

# ============================================================================
# CHECKPOINT STRUCTURE (Section 6)
# ============================================================================
struct D20Checkpoint
    schema::Int
    run_id::String
    label::String
    branch::Symbol            # :upper or :lower
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    g::Float64                 # gamma'_focal at checkpoint time
    zfree::Vector{Float64}      # reduced A-block free coords (length D^2-1) at checkpoint time
    logA_full::Matrix{Float64}  # D x D, full log(Aod_theta), gravity-feasible by construction
    dual_warm_start::Vector{Float64}    # copy of ctx.obj.x at checkpoint time
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any          # NamedTuple or nothing -- current best feasible point tracked so far
    n_eval::Int
    knitro_iter::Int
    wall_elapsed::Float64
    checkpoint_reason::Symbol   # :iteration | :new_best | :wall_interval | :stage_complete
    screen_counts::NamedTuple   # (pairwise=.., witness=.., winner=.., envelope=.., winning_range=.., safety_net=.., passed=..) cumulative at ckpt time -- loosely typed field, old 4-key checkpoints from before the fast-range-screen wiring still deserialize fine (never destructured on resume, report-only)
    # ---- fields ONLY for the resume-reproducibility acceptance test, not needed for optimization ----
    verify_Delta_dual::Float64
    verify_gravity_value::Float64
    verify_max_abs_moment_kkt_resid::Float64
    verify_moment_resid_norm::Float64
    solver_state_note::String
end

"Atomic-ish checkpoint write: serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint that a resume could load."
function save_checkpoint(path::AbstractString, ckpt::D20Checkpoint)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end
load_checkpoint(path::AbstractString) = deserialize(path)::D20Checkpoint

x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# ============================================================================
# Screened evaluation wrapper (Section 5, extended by the fast-range-screen
# integration): EVERY call in this driver goes through this, never raw
# evaluate_fullA_fast. Tracks per-stage rejection counts for the report.
#
# Now routes through evaluate_fullA_screened_ranged (fast_range_screen.jl) --
# the pre-winner envelope + fused winning-range + general safety-net screens
# are THE production path, not an opt-in alternative alongside the old one.
# When `rsc.envelope === nothing` (unsupported ctx -- see fast_range_screen.jl's
# EnvelopeUnsupportedContext), evaluate_fullA_screened_ranged itself falls
# back to the existing zero-winner-only screen_hard_winners automatically, so
# this wrapper needs no separate fallback branch.
# ============================================================================
mutable struct ScreenCounters
    pairwise::Int
    witness::Int
    winner::Int
    envelope::Int
    winning_range::Int
    safety_net::Int
    passed::Int
    rejections::Vector{NamedTuple}   # audit trail: (stage, o, d, n_eval)
end
ScreenCounters() = ScreenCounters(0, 0, 0, 0, 0, 0, 0, NamedTuple[])
as_namedtuple(sc::ScreenCounters) = (pairwise = sc.pairwise, witness = sc.witness, winner = sc.winner,
                                      envelope = sc.envelope, winning_range = sc.winning_range,
                                      safety_net = sc.safety_net, passed = sc.passed)

function screened_eval(xf::AbstractVector{Float64}, ctx, rsc::RangedScreenContext, sc::ScreenCounters,
        n_eval_ref::Ref{Int}; warm::Bool = true)
    result, screen_meta = evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = nothing, use_cache = false, warm = warm, tag = "",
        pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
    st = screen_meta.screen_status
    if st === :pairwise_certified_infeasible
        sc.pairwise += 1
        push!(sc.rejections, (stage = :pairwise, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :witness_certified_infeasible
        sc.witness += 1
        push!(sc.rejections, (stage = :witness, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :winner_scan_infeasible
        sc.winner += 1
        push!(sc.rejections, (stage = :winner, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :EXACT_INFEASIBLE_PREWINNER_ENVELOPE
        sc.envelope += 1
        push!(sc.rejections, (stage = :envelope, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :EXACT_INFEASIBLE_WINNING_RANGE
        sc.winning_range += 1
        push!(sc.rejections, (stage = :winning_range, o = screen_meta.worst_o, d = screen_meta.worst_d, n_eval = n_eval_ref[]))
    elseif st === :EXACT_INFEASIBLE_MOMENT_RANGE
        sc.safety_net += 1
        push!(sc.rejections, (stage = :safety_net, o = get(screen_meta, :certificate, nothing) === nothing ? 0 : screen_meta.certificate.origin,
                               d = get(screen_meta, :certificate, nothing) === nothing ? 0 : screen_meta.certificate.destination, n_eval = n_eval_ref[]))
    else
        sc.passed += 1
    end
    return result, screen_meta
end

# ============================================================================
# PROFILE stage: fixed g, minimize Delta_dual over the A-block only.
# Direct extension of c9_phase8_d20_pilot.jl::profile_minimize with
# screening (Section 5) + checkpointing (Section 6) wired in.
# ============================================================================
function run_profile_checkpointed(label::String, g_in::Float64, find_smallest_in::Bool, zfree_start_in::Vector{Float64};
        maxtime_real::Float64 = 900.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, delta_in::Float64 = 1.0, draw_seed_in::Int = 20260719,
        ckpt_dir::AbstractString, checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        logio::Union{Nothing,IO} = nothing)
    lp(xs...) = (println(xs...); logio !== nothing && (println(logio, xs...); flush(logio)); flush(stdout))

    mkpath(ckpt_dir)
    resumed = resume_from === nothing ? nothing : load_checkpoint(resume_from)
    g = g_in; find_smallest = find_smallest_in; zfree_start = copy(zfree_start_in)
    W = W_in; delta = delta_in; draw_seed = draw_seed_in
    bandwidth_cache = Dict{Int,Float64}()
    if resumed !== nothing
        g = resumed.g; find_smallest = resumed.find_smallest; zfree_start = copy(resumed.zfree)
        W = resumed.W; delta = resumed.delta; draw_seed = resumed.draw_seed
        bandwidth_cache = copy(resumed.bandwidth_cache)
        lp("[", label, "] RESUMING from ", resume_from, " (reason=", resumed.checkpoint_reason,
           " n_eval=", resumed.n_eval, " knitro_iter=", resumed.knitro_iter, " wall_elapsed=", resumed.wall_elapsed, "s)")
    end

    # REQUIRED for reproducibility across processes -- see this file's header comment.
    Random.seed!(draw_seed)
    ctx = d20_real_setup(W = W, δ = delta, find_smallest = find_smallest)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2; n = D2 - 1
    rsc = build_ranged_screen_context(ctx)
    lp("[", label, "] ctx built, D=", D, " W=", W, " draw_seed=", draw_seed,
       " screen_setup_wall=", ctx.screen_setup_wall,
       " envelope_screen_supported=", rsc.envelope !== nothing,
       rsc.envelope === nothing ? " (reason: $(rsc.unsupported_reason))" : "")

    if resumed !== nothing
        ctx.obj.x .= resumed.dual_warm_start
        r_verify, _ = evaluate_fullA_screened_ranged(x_free_from_w(vcat(g, zfree_start), pe), ctx, rsc;
            moment_representation = :compressed, cache = nothing, use_cache = false, warm = true,
            pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
        d_delta = abs(r_verify.Delta_dual - resumed.verify_Delta_dual)
        d_grav = abs(r_verify.gravity_value - resumed.verify_gravity_value)
        d_kkt = abs(r_verify.max_abs_moment_kkt_resid - resumed.verify_max_abs_moment_kkt_resid)
        d_mr = abs(norm(r_verify.moment_resid) - resumed.verify_moment_resid_norm)
        lp("[", label, "] RESUME VALIDATION at checkpoint's own point: ",
           "|ΔDelta_dual|=", d_delta, " |Δgravity_value|=", d_grav,
           " |Δmax_abs_moment_kkt_resid|=", d_kkt, " |Δ||moment_resid|||=", d_mr)
        lp("[", label, "]   original: Delta_dual=", resumed.verify_Delta_dual, " gravity=", resumed.verify_gravity_value)
        lp("[", label, "]   resumed:  Delta_dual=", r_verify.Delta_dual, " gravity=", r_verify.gravity_value)
    end

    sc = ScreenCounters()
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)

    # seed the compressed warm-start cache with a cold solve first (same fix c9_phase8_d20_pilot.jl
    # found necessary -- a fresh ctx's very first warm=true call has no prior state to warm-start
    # from and can spuriously report infeasible).
    r_seed, _ = screened_eval(x_free_from_w(vcat(g, zfree_start), pe), ctx, rsc, sc, n_eval; warm = false)
    lp("[", label, "] warm-cache seed (cold): inner_status=", r_seed.inner_status, " Delta=", r_seed.Delta_dual,
       " screen_status=", get(r_seed, :screen_status, :unknown))
    r_seed.inner_status in FEASIBLE_CODES || error("run_profile_checkpointed($label): start point not inner-feasible, cannot proceed")

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    KNITRO.KN_set_param_by_name(kc, "algorithm", 3)
    xIndices = KNITRO.KN_add_vars(kc, n)
    z_halfwidth = 30.0   # see c9_phase8_d20_pilot.jl's box-bounds root-cause comment
    KNITRO.KN_set_var_lobnds_all(kc, zfree_start .- z_halfwidth)
    KNITRO.KN_set_var_upbnds_all(kc, zfree_start .+ z_halfwidth)
    KNITRO.KN_set_var_primal_init_values_all(kc, zfree_start)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best = Ref{Union{Nothing,NamedTuple}}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy()
    policy.cache = bandwidth_cache
    trace = NamedTuple[]
    t_start = time()
    last_ckpt_wall = Ref(time())
    knitro_iter = Ref(resumed !== nothing ? resumed.knitro_iter : 0)
    run_id = "c10_prod_$(label)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64}, r::NamedTuple)
        zfree_now = w_current[2:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = D20Checkpoint(1, run_id, label, find_smallest ? :lower : :upper, find_smallest, delta, W, draw_seed,
            w_current[1], copy(zfree_now), logA_full, copy(ctx.obj.x), copy(policy.cache),
            best[], n_eval[], knitro_iter[], time() - t_start, reason, as_namedtuple(sc),
            r.Delta_dual, r.gravity_value, r.max_abs_moment_kkt_resid, norm(r.moment_resid),
            SOLVER_STATE_NOTE)
        save_checkpoint(joinpath(ckpt_dir, "$(label)_latest.jls"), ckpt)
        reason in (:new_best, :stage_complete) && save_checkpoint(joinpath(ckpt_dir, "$(label)_$(reason)_neval$(n_eval[]).jls"), ckpt)
        return ckpt
    end

    n_cold_retries = Ref(0); n_rejected = Ref(0)
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w, pe)
        r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true)
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            throw(DomainError(w[1], "run_profile_checkpointed($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting"))
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        is_new_best = best[] === nothing || Δ < best[].Delta_dual
        if is_new_best
            best[] = (zfree = copy(zfree), Delta_dual = Δ, gravity_value = r.gravity_value,
                      max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                      t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, Delta_dual = Δ, inner_status = r.inner_status,
                       delta_feasible = Δ <= ctx.δ + 1e-6))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s Delta=", Δ, " status=", r.inner_status,
               " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
        end
        if is_new_best
            do_checkpoint(:new_best, w, r)
        end
        if time() - last_ckpt_wall[] > checkpoint_interval_s
            do_checkpoint(:wall_interval, w, r)
            last_ckpt_wall[] = time()
        end
        return 0
    end
    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            n_g_recompute[] += 1
            r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
            end
            r_g.inner_status in FEASIBLE_CODES || throw(DomainError(w[1], "run_profile_checkpointed($label): cb_G! could not recompute a feasible base state"))
            base = BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        invalidated, reason = maybe_invalidate!(policy, w)
        gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true,
                                                   h_mode = :cached, bandwidth_cache = policy.cache)
        meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        n_grad_calls[] += 1
        evalResult.objGrad .= gfull[2:end]
        return 0
    end
    # Section 6: checkpoint after every accepted KNITRO outer iterate. Cheap here (a warm
    # re-evaluation via the compressed path, plus a small serialize) relative to the gradient
    # call the SAME iterate already paid for.
    function cb_newpt!(kc2, x, lambda, user_data)
        knitro_iter[] += 1
        w_now = vcat(g, x)
        xf_now = x_free_from_w(w_now, pe)
        r_now, _ = screened_eval(xf_now, ctx, rsc, sc, n_eval; warm = true)
        if r_now.inner_status in FEASIBLE_CODES && isfinite(r_now.Delta_dual)
            do_checkpoint(:iteration, w_now, r_now)
        end
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)
    KNITRO.KN_set_newpt_callback(kc, cb_newpt!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best[]
    lp("[", label, "] PROFILE DONE: status=", nStatus_code, " wall_ext=", round(wall_ext, digits = 1),
       "s n_eval=", n_eval[], " n_grad_calls=", n_grad_calls[],
       " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
    if b !== nothing
        lp("  best: Delta=", b.Delta_dual, " gravity=", b.gravity_value, " found_at_eval=", b.n_eval)
    end
    # final stage-complete checkpoint at the terminal point
    w_final = vcat(g, collect(xsol))
    r_final, _ = screened_eval(x_free_from_w(w_final, pe), ctx, rsc, sc, n_eval; warm = true)
    final_ckpt = do_checkpoint(:stage_complete, w_final, r_final)

    return (label = label, g = g, find_smallest = find_smallest, ctx = ctx, pe = pe,
            knitro_status = nStatus_code, wall_ext = wall_ext, n_eval = n_eval[], n_grad_calls = n_grad_calls[],
            zfree_terminal = collect(xsol), best = b, trace = trace, screen_counts = as_namedtuple(sc),
            screen_rejections = sc.rejections, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end

# ============================================================================
# POLISH stage: joint (gamma', A) constrained solve, warm-started from a
# profile's terminal A-block. Same screening+checkpointing discipline as
# run_profile_checkpointed above; direct extension of
# c9_phase8_d20_pilot.jl::joint_polish.
# ============================================================================
function run_polish_checkpointed(label::String, find_smallest_in::Bool, g_start_in::Float64, zfree_start_in::Vector{Float64};
        maxtime_real::Float64 = 450.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, delta_in::Float64 = 1.0, draw_seed_in::Int = 20260719,
        ckpt_dir::AbstractString, checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        logio::Union{Nothing,IO} = nothing,
        inner_opt_override::Union{Nothing,AbstractString} = nothing,
        skip_cold_retry::Bool = true)   # validated 2026-07-20: cold retry rescued 0/30 warm failures (all genuine
    # primal infeasibility, confirmed by clean KNITRO -300/unbounded status on both attempts) while costing an
    # extra ~13.5s per rejected point; skipping it is a pure win (identical kappa reached in matched A/B tests,
    # ~1.4x more outer attempts explored per unit time). See docs handoff for the full investigation.
    lp(xs...) = (println(xs...); logio !== nothing && (println(logio, xs...); flush(logio)); flush(stdout))

    mkpath(ckpt_dir)
    resumed = resume_from === nothing ? nothing : load_checkpoint(resume_from)
    find_smallest = find_smallest_in; g_start = g_start_in; zfree_start = copy(zfree_start_in)
    W = W_in; delta = delta_in; draw_seed = draw_seed_in
    bandwidth_cache = Dict{Int,Float64}()
    if resumed !== nothing
        find_smallest = resumed.find_smallest; g_start = resumed.g; zfree_start = copy(resumed.zfree)
        W = resumed.W; delta = resumed.delta; draw_seed = resumed.draw_seed
        bandwidth_cache = copy(resumed.bandwidth_cache)
        lp("[", label, "] RESUMING from ", resume_from, " (reason=", resumed.checkpoint_reason,
           " n_eval=", resumed.n_eval, " knitro_iter=", resumed.knitro_iter, ")")
    end

    Random.seed!(draw_seed)   # see file header -- required for cross-process reproducibility
    ctx = inner_opt_override === nothing ? d20_real_setup(W = W, δ = delta, find_smallest = find_smallest) :
                                            d20_real_setup(W = W, δ = delta, find_smallest = find_smallest, inner_loop_opt = inner_opt_override)
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2
    rsc = build_ranged_screen_context(ctx)
    lp("[", label, "] ctx built, D=", D, " W=", W, " draw_seed=", draw_seed, " screen_setup_wall=", ctx.screen_setup_wall,
       " inner_opt=", inner_opt_override === nothing ? "default" : inner_opt_override,
       " envelope_screen_supported=", rsc.envelope !== nothing,
       rsc.envelope === nothing ? " (reason: $(rsc.unsupported_reason))" : "")

    w0 = vcat(g_start, zfree_start)
    if resumed !== nothing
        ctx.obj.x .= resumed.dual_warm_start
        r_verify, _ = evaluate_fullA_screened_ranged(x_free_from_w(w0, pe), ctx, rsc; moment_representation = :compressed,
            cache = nothing, use_cache = false, warm = true, pairwise = ctx.pairwise, witness = ctx.witness,
            use_witness = ctx.witness !== nothing)
        lp("[", label, "] RESUME VALIDATION: |ΔDelta_dual|=", abs(r_verify.Delta_dual - resumed.verify_Delta_dual),
           " |Δgravity_value|=", abs(r_verify.gravity_value - resumed.verify_gravity_value),
           " |Δmax_abs_moment_kkt_resid|=", abs(r_verify.max_abs_moment_kkt_resid - resumed.verify_max_abs_moment_kkt_resid),
           " |Δ||moment_resid|||=", abs(norm(r_verify.moment_resid) - resumed.verify_moment_resid_norm))
    end

    sc = ScreenCounters()
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    r0, _ = screened_eval(x_free_from_w(w0, pe), ctx, rsc, sc, n_eval; warm = false)
    lp("[", label, "] polish start point: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)
    r0.inner_status in FEASIBLE_CODES || error("run_polish_checkpointed($label): start point not inner-feasible, cannot proceed")

    z_halfwidth = 30.0
    w_lo = vcat(ctx.bounds.γp_lo, zfree_start .- z_halfwidth)
    w_hi = vcat(ctx.bounds.γp_hi, zfree_start .+ z_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy(); policy.cache = bandwidth_cache
    trace = NamedTuple[]
    t_start = time(); last_ckpt_wall = Ref(time())
    knitro_iter = Ref(resumed !== nothing ? resumed.knitro_iter : 0)
    run_id = "c10_prod_$(label)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64}, r::NamedTuple)
        zfree_now = w_current[2:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = D20Checkpoint(1, run_id, label, find_smallest ? :lower : :upper, find_smallest, delta, W, draw_seed,
            w_current[1], copy(zfree_now), logA_full, copy(ctx.obj.x), copy(policy.cache),
            best_feasible[], n_eval[], knitro_iter[], time() - t_start, reason, as_namedtuple(sc),
            r.Delta_dual, r.gravity_value, r.max_abs_moment_kkt_resid, norm(r.moment_resid), SOLVER_STATE_NOTE)
        save_checkpoint(joinpath(ckpt_dir, "$(label)_latest.jls"), ckpt)
        reason in (:new_best, :stage_complete) && save_checkpoint(joinpath(ckpt_dir, "$(label)_$(reason)_neval$(n_eval[]).jls"), ckpt)
        return ckpt
    end

    n_cold_retries = Ref(0); n_rejected = Ref(0)
    warm_cold_trace = NamedTuple[]   # additive diagnostic: filled only on warm-attempt failure (rare), read by caller after KN_solve
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        t_warm0 = time()
        r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true)
        t_warm = time() - t_warm0
        cold_time = NaN; cold_status = missing
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            warm_status = r.inner_status
            if !skip_cold_retry
                t_cold0 = time()
                r, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
                cold_time = time() - t_cold0
                cold_status = r.inner_status
            end
            push!(warm_cold_trace, (n_eval = n_eval[], gp = w[1], warm_time = t_warm, warm_status = warm_status,
                                     cold_time = cold_time, cold_status = cold_status,
                                     rescued = !skip_cold_retry && r.inner_status in FEASIBLE_CODES && isfinite(r.Delta_dual)))
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            throw(DomainError(w[1], "run_polish_checkpointed($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting"))
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        feasible = Δ <= ctx.δ + 1e-6
        base = BaseDualState(collect(xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
        is_new_best = feasible && (best_feasible[] === nothing || (find_smallest ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, gravity = r.gravity_value,
                                kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                                t_elapsed = t_el, n_eval = n_eval[])
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = w[1], Delta_dual = Δ, inner_status = r.inner_status, feasible = feasible))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s gp=", w[1], " Delta=", Δ,
               " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
        end
        if is_new_best
            do_checkpoint(:new_best, w, r)
        end
        if time() - last_ckpt_wall[] > checkpoint_interval_s
            do_checkpoint(:wall_interval, w, r)
            last_ckpt_wall[] = time()
        end
        return 0
    end
    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w, pe)
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        if base === nothing
            n_g_recompute[] += 1
            r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
            end
            r_g.inner_status in FEASIBLE_CODES || throw(DomainError(w[1], "run_polish_checkpointed($label): cb_G! could not recompute a feasible base state"))
            base = BaseDualState(collect(xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end
        invalidated, reason = maybe_invalidate!(policy, w)
        gfull, meta = composite_gradient_at_fast_buffered(xf, ctx, pe; base = base, threaded = true,
                                                   h_mode = :cached, bandwidth_cache = policy.cache)
        meta.tie_fallback || record_hits!(policy, meta.cache_hits[2:end])
        n_grad_calls[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        evalResult.jac .= gfull
        return 0
    end
    function cb_newpt!(kc2, x, lambda, user_data)
        knitro_iter[] += 1
        xf_now = x_free_from_w(x, pe)
        r_now, _ = screened_eval(xf_now, ctx, rsc, sc, n_eval; warm = true)
        if r_now.inner_status in FEASIBLE_CODES && isfinite(r_now.Delta_dual)
            do_checkpoint(:iteration, collect(x), r_now)
        end
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)
    KNITRO.KN_set_newpt_callback(kc, cb_newpt!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    σ = ctx.σ
    κ = NaN
    if b !== nothing
        κ = 1 - b.gp^(σ / (σ - 1))
    end
    lp("[", label, "] POLISH DONE: status=", nStatus_code, " wall_ext=", round(wall_ext, digits = 1),
       "s n_eval=", n_eval[], " kappa=", κ, " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed,
       " n_cold_retries(F)=", n_cold_retries[], " n_rejected(F)=", n_rejected[], " n_g_recompute=", n_g_recompute[])

    w_final = collect(xsol)
    r_final, _ = screened_eval(x_free_from_w(w_final, pe), ctx, rsc, sc, n_eval; warm = true)
    final_ckpt = do_checkpoint(:stage_complete, w_final, r_final)

    return (label = label, find_smallest = find_smallest, ctx = ctx, pe = pe,
            knitro_status = nStatus_code, wall_ext = wall_ext, n_eval = n_eval[], n_grad_calls = n_grad_calls[],
            best_feasible = b, kappa = κ, trace = trace, screen_counts = as_namedtuple(sc),
            screen_rejections = sc.rejections, final_checkpoint = final_ckpt,
            n_cold_retries = n_cold_retries[], n_rejected = n_rejected[], n_g_recompute = n_g_recompute[],
            warm_cold_trace = warm_cold_trace,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end
