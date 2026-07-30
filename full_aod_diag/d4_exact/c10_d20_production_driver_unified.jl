# ============================================================================
# Unified outer-coordinate production driver (addendum, 2026-07-25). ONE driver function,
# `run_polish_checkpointed_unified`, covering every (trade_elasticity_mode, A_coordinate_mode,
# gp_coordinate_mode) combination via `OuterCoordinateLayout` (outer_coordinate_layout.jl) --
# supersedes the original port's SEPARATE flexible-only driver
# (c10_d20_production_driver_flexible_theta_A.jl) per the addendum's explicit "do not finalize
# production with separate, near-duplicate fixed and flexible outer drivers" instruction.
#
# Reuses, UNCHANGED: screens (screened_eval/evaluate_fullA_screened_ranged), DualBank,
# SafeExactCache, SafeNegativeCache, incumbent/checkpoint discipline pattern, organic-failure
# capture, cold-retry policy, the C+ gradient kernel (composite_gradient_at_Cplus) -- exactly as
# both prior drivers already did. The layout abstraction supplies: decode (decode_outer_unified),
# gradient rescale (gradient_transform_unified), and checkpoint-field construction generic across
# every mode. Theta's OWN derivative (fixed-dual secant) is only computed when
# layout.trade_elasticity_mode==:flexible -- fixed-mode calls skip that machinery entirely (no
# freeze_theta_ctx() needed either: a fixed-mode ctx already has the shape composite_gradient_
# at_Cplus expects, since mu was never moved into free_idx).
# ============================================================================

isdefined(Main, :run_polish_checkpointed) || error("c10_d20_production_driver_unified.jl requires c10_d20_production_driver.jl to already be included.")
isdefined(Main, :make_layout) || error("c10_d20_production_driver_unified.jl requires outer_coordinate_layout.jl to already be included.")
isdefined(Main, :make_flexible_theta) || error("c10_d20_production_driver_unified.jl requires flexible_theta.jl (freeze_theta_ctx) to already be included.")
isdefined(Main, :theta_fixed_dual_delta_pivot_A) || error("c10_d20_production_driver_unified.jl requires flexible_theta_aspace_production.jl to already be included.")
isdefined(Main, :print_production_backend_manifest) || error("c10_d20_production_driver_unified.jl requires production_backend_manifest.jl to already be included.")
isdefined(Main, :set_production_outer_algorithm!) || error("c10_d20_production_driver_unified.jl requires knitro_outer_algorithm.jl to already be included.")
isdefined(Main, :theta_cplus_secant) || include(joinpath(@__DIR__, "theta_cplus.jl"))
isdefined(Main, :prepare_production_run) || include(joinpath(@__DIR__, "production_bundle_api.jl"))   # architecture/production-operator-bundle-hardening-2026-07-30

const CHECKPOINT_SCHEMA_UNIFIED = 1

"""
    D20CheckpointUnified

Single checkpoint schema for every (trade_elasticity_mode, A_coordinate_mode,
gp_coordinate_mode) combination -- addendum §4: "fixed-theta checkpoints may continue storing
the canonical economic/legacy z-space representation, provided they also store: active search
coordinate mode; theta; a-space fingerprint; exact mapping version." `zfree` here ALWAYS holds
genuine z-space coordinates (universal/theta-consistent, recoverable regardless of which
coordinate the outer search actually used) -- exactly like `D20CheckpointV4`/`D20CheckpointFlexA`
before it. `A_native` holds whatever the outer vector's OWN coordinate was (== `zfree` under
:legacy_z, `a_nonpivot` under :powered_aspace) so a resume reconstructs the identical search
point without re-deriving anything. `gp_native` similarly holds the raw OR scaled-log gp
coordinate actually searched (== `g` under :raw).
"""
struct D20CheckpointUnified
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    g::Float64                        # raw gp (always -- canonical/legacy representation)
    zfree::Vector{Float64}            # genuine z-space nonpivot coords (always -- canonical)
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    knitro_iter::Int
    wall_elapsed::Float64
    checkpoint_reason::Symbol
    screen_counts::NamedTuple
    verify_Delta_dual::Float64
    verify_gravity_value::Float64
    verify_max_abs_moment_kkt_resid::Float64
    verify_moment_resid_norm::Float64
    solver_state_note::String
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    knitro_version::String
    destination_sample::Symbol
    row_idx::Union{Nothing,Int}
    D_dest::Int
    # ---- addendum-unified fields ----
    trade_elasticity_mode::Symbol
    A_coordinate_mode::Symbol
    gp_coordinate_mode::Symbol
    amap_version::Int                 # AMAP_VERSION at write time (exact mapping version, addendum §4)
    eta_theta::Float64                # log(theta) always (== log(ctx theta_star) in fixed mode)
    theta::Float64
    theta_lo::Float64                 # == theta == theta_hi in fixed mode (degenerate box, documented)
    theta_hi::Float64
    gp_native::Float64                # the outer-searched gp coordinate (raw gp, or u_g if scaled_log)
    A_native::Vector{Float64}         # the outer-searched A-block coordinate (z_nonpivot or a_nonpivot)
    gp_star::Float64                  # GpScale.gp_star if gp_coordinate_mode==:scaled_log, else NaN
    s_g::Float64                      # GpScale.s_g if gp_coordinate_mode==:scaled_log, else NaN
end

function save_checkpoint_unified(path::AbstractString, ckpt::D20CheckpointUnified)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

function load_checkpoint_unified(path::AbstractString)::D20CheckpointUnified
    ckpt = deserialize(path)
    ckpt isa D20CheckpointUnified || error("load_checkpoint_unified($path): file does not contain a D20CheckpointUnified (got $(typeof(ckpt))).")
    ckpt.schema == CHECKPOINT_SCHEMA_UNIFIED || error("load_checkpoint_unified($path): schema=$(ckpt.schema), expected $(CHECKPOINT_SCHEMA_UNIFIED).")
    ckpt.amap_version == AMAP_VERSION || error("load_checkpoint_unified($path): amap_version=$(ckpt.amap_version), expected $(AMAP_VERSION) -- the a<->z mapping formula may have changed, refusing to resume.")
    return ckpt
end

"""
    build_unified_ctx(layout, ctx_base; theta_lo=nothing, theta_hi=nothing) -> ctx

Builds the appropriately-shaped ctx for `layout`: `ctx_base` unchanged for :fixed (theta stays
at ctx_base's own calibrated value), or `make_flexible_theta(ctx_base; theta_lo, theta_hi)` for
:flexible (theta_lo/theta_hi required).
"""
function build_unified_ctx(layout::OuterCoordinateLayout, ctx_base; theta_lo::Union{Nothing,Float64} = nothing, theta_hi::Union{Nothing,Float64} = nothing)
    if layout.trade_elasticity_mode == :fixed
        return ctx_base
    else
        theta_lo === nothing || theta_hi === nothing && error("build_unified_ctx: :flexible requires theta_lo and theta_hi")
        return make_flexible_theta(ctx_base; theta_lo = theta_lo, theta_hi = theta_hi, A_coordinate_mode = layout.A_coordinate_mode)
    end
end

"""
    run_polish_checkpointed_unified(label, find_smallest, w_start; layout, kwargs...)

The ONE production driver for every outer-coordinate combination. See file header. `ctx` passed
in must already be shaped per `layout` (fixed ctx for :fixed, `make_flexible_theta`'d for
:flexible -- see `build_unified_ctx`).
"""
function run_polish_checkpointed_unified(label::String, find_smallest_in::Bool, w_start_in::Vector{Float64};
        layout::OuterCoordinateLayout,
        theta_lo::Float64 = NaN, theta_hi::Float64 = NaN,   # required iff layout.trade_elasticity_mode==:flexible
        gp_scale::Union{Nothing,GpScale} = nothing,          # required iff layout.gp_coordinate_mode==:scaled_log
        maxtime_real::Float64 = 600.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, delta_in::Float64 = 1.0, draw_seed_in::Int = 20260719,
        draw_design_in::Union{Nothing,Symbol} = nothing,
        ckpt_dir::AbstractString, checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        logio::Union{Nothing,IO} = nothing,
        use_dual_bank::Bool = true, dual_bank_size::Int = 8, use_exact_cache::Bool = true,
        exact_cache_override::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,
        organic_failures::Union{Nothing,OrganicFailureCollector} = nothing,
        price_cache_backend::Union{Nothing,Symbol} = :cplus,
        maxit_override::Union{Nothing,Int} = nothing,
        h_theta::Float64 = 1e-3, a_halfwidth::Float64 = 30.0,
        skip_cold_retry::Bool = true,
        use_neg_cache::Bool = false, neg_cache_code_version::String = "unified_v1",
        destination_sample::Symbol = :exclude_row,
        blas_threads::Union{Nothing,Int} = nothing,   # reconciliation (task §1/Phase 1): same
        # kwarg/semantics as run_polish_checkpointed's -- process-scoped BLAS thread count, set
        # once right after ctx build, nothing (default) leaves the ambient count untouched.
        pin_outer_algorithm::Bool = false,   # reconciliation: same kwarg/semantics as
        # run_polish_checkpointed's -- opt-in explicit algorithm=2(Interior/CG)+hessopt=6(L-BFGS)
        # via knitro_outer_algorithm.jl, for matched benchmark A/Bs only.
        )   # architecture/production-operator-bundle-hardening-2026-07-30: the moment_representation
        # kwarg that previously lived here is REMOVED, not defaulted -- production runners must not
        # accept a representation choice at all (task §2). This function now always constructs
        # OperatorPsiBundle via prepare_production_run below. A dense reference bundle is available
        # only through DenseReferenceDiagnostics.prepare_context, never from this driver.
    lp(xs...) = (println(xs...); logio !== nothing && (println(logio, xs...); flush(logio)); flush(stdout))
    destination_sample in (:exclude_row, :all_legacy) ||
        error("run_polish_checkpointed_unified($label): destination_sample must be :exclude_row or :all_legacy.")
    layout.trade_elasticity_mode == :flexible && (isnan(theta_lo) || isnan(theta_hi)) &&
        error("run_polish_checkpointed_unified($label): :flexible requires theta_lo/theta_hi")
    layout.gp_coordinate_mode == :scaled_log && gp_scale === nothing &&
        error("run_polish_checkpointed_unified($label): :scaled_log requires gp_scale")

    mkpath(ckpt_dir)
    resumed = resume_from === nothing ? nothing : load_checkpoint_unified(resume_from)

    w0 = copy(w_start_in)
    W = W_in; delta = delta_in; draw_seed = draw_seed_in
    draw_design = draw_design_in === nothing ? :pseudorandom : draw_design_in
    find_smallest = find_smallest_in
    bandwidth_cache = Dict{Int,Float64}()
    if resumed !== nothing
        resumed.trade_elasticity_mode == layout.trade_elasticity_mode &&
            resumed.A_coordinate_mode == layout.A_coordinate_mode &&
            resumed.gp_coordinate_mode == layout.gp_coordinate_mode ||
            error("run_polish_checkpointed_unified($label): resume LAYOUT mismatch -- checkpoint has " *
                  "($(resumed.trade_elasticity_mode),$(resumed.A_coordinate_mode),$(resumed.gp_coordinate_mode)), " *
                  "this call requests ($(layout.trade_elasticity_mode),$(layout.A_coordinate_mode),$(layout.gp_coordinate_mode)). Refusing to resume.")
        w0 = layout.trade_elasticity_mode == :flexible ? vcat(resumed.eta_theta, resumed.gp_native, resumed.A_native) : vcat(resumed.gp_native, resumed.A_native)
        find_smallest = resumed.find_smallest
        W = resumed.W; delta = resumed.delta; draw_seed = resumed.draw_seed
        if layout.trade_elasticity_mode == :flexible
            theta_lo == resumed.theta_lo && theta_hi == resumed.theta_hi ||
                error("run_polish_checkpointed_unified($label): resume theta-bounds mismatch.")
        end
        draw_design = resumed.draw_design
        bandwidth_cache = copy(resumed.bandwidth_cache)
        resumed.destination_sample == destination_sample ||
            error("run_polish_checkpointed_unified($label): destination_sample MISMATCH on resume.")
        lp("[", label, "] RESUMING (unified) from ", resume_from, " (reason=", resumed.checkpoint_reason, " n_eval=", resumed.n_eval, ")")
    end

    ctx_base = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest,
                                      draw_design = draw_design, draw_seed = draw_seed, destination_sample = destination_sample)
    # Reconciliation (task §1/Phase 1): the same three campaign-lifetime workspace attaches and
    # BLAS-thread pin that run_polish_checkpointed itself carries -- attached to ctx_base BEFORE
    # build_unified_ctx so a flexible-mode `merge(ctx, (...))` (flexible_theta.jl:make_flexible_
    # theta) inherits them unchanged (merge keeps every field not explicitly overridden).
    ctx_base = attach_compressed_factual_workspace(ctx_base, ctx_base.D, ctx_base.D_dest, W)
    ctx_base = attach_canonical_price_precompute_workspace(ctx_base)
    ctx_base = attach_hard_score_b_cache(ctx_base)
    blas_threads !== nothing && BLAS.set_num_threads(blas_threads)
    ctx = build_unified_ctx(layout, ctx_base; theta_lo = layout.trade_elasticity_mode == :flexible ? theta_lo : nothing,
                             theta_hi = layout.trade_elasticity_mode == :flexible ? theta_hi : nothing)
    # architecture/production-operator-bundle-hardening-2026-07-30 (task §4): the ONLY call in this
    # function that decides bundle representation -- hardcoded :operator, not a passthrough kwarg.
    # prepare_production_run wraps the result in a type-safe ProductionContext, derives the live
    # backend manifest, and fatally asserts the OperatorPsiBundle invariant before this driver does
    # anything else with ctx.
    prepared = prepare_production_run(:unrestricted, "run_polish_checkpointed_unified",
        () -> build_unrestricted_operator_ctx(ctx; moment_representation = :operator))
    ctx = prepared.ctx.inner
    xy = precompute_aspace_XY(ctx)
    D = ctx.D; Ddest = _flex_ddest(ctx)
    n_outer = outer_dim(layout, D, Ddest)
    rsc = build_ranged_screen_context(ctx)
    resolved_backend = price_cache_backend === nothing ? :cplus : price_cache_backend
    # Phase F remediation (production-audit continuation, 2026-07-26): this kwarg previously
    # accepted ANY Symbol but only ever had a live effect on whether lfix_c_ws gets built below --
    # cb_G! (further down this function) unconditionally calls composite_gradient_at_Cplus(...,
    # lfix_c_ws, ...), which requires ws::LFixFactorizedWorkspace (a concrete, non-nullable type).
    # Passing any value other than :cplus/nothing therefore set lfix_c_ws=nothing and would have
    # crashed with a confusing MethodError deep inside cb_G! mid-solve, not a clear error at
    # call time -- confirmed live in the baseline static audit (this driver, unlike the older
    # c10_d20_production_driver.jl, never ported the :pooled/:aplus/:kbplus alternative gradient
    # backends; :cplus is the only gradient kernel this unified driver's own header describes:
    # "Reuses ... the C+ gradient kernel (composite_gradient_at_Cplus) ... exactly as both prior
    # drivers already did"). Fail fast with an actionable message instead of a silent no-op that
    # only surfaces as a crash later.
    resolved_backend == :cplus ||
        error("run_polish_checkpointed_unified: price_cache_backend=:$resolved_backend is not " *
              "supported -- this driver's cb_G! only ever calls the C+ gradient kernel " *
              "(composite_gradient_at_Cplus); pass :cplus or nothing (default).")
    grad_pool = build_grad_workspace_pool(W)
    lfix_c_ws = resolved_backend == :cplus ? build_lfix_factorized_workspace(D, Ddest, W) : nothing
    theta_ws = layout.trade_elasticity_mode == :flexible ? build_theta_cplus_workspace(D, Ddest, W; h_theta = h_theta) : nothing

    print_production_backend_manifest(resolve_unrestricted_manifest(; blas_threads = blas_threads,   # 2026-07-25 continuation: read the LIVE UNRESTRICTED_CORE_HESSIAN_BACKEND[] default (matching run_profile_checkpointed/run_polish_checkpointed's own fix, both of which stopped hardcoding :dense_exact for the same reason: it was silently misreporting the shared winner-pair backend)
        bundle_type = Symbol(nameof(typeof(ctx.obj))),   # unrestricted operator-bundle wiring task (2026-07-29): the REAL type, read off ctx AFTER build_unrestricted_operator_ctx above -- never an asserted literal
        trade_elasticity_mode = layout.trade_elasticity_mode, A_coordinate_mode = layout.A_coordinate_mode,
        gp_coordinate_mode = layout.gp_coordinate_mode,
        theta_bounds = layout.trade_elasticity_mode == :flexible ? (theta_lo, theta_hi) : nothing,
        outer_dimension = n_outer))
    write_backend_manifest_atomic(prepared.manifest, joinpath(ckpt_dir, "$(label)_backend_manifest.json"))   # architecture/production-operator-bundle-hardening-2026-07-30 (task §7): live manifest, replaces the static print above as the source of truth
    print_active_layout_banner(ctx, "unified_$(layout.trade_elasticity_mode)_$(layout.A_coordinate_mode)")
    flush(stdout)

    pgc = build_pivot_elimination_cheap(ctx;
        mu_probe1 = layout.trade_elasticity_mode == :flexible ? 1.0 / theta_lo : 1.0 / (hasproperty(ctx, :theta_star) ? ctx.theta_star : 1.0 / ctx.μHat) * 0.999,
        mu_probe2 = layout.trade_elasticity_mode == :flexible ? 1.0 / theta_hi : 1.0 / (hasproperty(ctx, :theta_star) ? ctx.theta_star : 1.0 / ctx.μHat) * 1.001)

    if resumed !== nothing
        d_resume = decode_outer_unified(w0, ctx, layout, pgc, xy, gp_scale)
        r_verify, _ = evaluate_fullA_screened_ranged(d_resume.xf, ctx, rsc; moment_representation = :compressed,
            cache = nothing, use_cache = false, warm = true, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
        d_delta = abs(r_verify.Delta_dual - resumed.verify_Delta_dual)
        check_resume_tolerances!(label, "run_polish_checkpointed_unified", d_delta,
            abs(r_verify.gravity_value - resumed.verify_gravity_value),
            abs(r_verify.max_abs_moment_kkt_resid - resumed.verify_max_abs_moment_kkt_resid),
            abs(norm(r_verify.benchmark_unweighted_moment_mean) - resumed.verify_moment_resid_norm))
        ctx.obj.x .= resumed.dual_warm_start
    end

    sc = ScreenCounters()
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    bank = use_dual_bank ? DualBank(dual_bank_size) : nothing
    exact_cache = exact_cache_override !== nothing ? exact_cache_override : (use_exact_cache ? SafeExactCache() : nothing)
    neg_cache = use_neg_cache ? SafeNegativeCache() : nothing

    d0 = decode_outer_unified(w0, ctx, layout, pgc, xy, gp_scale)
    r0, _ = screened_eval(d0.xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)
    lp("[", label, "] unified cold start: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual,
       " gravity=", r0.gravity_value, " theta0=", d0.theta, " gp0=", d0.gp)
    r0.inner_status in FEASIBLE_CODES || error("run_polish_checkpointed_unified($label): start point not inner-feasible")

    gp_dir_lo, gp_dir_hi = ctx_base.bounds.γp_lo, ctx_base.bounds.γp_hi
    gp_coord_lo = encode_gp(gp_dir_lo, layout, gp_scale); gp_coord_hi = encode_gp(gp_dir_hi, layout, gp_scale)
    gp_coord_lo, gp_coord_hi = min(gp_coord_lo, gp_coord_hi), max(gp_coord_lo, gp_coord_hi)
    if layout.trade_elasticity_mode == :flexible
        w_lo = vcat(log(theta_lo), gp_coord_lo, w0[3:end] .- a_halfwidth)
        w_hi = vcat(log(theta_hi), gp_coord_hi, w0[3:end] .+ a_halfwidth)
    else
        w_lo = vcat(gp_coord_lo, w0[2:end] .- a_halfwidth)
        w_hi = vcat(gp_coord_hi, w0[2:end] .+ a_halfwidth)
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    pin_outer_algorithm && set_production_outer_algorithm!(kc)   # opt-in only; default leaves the .opt file's algorithm=auto in effect
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    xIndices = KNITRO.KN_add_vars(kc, n_outer)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx_base.δ)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    seed_cand_feasible = r0.inner_status in FEASIBLE_CODES && isfinite(r0.Delta_dual) && r0.Delta_dual <= ctx_base.δ + 1e-6
    seed_cand = (gp = d0.gp, w = copy(w0), Delta = r0.Delta_dual, gravity = r0.gravity_value,
                 kkt = r0.max_abs_moment_kkt_resid, inner_status = r0.inner_status, t_elapsed = 0.0, n_eval = n_eval[])
    best_feasible = Ref{Any}(seed_incumbent(resumed !== nothing ? resumed.best_feasible : nothing, seed_cand_feasible, seed_cand))
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy(); policy.cache = bandwidth_cache
    trace = NamedTuple[]
    t_start = time(); last_ckpt_wall = Ref(time())
    knitro_iter = Ref(resumed !== nothing ? resumed.knitro_iter : 0)
    run_id = "c10_prod_unified_$(label)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    theta_total_wall = Ref(0.0); a_grad_wall = Ref(0.0)

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64}, r::NamedTuple, d::NamedTuple)
        ckpt = D20CheckpointUnified(CHECKPOINT_SCHEMA_UNIFIED, run_id, label, find_smallest ? :upper : :lower, find_smallest,
            delta, W, draw_seed, d.gp, copy(d.z_nonpivot), pivot_expand_cheap(d.z_nonpivot, pgc, d.mu),
            copy(ctx.obj.x), copy(policy.cache), best_feasible[], n_eval[], knitro_iter[], time() - t_start, reason,
            as_namedtuple(sc), r.Delta_dual, r.gravity_value, r.max_abs_moment_kkt_resid,
            norm(r.benchmark_unweighted_moment_mean), SOLVER_STATE_NOTE, draw_design,
            ctx_base.draw_meta.checksum_uniform, ctx_base.draw_meta.checksum_transformed, LOADED_KNITRO_RELEASE,
            destination_sample, ctx_base.row_idx, ctx_base.D_dest,
            layout.trade_elasticity_mode, layout.A_coordinate_mode, layout.gp_coordinate_mode, AMAP_VERSION,
            d.eta_theta, d.theta,
            layout.trade_elasticity_mode == :flexible ? theta_lo : d.theta,
            layout.trade_elasticity_mode == :flexible ? theta_hi : d.theta,
            w_current[layout.trade_elasticity_mode == :flexible ? 2 : 1], copy(d.A_nonpivot_native),
            gp_scale === nothing ? NaN : gp_scale.gp_star, gp_scale === nothing ? NaN : gp_scale.s_g)
        latest_path = joinpath(ckpt_dir, "$(label)_unified_latest.jls")
        save_checkpoint_unified(latest_path, ckpt)
        reason in (:new_best, :stage_complete, :stage_complete_unverified) &&
            save_checkpoint_unified(joinpath(ckpt_dir, "$(label)_unified_$(reason)_neval$(n_eval[]).jls"), ckpt)
        return ckpt
    end

    n_cold_retries = Ref(0); n_rejected = Ref(0)
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        d = decode_outer_unified(w, ctx, layout, pgc, xy, gp_scale)
        r, _ = screened_eval(d.xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = dual_bank_zfree(d, layout), exact_cache = exact_cache, neg_cache = neg_cache)
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            if !skip_cold_retry
                r, _ = screened_eval(d.xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)
            end
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            reject_point(w[1], "run_polish_checkpointed_unified($label): infeasible/non-finite point, rejecting")
        end
        Δ = r.Delta_dual
        gp_idx = layout.trade_elasticity_mode == :flexible ? 2 : 1
        evalResult.obj[1] = find_smallest ? w[gp_idx] : -w[gp_idx]
        evalResult.c[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        feasible = Δ <= ctx_base.δ + 1e-6
        # Unrestricted operator-bundle wiring task (2026-07-29): solve_base_state (three_way_
        # derivatives.jl) unconditionally uses the fully-dense CS.inner_loop_internal + obj(...)
        # functor -- a genuine crash site for OperatorPsiBundle on any cache hit (confirmed live via
        # the real driver run this task added, test_unrestricted_operator_ctx_driver_wiring_2026-07-29.jl).
        # compressed_base_state (compressed_live.jl) is the already-existing, already-dispatch-aware
        # equivalent (goes through inner_loop_internal_compressed, which already handles both bundle
        # types) -- used ONLY for OperatorPsiBundle here so :dense_reference's cache-hit path stays
        # byte-identical to pre-port production (unchanged function, unchanged numerics).
        base = r.cache_hit ? (ctx.obj isa OperatorPsiBundle ? compressed_base_state(d.xf, ctx) : solve_base_state(d.xf, ctx)) :
            BaseDualState(collect(d.xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base, r = r, d = d)
        is_new_best = feasible && is_verified_success(r) &&
            is_better_polish(d.gp, best_feasible[] === nothing ? nothing : best_feasible[].gp, find_smallest)
        if is_new_best
            best_feasible[] = (gp = d.gp, w = copy(w), Delta = Δ, gravity = r.gravity_value,
                                kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                                t_elapsed = t_el, n_eval = n_eval[], theta = d.theta, eta_theta = d.eta_theta)
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = d.gp, theta = d.theta, Delta_dual = Δ, inner_status = r.inner_status, feasible = feasible))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s gp=", d.gp, " theta=", d.theta, " Delta=", Δ,
               " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
        end
        is_new_best && do_checkpoint(:new_best, w, r, d)
        if time() - last_ckpt_wall[] > checkpoint_interval_s
            do_checkpoint(:wall_interval, w, r, d)
            last_ckpt_wall[] = time()
        end
        return 0
    end

    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        d = shared !== nothing && shared.w == w ? shared.d : nothing
        if base === nothing || d === nothing
            n_g_recompute[] += 1
            d = decode_outer_unified(w, ctx, layout, pgc, xy, gp_scale)
            r_g, _ = screened_eval(d.xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = dual_bank_zfree(d, layout), exact_cache = exact_cache, neg_cache = neg_cache)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _ = screened_eval(d.xf, ctx, rsc, sc, n_eval; warm = false, exact_cache = exact_cache)
            end
            r_g.inner_status in FEASIBLE_CODES || reject_point(w[1], "run_polish_checkpointed_unified($label): cb_G! could not recompute a feasible base state")
            # See cb_F!'s identical fix above (unrestricted operator-bundle wiring task, 2026-07-29).
            base = r_g.cache_hit ? (ctx.obj isa OperatorPsiBundle ? compressed_base_state(d.xf, ctx) : solve_base_state(d.xf, ctx)) :
                BaseDualState(collect(d.xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end

        t_a0 = time()
        xf_reduced = vcat(d.gp, d.xf[(layout.trade_elasticity_mode == :flexible ? 3 : 2):end])
        if layout.trade_elasticity_mode == :flexible
            ctx_frozen = freeze_theta_ctx(ctx, d.mu)
            pe_here = pivot_elim_from_cache(pgc, d.mu)
            base_frozen = BaseDualState(xf_reduced, base.θ_full0, base.ζstar, base.λstar, base.m_star, base.inner_status)
            gfull_reduced_z, meta = composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws;
                base = base_frozen, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache)
        else
            pe_here = pivot_elim_from_cache(pgc, d.mu)
            gfull_reduced_z, meta = composite_gradient_at_Cplus(xf_reduced, ctx, pe_here, grad_pool, lfix_c_ws;
                base = base, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache)
        end
        record_hits!(policy, meta.cache_hits[2:end])
        gfull_reduced = gradient_transform_unified(gfull_reduced_z, d.theta, d.gp, layout, gp_scale)
        a_grad_wall[] += time() - t_a0

        if layout.trade_elasticity_mode == :flexible
            t_theta0 = time()
            # theta C+ fast fixed-dual secant (2026-07-26): replaces the old theta_fixed_dual_
            # delta_pivot_A brute-force path (generic obj.moments!/CS.reconstruct_full, ~1.4s/
            # ~743MB per probe) with the compressed winner-form pipeline (build_compressed_
            # factual!/compressed_cc_value_grad) -- an EXACT full re-scan of every origin at both
            # perturbed theta values (addendum: no stability-radius shortcut), routed through
            # already-existing/already-validated production machinery instead of the dense path.
            # No copy(w)/w_plus/w_minus (theta_cplus_secant reads gp/a_nonpivot via a view into
            # w), and no third base-point reconstruction: verified dead (see theta_cplus.jl header
            # -- ctx.obj.H gets unconditionally rebuilt by inner_loop_internal at the START of the
            # next real inner solve regardless, and hessian! -- obj.H's only reader -- never fires
            # at the outer-gradient-callback level).
            sec = theta_cplus_secant(w, ctx, xy, pgc, base, theta_ws)
            grad_eta_theta = sec.grad_eta_theta
            theta_total_wall[] += time() - t_theta0
            jac_full = vcat(grad_eta_theta, gfull_reduced)
            evalResult.objGrad .= 0.0; evalResult.objGrad[2] = find_smallest ? 1.0 : -1.0
        else
            jac_full = gfull_reduced
            evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        end
        n_grad_calls[] += 1
        evalResult.jac .= jac_full
        return 0
    end

    function cb_newpt!(kc2, x, lambda, user_data)
        knitro_iter[] += 1
        shared = last_F_state[]
        r_now = (shared !== nothing && shared.w == x) ? shared.r : nothing
        d_now = (shared !== nothing && shared.w == x) ? shared.d : nothing
        if r_now === nothing
            d_now = decode_outer_unified(collect(x), ctx, layout, pgc, xy, gp_scale)
            r_now, _ = screened_eval(d_now.xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = dual_bank_zfree(d_now, layout), exact_cache = exact_cache)
        end
        r_now.inner_status in FEASIBLE_CODES && isfinite(r_now.Delta_dual) && do_checkpoint(:iteration, collect(x), r_now, d_now)
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], n_outer), jacIndexVars = xIndices)
    KNITRO.KN_set_newpt_callback(kc, cb_newpt!)

    pin_outer_algorithm && assert_outer_algorithm_explicit!(kc; context = "run_polish_checkpointed_unified($label)")
    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    native_outer_diag = full_status_record(nStatus_code, kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    σ = ctx_base.σ
    κ = b !== nothing ? 1 - b.gp^(σ / (σ - 1)) : NaN
    lp("[", label, "] UNIFIED POLISH DONE: status=", nStatus_code, " (", native_outer_diag.status_name, "/", native_outer_diag.status_category,
       ") wall_ext=", round(wall_ext, digits = 1), "s n_eval=", n_eval[], " kappa=", κ, " n_grad_calls=", n_grad_calls[])
    if theta_ws !== nothing
        lp("[", label, "] theta_derivative_backend = cplus_fixed_dual_secant")
        lp("[", label, "] theta_generic_moments_calls = ", theta_ws.generic_moments_calls, " (must be 0)")
        lp("[", label, "] theta_secant_calls = ", theta_ws.n_calls, "  theta_winner_changes_total(plus_vs_minus) = ", theta_ws.n_winner_changes_total,
           "  theta_wall_total = ", round(theta_total_wall[], digits = 3), "s")
    end

    w_final = collect(xsol)
    d_final = decode_outer_unified(w_final, ctx, layout, pgc, xy, gp_scale)
    r_final, _ = screened_eval(d_final.xf, ctx, rsc, sc, n_eval; warm = true, bank = bank, zfree = dual_bank_zfree(d_final, layout), exact_cache = exact_cache)
    final_ckpt = do_checkpoint((!(r_final.inner_status in FEASIBLE_CODES) || !is_verified_success(r_final)) ? :stage_complete_unverified : :stage_complete, w_final, r_final, d_final)

    return (label = label, find_smallest = find_smallest, ctx = ctx, ctx_base = ctx_base, xy = xy, pgc = pgc, layout = layout,
            knitro_status = nStatus_code, native_outer_diag = native_outer_diag, wall_ext = wall_ext,
            n_eval = n_eval[], n_grad_calls = n_grad_calls[], best_feasible = b, kappa = κ, trace = trace,
            screen_counts = as_namedtuple(sc), final_checkpoint = final_ckpt,
            n_cold_retries = n_cold_retries[], n_rejected = n_rejected[], n_g_recompute = n_g_recompute[],
            theta_total_wall = theta_total_wall[], a_grad_wall = a_grad_wall[])
end
