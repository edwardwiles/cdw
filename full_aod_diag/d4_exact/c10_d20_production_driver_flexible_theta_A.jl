# ============================================================================
# Flexible-theta a-space (theta_decoupled_aspace) production D=20 driver, 2026-07-25 port.
#
# Direct extension of c10_d20_production_driver.jl's run_polish_checkpointed (the joint
# gp+A "polish" driver: minimize/maximize gp subject to Delta_dual<=delta) -- reuses its
# screens, DualBank, SafeExactCache, SafeNegativeCache, incumbent/checkpoint discipline,
# organic-failure capture, and cold-retry policy UNCHANGED. Per task §4/§8, ONLY THREE
# things differ from the fixed-theta driver:
#   1. the decode (decode_and_expand_flexible_A, flexible_theta_aspace_production.jl)
#   2. the A-block + gp gradient: the EXISTING production C+ gradient
#      (composite_gradient_at_Cplus), evaluated on a THETA-FROZEN ctx (freeze_theta_ctx,
#      flexible_theta.jl) at the current base point's mu, then rescaled by the exact scalar
#      dz/da = -theta for the A-block entries only (gp's own gradient is untouched by a).
#   3. the theta gradient: a central-difference secant on the fixed-dual objective
#      (theta_fixed_dual_delta_pivot_A), holding a_nonpivot fixed across the +-h probe --
#      NOT the existing per-A-cell incremental gradient machinery, which has no theta axis.
#
# Outer vector: w = [eta_theta; gp; a_nonpivot], length D*Ddest+1 (381 at real D=20
# post-omit-ROW: D=20, Ddest=19, D*Ddest-1=379 free A coords + eta_theta + gp = 381).
#
# See docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md and
# docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md.
# ============================================================================

isdefined(Main, :run_polish_checkpointed) || error("c10_d20_production_driver_flexible_theta_A.jl requires c10_d20_production_driver.jl to already be included.")
isdefined(Main, :decode_and_expand_flexible_A) || error("c10_d20_production_driver_flexible_theta_A.jl requires flexible_theta_aspace_production.jl to already be included.")
isdefined(Main, :freeze_theta_ctx) || error("c10_d20_production_driver_flexible_theta_A.jl requires flexible_theta.jl to already be included.")

const CHECKPOINT_SCHEMA_FLEX_A = 1
const A_COORDINATE_MODE_DEFAULT = :theta_decoupled_aspace

"""
    D20CheckpointFlexA

Flexible-theta a-space checkpoint. Carries `D20CheckpointV4`'s COMPLETE field set (schema 4,
unchanged, see c10_d20_production_driver.jl) PLUS `trade_elasticity_mode`, `A_coordinate_mode`,
`eta_theta`, `theta`, `theta_lo`, `theta_hi`, `a_nonpivot` -- new type name for the SAME reason
`D20Checkpoint`->`D20CheckpointV4` and the CM-family `CMCheckpointV*` series exist (adding fields
to an existing struct name breaks deserialization of already-written files under that name; see
checkpoint-schema-bump-collision-check memory). `zfree` here ALWAYS holds the genuine z-space
coordinates (`z_nonpivot = log(Aod_theta)_nonpivot`, universal/theta-consistent at the
checkpoint's own theta), exactly like `logA_full` -- the SAME convention the original z-space
port's `FlexibleThetaCheckpoint` used ("zfree is always stored in genuine z-space"). `a_nonpivot`
is the ADDITIONAL a-space coordinate this mode searches over; a fixed-mode `D20CheckpointV4`
never has this field (Union{Nothing,...} would be a schema-ambiguity risk -- this struct is
ALWAYS a-space, distinguishable purely by type name, matching the CM-checkpoint convention of
one struct name per (mode, schema) pair).
"""
struct D20CheckpointFlexA
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    g::Float64
    zfree::Vector{Float64}            # genuine z-space (log Aod_theta) nonpivot coords
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
    # ---- flexible-theta a-space fields (this port, additive relative to D20CheckpointV4) ----
    trade_elasticity_mode::Symbol     # always :flexible
    A_coordinate_mode::Symbol         # always :theta_decoupled_aspace for this struct
    eta_theta::Float64
    theta::Float64
    theta_lo::Float64
    theta_hi::Float64
    a_nonpivot::Vector{Float64}
end

function save_checkpoint_flexA(path::AbstractString, ckpt::D20CheckpointFlexA)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"""
    load_checkpoint_flexA(path) -> D20CheckpointFlexA

Rejects loudly (task §11) on: wrong schema, wrong struct (a stray fixed-mode file), mode/coord
mismatch. Never silently proceeds past a mismatch.
"""
function load_checkpoint_flexA(path::AbstractString)::D20CheckpointFlexA
    ckpt = deserialize(path)
    ckpt isa D20CheckpointFlexA || error("load_checkpoint_flexA($path): file does not contain a D20CheckpointFlexA (got $(typeof(ckpt))) -- refusing to load a fixed-mode or foreign checkpoint through this flexible-theta a-space loader.")
    ckpt.schema == CHECKPOINT_SCHEMA_FLEX_A || error("load_checkpoint_flexA($path): schema=$(ckpt.schema), expected $(CHECKPOINT_SCHEMA_FLEX_A). Start a fresh run.")
    ckpt.trade_elasticity_mode == :flexible || error("load_checkpoint_flexA($path): trade_elasticity_mode=$(ckpt.trade_elasticity_mode), expected :flexible.")
    ckpt.A_coordinate_mode == :theta_decoupled_aspace || error("load_checkpoint_flexA($path): A_coordinate_mode=$(ckpt.A_coordinate_mode), expected :theta_decoupled_aspace.")
    return ckpt
end

"""
    run_polish_checkpointed_flexible_theta_A(label, find_smallest_in, w_ext_a_start_in; theta_lo, theta_hi, kwargs...)

Near-duplicate of `run_polish_checkpointed` (matching this codebase's own established
near-duplicate-rather-than-deep-parameterize convention -- see docs/FLEXIBLE_THETA_ASPACE_
PRODUCTION_PORT_2026-07-25.md for why a shared-core refactor was judged higher-risk than this
for a first production port). Reuses screens/DualBank/SafeExactCache/SafeNegativeCache/
organic-failure-capture/cold-retry exactly as run_polish_checkpointed does; only decode,
gradient, and checkpoint schema differ (see file header).
"""
function run_polish_checkpointed_flexible_theta_A(label::String, find_smallest_in::Bool, w_ext_a_start_in::Vector{Float64};
        theta_lo::Float64, theta_hi::Float64,
        maxtime_real::Float64 = 600.0, hessopt_tag::String = "sr1",
        W_in::Int = 80000, delta_in::Float64 = 1.0, draw_seed_in::Int = 20260719,
        draw_design_in::Union{Nothing,Symbol} = nothing,
        ckpt_dir::AbstractString, checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        logio::Union{Nothing,IO} = nothing,
        use_dual_bank::Bool = true, dual_bank_size::Int = 8, use_exact_cache::Bool = true,
        exact_cache_override::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,
        organic_failures::Union{Nothing,OrganicFailureCollector} = nothing,
        price_cache_backend::Union{Nothing,Symbol} = :cplus,   # task §8: production default IS C+ for flexible-theta A-block gradients
        maxit_override::Union{Nothing,Int} = nothing,
        h_theta::Float64 = 1e-3,           # theta-secant central-difference step (task §9)
        a_halfwidth::Float64 = 30.0,       # a-space box half-width around the starting a_nonpivot
        skip_cold_retry::Bool = true,
        use_neg_cache::Bool = false, neg_cache_code_version::String = "flexA_v1",
        destination_sample::Symbol = :exclude_row)
    lp(xs...) = (println(xs...); logio !== nothing && (println(logio, xs...); flush(logio)); flush(stdout))
    destination_sample in (:exclude_row, :all_legacy) ||
        error("run_polish_checkpointed_flexible_theta_A($label): destination_sample=:$destination_sample requested, must be :exclude_row or :all_legacy.")

    mkpath(ckpt_dir)
    resumed = resume_from === nothing ? nothing : load_checkpoint_flexA(resume_from)

    w0 = copy(w_ext_a_start_in)
    W = W_in; delta = delta_in; draw_seed = draw_seed_in
    draw_design = draw_design_in === nothing ? :pseudorandom : draw_design_in
    find_smallest = find_smallest_in
    bandwidth_cache = Dict{Int,Float64}()
    if resumed !== nothing
        w0 = vcat(resumed.eta_theta, resumed.g, resumed.a_nonpivot)
        find_smallest = resumed.find_smallest
        W = resumed.W; delta = resumed.delta; draw_seed = resumed.draw_seed
        theta_lo == resumed.theta_lo && theta_hi == resumed.theta_hi ||
            error("run_polish_checkpointed_flexible_theta_A($label): resume theta-bounds mismatch -- checkpoint has " *
                  "[$(resumed.theta_lo),$(resumed.theta_hi)], this call requests [$theta_lo,$theta_hi]. Refusing to resume.")
        if draw_design_in !== nothing && draw_design_in != resumed.draw_design
            error("run_polish_checkpointed_flexible_theta_A($label): resume draw_design mismatch -- checkpoint has :$(resumed.draw_design), caller requested :$(draw_design_in).")
        end
        draw_design = resumed.draw_design
        bandwidth_cache = copy(resumed.bandwidth_cache)
        resumed.destination_sample == destination_sample ||
            error("run_polish_checkpointed_flexible_theta_A($label): destination_sample MISMATCH on resume -- checkpoint :$(resumed.destination_sample), this call :$destination_sample.")
        lp("[", label, "] RESUMING (flexA) from ", resume_from, " (reason=", resumed.checkpoint_reason,
           " n_eval=", resumed.n_eval, " theta=", resumed.theta, ")")
    end

    ctx_base = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest,
                                      draw_design = draw_design, draw_seed = draw_seed, destination_sample = destination_sample)
    ctx = make_flexible_theta(ctx_base; theta_lo = theta_lo, theta_hi = theta_hi, A_coordinate_mode = A_COORDINATE_MODE_DEFAULT)
    xy = precompute_aspace_XY(ctx)
    D = ctx.D; Ddest = _flex_ddest(ctx); D2 = D * Ddest + 1   # +1 for eta_theta (gp+A alone would be D*Ddest+1 already; total outer dim below)
    n_outer = D * Ddest + 1   # eta_theta + gp + a_nonpivot(D*Ddest-1) = D*Ddest+1
    rsc = build_ranged_screen_context(ctx)
    resolved_backend = price_cache_backend === nothing ? :cplus : price_cache_backend
    resolved_backend == :cplus || lp("[", label, "] WARNING: price_cache_backend=:", resolved_backend,
        " requested for flexible-theta a-space -- task §8's validated/default backend is :cplus; other backends are wired generically but not part of this port's validated gate set.")
    grad_pool = build_grad_workspace_pool(W)
    lfix_c_ws = resolved_backend == :cplus ? build_lfix_factorized_workspace(D, Ddest, W) : nothing

    lp("[", label, "] ============================================================")
    lp("[", label, "] trade_elasticity_mode = flexible")
    lp("[", label, "] A_coordinate_mode = ", ctx.A_coordinate_mode)
    lp("[", label, "] theta_star = ", ctx.theta_star)
    lp("[", label, "] theta_bounds = [", theta_lo, ", ", theta_hi, "]")
    lp("[", label, "] outer_dimension = ", n_outer)
    lp("[", label, "] core_top1_engine = canonical_log_additive")
    lp("[", label, "] outer_gradient_top3_engine = ", resolved_backend)
    lp("[", label, "] ============================================================")
    lp("[", label, "] ctx built (flexA), D=", D, " Ddest=", Ddest, " W=", W, " draw_seed=", draw_seed,
       " draw_design=", draw_design, " envelope_screen_supported=", rsc.envelope !== nothing,
       rsc.envelope === nothing ? " (reason: $(rsc.unsupported_reason) -- expected in flexible mode, see docs/FLEXIBLE_THETA_RECTANGULAR_GRAVITY_AUDIT_2026-07-25.md)" : "")
    print_active_layout_banner(ctx, "unrestricted_flexible_theta_aspace")
    println("[screen-stack] mode=unrestricted_flexible_theta_aspace enabled=true")
    println("[screen-stack] ordered active screens: pairwise_certificate, screen_hard_winners, ",
            "winning_range, safety_net_moment_range (envelope DISABLED -- mu not fixed)")
    th = ctx.obj.threshold_state
    println("[threshold-config] mode=unrestricted_flexible_theta_aspace requested_delta=", delta_in,
            " resolved_active_threshold=", th.threshold)
    flush(stdout)

    if resumed !== nothing
        d_resume = decode_and_expand_flexible_A(w0, ctx, xy)
        r_verify, _ = evaluate_fullA_screened_ranged(d_resume.xf, ctx, rsc; moment_representation = :compressed,
            cache = nothing, use_cache = false, warm = true, pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
        d_delta = abs(r_verify.Delta_dual - resumed.verify_Delta_dual)
        d_grav = abs(r_verify.gravity_value - resumed.verify_gravity_value)
        d_kkt = abs(r_verify.max_abs_moment_kkt_resid - resumed.verify_max_abs_moment_kkt_resid)
        d_mr = abs(norm(r_verify.benchmark_unweighted_moment_mean) - resumed.verify_moment_resid_norm)
        lp("[", label, "] RESUME VALIDATION (flexA): |ΔDelta_dual|=", d_delta, " |Δgravity|=", d_grav, " |Δkkt|=", d_kkt, " |Δmoment_resid|=", d_mr)
        check_resume_tolerances!(label, "run_polish_checkpointed_flexible_theta_A", d_delta, d_grav, d_kkt, d_mr)
        ctx.obj.x .= resumed.dual_warm_start
    end

    sc = ScreenCounters()
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    bank = use_dual_bank ? DualBank(dual_bank_size) : nothing
    exact_cache = exact_cache_override !== nothing ? exact_cache_override : (use_exact_cache ? SafeExactCache() : nothing)
    neg_cache = use_neg_cache ? SafeNegativeCache() : nothing
    n_neg_confirmed = Ref(0)

    r0, meta0, d0 = screened_eval_flexible_A(w0, ctx, rsc, sc, n_eval, xy; warm = false, exact_cache = exact_cache)
    lp("[", label, "] flexA cold start: inner_status=", r0.inner_status, " Delta=", r0.Delta_dual,
       " gravity=", r0.gravity_value, " theta0=", d0.theta)
    r0.inner_status in FEASIBLE_CODES || error("run_polish_checkpointed_flexible_theta_A($label): start point not inner-feasible, cannot proceed")

    gp_dir_lo, gp_dir_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    eta_lo, eta_hi = log(theta_lo), log(theta_hi)
    w_lo = vcat(eta_lo, gp_dir_lo, w0[3:end] .- a_halfwidth)
    w_hi = vcat(eta_hi, gp_dir_hi, w0[3:end] .+ a_halfwidth)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
    xIndices = KNITRO.KN_add_vars(kc, n_outer)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_checkpoint_reuse_hits = Ref(0)
    seed_cand_feasible = r0.inner_status in FEASIBLE_CODES && isfinite(r0.Delta_dual) && r0.Delta_dual <= ctx.δ + 1e-6
    seed_cand = (gp = w0[2], w = copy(w0), Delta = r0.Delta_dual, gravity = r0.gravity_value,
                 kkt = r0.max_abs_moment_kkt_resid, inner_status = r0.inner_status, t_elapsed = 0.0, n_eval = n_eval[])
    best_feasible = Ref{Any}(seed_incumbent(resumed !== nothing ? resumed.best_feasible : nothing, seed_cand_feasible, seed_cand))
    n_grad_calls = Ref(0)
    policy = BandwidthCachePolicy(); policy.cache = bandwidth_cache
    trace = NamedTuple[]
    t_start = time(); last_ckpt_wall = Ref(time())
    knitro_iter = Ref(resumed !== nothing ? resumed.knitro_iter : 0)
    run_id = "c10_prod_flexA_$(label)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    theta_plus_wall = Ref(0.0); theta_minus_wall = Ref(0.0); theta_restore_wall = Ref(0.0)
    theta_total_wall = Ref(0.0); a_grad_wall = Ref(0.0); complete_grad_wall = Ref(0.0)

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64}, r::NamedTuple, d::NamedTuple)
        ckpt = D20CheckpointFlexA(CHECKPOINT_SCHEMA_FLEX_A, run_id, label, find_smallest ? :upper : :lower, find_smallest,
            delta, W, draw_seed, w_current[2], copy(d.z_nonpivot), pivot_expand_cheap(d.z_nonpivot, d.pgc, d.mu),
            copy(ctx.obj.x), copy(policy.cache), best_feasible[], n_eval[], knitro_iter[], time() - t_start, reason,
            as_namedtuple(sc), r.Delta_dual, r.gravity_value, r.max_abs_moment_kkt_resid,
            norm(r.benchmark_unweighted_moment_mean), SOLVER_STATE_NOTE, draw_design,
            ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed, LOADED_KNITRO_RELEASE,
            destination_sample, ctx.row_idx, ctx.D_dest,
            :flexible, ctx.A_coordinate_mode, w_current[1], d.theta, theta_lo, theta_hi, copy(d.a_nonpivot))
        latest_path = joinpath(ckpt_dir, "$(label)_flexA_latest.jls")
        save_checkpoint_flexA(latest_path, ckpt)
        reason in (:new_best, :stage_complete, :stage_complete_unverified) &&
            save_checkpoint_flexA(joinpath(ckpt_dir, "$(label)_flexA_$(reason)_neval$(n_eval[]).jls"), ckpt)
        return ckpt
    end

    n_cold_retries = Ref(0); n_rejected = Ref(0)
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        r, _, d = screened_eval_flexible_A(w, ctx, rsc, sc, n_eval, xy; warm = true, bank = bank, exact_cache = exact_cache, neg_cache = neg_cache)
        if !(r.inner_status in FEASIBLE_CODES)
            n_cold_retries[] += 1
            if !skip_cold_retry
                r, _, d = screened_eval_flexible_A(w, ctx, rsc, sc, n_eval, xy; warm = false, exact_cache = exact_cache)
            end
        end
        if !(r.inner_status in FEASIBLE_CODES) || !isfinite(r.Delta_dual)
            n_rejected[] += 1
            reject_point(w[1], "run_polish_checkpointed_flexible_theta_A($label): infeasible/non-finite point (inner_status=$(r.inner_status)), rejecting")
        end
        Δ = r.Delta_dual
        evalResult.obj[1] = find_smallest ? w[2] : -w[2]
        evalResult.c[1] = Δ
        n_eval[] += 1
        t_el = time() - t_start
        feasible = Δ <= ctx.δ + 1e-6
        base = r.cache_hit ? solve_base_state(d.xf, ctx) :
            BaseDualState(collect(d.xf), r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base, r = r, d = d)
        is_new_best = feasible && is_verified_success(r) &&
            is_better_polish(w[2], best_feasible[] === nothing ? nothing : best_feasible[].gp, find_smallest)
        if is_new_best
            best_feasible[] = (gp = w[2], w = copy(w), Delta = Δ, gravity = r.gravity_value,
                                kkt = r.max_abs_moment_kkt_resid, inner_status = r.inner_status,
                                t_elapsed = t_el, n_eval = n_eval[], theta = d.theta, eta_theta = d.eta_theta)
        end
        push!(trace, (idx = n_eval[], t_elapsed = t_el, gp = w[2], eta_theta = w[1], theta = d.theta,
                       Delta_dual = Δ, inner_status = r.inner_status, feasible = feasible))
        if n_eval[] <= 5 || n_eval[] % 10 == 0
            lp("  [", label, "] eval ", n_eval[], " t=", round(t_el, digits = 1), "s eta_theta=", w[1],
               " theta=", d.theta, " gp=", w[2], " Delta=", Δ,
               " screens(pw/wt/wn/env/wr/sn/pass)=", sc.pairwise, "/", sc.witness, "/", sc.winner, "/", sc.envelope, "/", sc.winning_range, "/", sc.safety_net, "/", sc.passed)
        end
        if is_new_best
            do_checkpoint(:new_best, w, r, d)
        end
        if time() - last_ckpt_wall[] > checkpoint_interval_s
            do_checkpoint(:wall_interval, w, r, d)
            last_ckpt_wall[] = time()
        end
        return 0
    end

    n_g_recompute = Ref(0)
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        t_cg0 = time()
        w = evalRequest.x
        shared = last_F_state[]
        base = shared !== nothing && shared.w == w ? shared.base : nothing
        d = shared !== nothing && shared.w == w ? shared.d : nothing
        if base === nothing || d === nothing
            n_g_recompute[] += 1
            r_g, _, d = screened_eval_flexible_A(w, ctx, rsc, sc, n_eval, xy; warm = true, bank = bank, exact_cache = exact_cache, neg_cache = neg_cache)
            if !(r_g.inner_status in FEASIBLE_CODES)
                r_g, _, d = screened_eval_flexible_A(w, ctx, rsc, sc, n_eval, xy; warm = false, exact_cache = exact_cache)
            end
            r_g.inner_status in FEASIBLE_CODES || reject_point(w[1], "run_polish_checkpointed_flexible_theta_A($label): cb_G! could not recompute a feasible base state")
            base = r_g.cache_hit ? solve_base_state(d.xf, ctx) :
                BaseDualState(collect(d.xf), r_g.θ_full, r_g.zeta, r_g.lambda, copy(ctx.obj.arg1), r_g.inner_status)
        end

        # ---- (1) A-block + gp gradient: existing production C+ kernel on a THETA-FROZEN ctx ----
        t_a0 = time()
        ctx_frozen = freeze_theta_ctx(ctx, d.mu)
        pgc_here = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_lo, mu_probe2 = 1.0 / theta_hi)
        pe_here = pivot_elim_from_cache(pgc_here, d.mu)
        xf_reduced = vcat(d.gp, d.xf[3:end])   # [gp; Aod_levels], matches ctx_frozen's own free_idx shape
        base_frozen = BaseDualState(xf_reduced, base.θ_full0, base.ζstar, base.λstar, base.m_star, base.inner_status)
        gfull_reduced_z, meta = composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws;
            base = base_frozen, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache)
        record_hits!(policy, meta.cache_hits[2:end])
        gfull_reduced = copy(gfull_reduced_z)
        gfull_reduced[2:end] .*= (-d.theta)   # chain rule dz/da = -theta (exact scalar, A-block only; gp untouched)
        a_grad_wall[] += time() - t_a0

        # ---- (2) theta gradient: fixed-dual central-difference secant, a_nonpivot held fixed ----
        t_theta0 = time()
        inner_x_fixed = copy(ctx.obj.x)   # obj.x currently holds this base point's own solved dual
        w_plus = copy(w); w_plus[1] += h_theta
        tp0 = time(); D_plus = theta_fixed_dual_delta_pivot_A(w_plus, inner_x_fixed, ctx, xy); theta_plus_wall[] += time() - tp0
        w_minus = copy(w); w_minus[1] -= h_theta
        tm0 = time(); D_minus = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx, xy); theta_minus_wall[] += time() - tm0
        grad_eta_theta = (D_plus - D_minus) / (2 * h_theta)
        # restore ctx.obj.H to the base point's state (theta_fixed_dual_delta_pivot_A mutates obj.H)
        tr0 = time()
        θ_full_base = CS.reconstruct_full(d.xf, ctx.m)
        ctx.obj.moments!(@view(ctx.obj.H[:, 1]), CS.select_G_from_H(ctx.obj, ctx.obj.H), θ_full_base, ctx.obj.U, ctx.obj)
        ctx.obj.H[:, 2] .= 1.0
        theta_restore_wall[] += time() - tr0
        theta_total_wall[] += time() - t_theta0

        jac_full = vcat(grad_eta_theta, gfull_reduced)
        n_grad_calls[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[2] = find_smallest ? 1.0 : -1.0
        evalResult.jac .= jac_full
        complete_grad_wall[] += time() - t_cg0
        return 0
    end

    function cb_newpt!(kc2, x, lambda, user_data)
        knitro_iter[] += 1
        shared = last_F_state[]
        r_now = (shared !== nothing && shared.w == x) ? shared.r : nothing
        d_now = (shared !== nothing && shared.w == x) ? shared.d : nothing
        if r_now === nothing
            r_now, _, d_now = screened_eval_flexible_A(collect(x), ctx, rsc, sc, n_eval, xy; warm = true, bank = bank, exact_cache = exact_cache)
        else
            n_checkpoint_reuse_hits[] += 1
        end
        if r_now.inner_status in FEASIBLE_CODES && isfinite(r_now.Delta_dual)
            do_checkpoint(:iteration, collect(x), r_now, d_now)
        end
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], n_outer), jacIndexVars = xIndices)
    KNITRO.KN_set_newpt_callback(kc, cb_newpt!)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    native_outer_diag = full_status_record(nStatus_code, kc)
    KNITRO.KN_free(kc)

    b = best_feasible[]
    σ = ctx.σ
    κ = b !== nothing ? 1 - b.gp^(σ / (σ - 1)) : NaN
    lp("[", label, "] FLEXIBLE-A POLISH DONE: status=", nStatus_code, " (", native_outer_diag.status_name, "/",
       native_outer_diag.status_category, ") wall_ext=", round(wall_ext, digits = 1), "s n_eval=", n_eval[],
       " kappa=", κ, " n_grad_calls=", n_grad_calls[],
       " theta_total_wall=", round(theta_total_wall[], digits = 1), "s a_grad_wall=", round(a_grad_wall[], digits = 1), "s")

    w_final = collect(xsol)
    r_final, _, d_final = screened_eval_flexible_A(w_final, ctx, rsc, sc, n_eval, xy; warm = true, bank = bank, exact_cache = exact_cache)
    if !(r_final.inner_status in FEASIBLE_CODES) || !is_verified_success(r_final)
        lp("[", label, "] WARNING: terminal point failed verification (inner_status=", r_final.inner_status, ") -- checkpointing as :stage_complete_unverified.")
        final_ckpt = do_checkpoint(:stage_complete_unverified, w_final, r_final, d_final)
    else
        final_ckpt = do_checkpoint(:stage_complete, w_final, r_final, d_final)
    end

    return (label = label, find_smallest = find_smallest, ctx = ctx, xy = xy,
            knitro_status = nStatus_code, native_outer_diag = native_outer_diag, wall_ext = wall_ext,
            n_eval = n_eval[], n_grad_calls = n_grad_calls[], best_feasible = b, kappa = κ, trace = trace,
            screen_counts = as_namedtuple(sc), final_checkpoint = final_ckpt,
            n_cold_retries = n_cold_retries[], n_rejected = n_rejected[], n_g_recompute = n_g_recompute[],
            theta_plus_wall = theta_plus_wall[], theta_minus_wall = theta_minus_wall[],
            theta_restore_wall = theta_restore_wall[], theta_total_wall = theta_total_wall[],
            a_grad_wall = a_grad_wall[], complete_grad_wall = complete_grad_wall[])
end
