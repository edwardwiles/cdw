# ============================================================================
# Checkpoint schema + opt-in production launcher for the origin-specific
# pairwise-zero-covariance restriction (task brief Section 12/13).
#
# Extends the CM checkpoint schema WITHOUT mutating CMCheckpoint/V3/V4
# (frozen, per cm_checkpoint.jl's own Serialization-gotcha discipline --
# CMCheckpointV5 is a NEW type name). cm_checkpoint.jl is NOT modified at
# all: `load_cm_checkpoint_v5` below tries V5 first, then falls back to the
# EXISTING `load_cm_checkpoint` (itself a V4/V3/legacy fallback chain),
# upgraded via `upgrade_schema4` -- zero edits to any existing file.
#
# `run_originzc_upper_checkpointed` is a SEPARATE new function (not a change
# to `run_cm_upper_checkpointed`'s body) -- this is the "explicit opt-in, not
# default" the task brief asks for: callers must call a different function
# by name to reach this restriction family at all.
# ============================================================================
using Serialization, Dates

const CM_CHECKPOINT_SCHEMA_V5 = 5
# Bumped 4 -> 5 (origin-specific-ZC integration, 2026-07-23): adds
# distribution_restriction, power_target_layout, origin_D to the persisted
# schema. `cm_extension`/`meanzc_K_mean`/`meanzc_K_pair`/`meanzc_basis`/
# `eta_nu`/`moment_layout_version` (V4 fields) are RETAINED, now describing
# the CM-family restriction only; `distribution_restriction`/`K_mean`/
# `K_pair` (new fields, `originzc_K_mean`/`originzc_K_pair` to avoid a name
# clash with the retained `meanzc_K_mean`/`meanzc_K_pair`) describe the
# ORIGIN-family restriction. A single checkpoint uses exactly one family
# non-trivially -- `save_originzc_checkpoint` asserts this (see below),
# never silently combines them (out of scope for this task).
const ORIGINZC_MOMENT_LAYOUT_VERSION = 1   # wrap_moments_with_originzc's column order, cm_originzc_moments.jl

"""
    CMCheckpointV5

Schema-5 checkpoint layout: `CMCheckpointV4`'s complete field set (CM-family
restriction, unchanged), PLUS `distribution_restriction`, `origin_K_mean`,
`origin_K_pair`, `power_target_layout`, `origin_D`, `origin_moment_layout_version`
(origin-family restriction). `eta_nu` (inherited field) holds whichever
family's current eta vector is live: length `meanzc_K_mean` for a CM-family
run, length `n_eta(layout)` (`origin_K_mean*origin_D` for `:origin_by_power`,
`origin_K_mean` for `:shared_by_power`) for an origin-family run.
"""
struct CMCheckpointV5
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol
    meanzc_K_mean::Int
    meanzc_K_pair::Int
    meanzc_basis::Symbol
    moment_layout_version::Int
    # ---- NEW (schema 5): origin-specific-ZC (no CM) restriction family ----
    distribution_restriction::Symbol   # :unrestricted | :origin_specific_moments | :origin_specific_moments_zero_covariance
    origin_K_mean::Int                 # 0 for :unrestricted
    origin_K_pair::Int                 # 0 for :unrestricted or :origin_specific_moments
    power_target_layout::Symbol        # :none | :shared_by_power | :origin_by_power
    origin_D::Int                      # 0 unless power_target_layout==:origin_by_power
    origin_moment_layout_version::Int  # bump if wrap_moments_with_originzc's column order ever changes
    # ---- outer-point / incumbent state ----
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
end

"Atomic-ish checkpoint write for CMCheckpointV5 (same discipline as `save_cm_checkpoint`)."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV5)
    (ckpt.cm_extension === :cm_only || ckpt.distribution_restriction === :unrestricted) ||
        error("save_cm_checkpoint: a single checkpoint must use exactly one restriction family non-trivially -- " *
              "got cm_extension=:$(ckpt.cm_extension) AND distribution_restriction=:$(ckpt.distribution_restriction) " *
              "both active. Combining CM with the origin-specific restriction is out of scope for this task.")
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"""
Upgrades a schema-4 `CMCheckpointV4` (origin-specific-ZC did not exist at
that schema) to `CMCheckpointV5`, filling distribution_restriction=:unrestricted,
origin_K_mean=origin_K_pair=0, power_target_layout=(old.cm_extension===:cm_only
? :none : :shared_by_power), origin_D=0, origin_moment_layout_version=0 --
CORRECT (not a guess): the origin-specific-ZC extension point did not exist
anywhere in the codebase when any schema-4 file was written, and every V4
meanzc arm used ONE shared nu_k (SharedByPowerLayout), so :shared_by_power is
the only value consistent with those files' own provenance.
"""
function upgrade_schema4(old::CMCheckpointV4)
    power_layout = old.cm_extension === :cm_only ? :none : :shared_by_power
    return CMCheckpointV5(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        :unrestricted, 0, 0, power_layout, 0, 0,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version)
end

"""
    load_cm_checkpoint_v5(path) -> CMCheckpointV5

Tries schema-5 (`CMCheckpointV5`) first; falls back to `load_cm_checkpoint`
(cm_checkpoint.jl, itself a V4/V3/legacy fallback chain) upgraded via
`upgrade_schema4`. Always returns a `CMCheckpointV5`. This is the loader
`run_originzc_upper_checkpointed` uses; plain CM-family callers continue to
use `load_cm_checkpoint` (V4) unchanged.
"""
function load_cm_checkpoint_v5(path::AbstractString)
    try
        return deserialize(path)::CMCheckpointV5
    catch e
        (e isa TypeError || e isa EOFError || e isa MethodError) || rethrow()
        return upgrade_schema4(load_cm_checkpoint(path))
    end
end

"""
    run_originzc_upper_checkpointed(w0; W, delta, draw_design, draw_seed, distribution_restriction,
        K_mean, K_pair, power_target_layout=:origin_by_power, meanzc_basis=:direct,
        nu_bounds=nothing, maxtime_real, opt_file, ckpt_dir, run_id, label, checkpoint_interval_s,
        resume_from=nothing, cm_gradient_backend=:cplus, ...) -> NamedTuple

Checkpointed outer loop for the origin-specific-ZC (no CM) restriction family
-- SEPARATE function from `run_cm_upper_checkpointed` (explicit opt-in per
task brief Section 12: callers must call a DIFFERENT function by name to
reach this restriction at all; `run_cm_upper_checkpointed` itself is
untouched). Structural analog of `run_cm_upper_checkpointed`, minus
everything CM-specific (no `L`/`probs`/`contrasts`/`cm_hessian_backend`
kwargs -- this arm has no CM-grid block and always uses Architecture A,
cm_originzc_production.jl).
"""
function run_originzc_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        maxtime_real::Float64 = 180.0, opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        ckpt_dir::AbstractString, run_id::String = string(Dates.now()), label::String = "originzc_upper",
        checkpoint_interval_s::Float64 = 90.0, resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        cm_gradient_backend::Symbol = :cplus,
        allow_backend_switch::Bool = false,
        distribution_restriction::Symbol,   # REQUIRED, no default -- explicit opt-in (task brief Section 12)
        K_mean::Int, K_pair::Int = 0,
        power_target_layout::Symbol = :origin_by_power, meanzc_basis::Symbol = :direct,
        nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing)
    lp(xs...) = (println(xs...); flush(stdout))
    mkpath(ckpt_dir)

    distribution_restriction !== :unrestricted ||
        error("run_originzc_upper_checkpointed($label): distribution_restriction=:unrestricted has no restriction to run " *
              "-- this function is the explicit opt-in entry point for the origin-specific-ZC family; use the plain " *
              "unrestricted driver directly instead.")
    cm_gradient_backend in (:reference, :cplus) ||
        error("run_originzc_upper_checkpointed($label): cm_gradient_backend must be :reference|:cplus, got :$cm_gradient_backend")
    lp("[", label, "] distribution_restriction=", distribution_restriction, " power_target_layout=", power_target_layout,
       " K_mean=", K_mean, " K_pair=", K_pair, " cm_gradient_backend=", cm_gradient_backend)

    cfg = OriginZCConfig(distribution_restriction = distribution_restriction, K_mean = K_mean, K_pair = K_pair,
                          power_target_layout = power_target_layout, meanzc_basis = meanzc_basis, nu_bounds = nu_bounds)
    K_mean_r, K_pair_r = originzc_resolve_K(cfg)   # raises on any config inconsistency

    resumed = resume_from === nothing ? nothing : load_cm_checkpoint_v5(resume_from)
    find_smallest = true
    backend_switched = false

    if resumed !== nothing
        (resumed.distribution_restriction == distribution_restriction && resumed.origin_K_mean == K_mean_r &&
         resumed.origin_K_pair == K_pair_r && resumed.power_target_layout == power_target_layout) ||
            error("run_originzc_upper_checkpointed($label): distribution_restriction/K/layout MISMATCH on resume -- " *
                  "checkpoint has distribution_restriction=:$(resumed.distribution_restriction), " *
                  "K_mean=$(resumed.origin_K_mean), K_pair=$(resumed.origin_K_pair), " *
                  "power_target_layout=:$(resumed.power_target_layout); this call requests " *
                  "distribution_restriction=:$distribution_restriction, K_mean=$K_mean_r, K_pair=$K_pair_r, " *
                  "power_target_layout=:$power_target_layout -- refusing to resume under a different moment layout.")
        resumed.origin_moment_layout_version == ORIGINZC_MOMENT_LAYOUT_VERSION ||
            error("run_originzc_upper_checkpointed($label): origin_moment_layout_version MISMATCH -- checkpoint=" *
                  "$(resumed.origin_moment_layout_version), current=$(ORIGINZC_MOMENT_LAYOUT_VERSION). Refusing to resume.")
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
        backend_switched = resumed.cm_gradient_backend != cm_gradient_backend
        if backend_switched && !allow_backend_switch
            error("run_originzc_upper_checkpointed($label): checkpoint was written with cm_gradient_backend=:" *
                  "$(resumed.cm_gradient_backend), but this call requests :$(cm_gradient_backend) -- refusing to " *
                  "silently switch backends on resume. Pass allow_backend_switch=true if intentional.")
        end
    end

    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed)
    pe = build_pivot_elimination(ctx)
    D = ctx.D
    layout = originzc_make_layout(cfg, D)
    originzc_validate_bounds(cfg, layout)

    if resumed !== nothing
        (power_target_layout != :origin_by_power || resumed.origin_D == D) ||
            error("run_originzc_upper_checkpointed($label): D MISMATCH on resume -- checkpoint origin_D=" *
                  "$(resumed.origin_D), current D=$D -- refusing to resume under a different origin dimension.")
        (ctx.draw_meta.checksum_uniform == resumed.draw_checksum_uniform &&
         ctx.draw_meta.checksum_transformed == resumed.draw_checksum_transformed) ||
            error("run_originzc_upper_checkpointed($label): draw checksum MISMATCH on resume -- regenerated draws " *
                  "(design=:$(draw_design), seed=$(draw_seed)) do not match the checkpoint's own recorded checksums.")
        w0 = vcat(resumed.g, resumed.zfree, resumed.eta_nu)
    elseif w0 === nothing
        error("run_originzc_upper_checkpointed($label): w0 required for a fresh (non-resumed) run -- must be " *
              "vcat(gp, zfree, eta) with length(eta)==$(n_eta(layout))")
    end

    pcx = build_originzc_production_context(ctx, CS, layout)
    D2_econ = length(w0) - n_eta(layout)

    bounds = cfg.nu_bounds === nothing ? originzc_default_nu_bounds(ctx, layout) : cfg.nu_bounds
    lp("[", label, "] eta box (per coordinate, log-nu units): n=", length(bounds))

    cplus_pool = cm_gradient_backend == :cplus ? build_grad_workspace_pool(size(ctx.obj.U, 1)) : nothing
    cplus_ws = cm_gradient_backend == :cplus ? build_lfix_factorized_workspace(ctx.D, size(ctx.obj.U, 1)) : nothing

    if backend_switched && resumed.best_feasible !== nothing
        xf_switch = x_free_from_w(resumed.best_feasible.w[1:D2_econ], pe)
        νvec_switch = exp.(resumed.best_feasible.w[D2_econ+1:end])
        (_, _, vs) = cm_originzc_production_value_verified(xf_switch, νvec_switch, pcx)
        is_verified_success(vs) ||
            error("run_originzc_upper_checkpointed($label): backend switch on resume requested, but the resumed " *
                  "incumbent FAILED independent cold re-verification under the new backend -- refusing to carry it forward.")
        lp("[", label, "] backend-switch cold re-verification of resumed incumbent: Delta_dual=", vs.Delta_dual,
           " (checkpoint recorded ", resumed.best_feasible.Delta, ") -- PASSED.")
    end

    D2 = length(w0)
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo_econ = vcat(gp_lo, w0[2:D2_econ] .- z_halfwidth)
    w_hi_econ = vcat(gp_hi, w0[2:D2_econ] .+ z_halfwidth)
    w_lo = vcat(w_lo_econ, [bounds[k][1] for k in 1:n_eta(layout)])
    w_hi = vcat(w_hi_econ, [bounds[k][2] for k in 1:n_eta(layout)])

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    n_grad = Ref(resumed !== nothing ? resumed.n_grad : 0)
    trace = NamedTuple[]
    bandwidth_cache = (resumed !== nothing && !backend_switched) ? copy(resumed.bandwidth_cache) : Dict{Int,Float64}()
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = w_current[2:D2_econ]
        eta_now = w_current[D2_econ+1:end]
        logA_full = pivot_expand(zfree_now, pe)
        dual_warm_src = pcx.ctx_cm.obj.x
        ckpt = CMCheckpointV5(CM_CHECKPOINT_SCHEMA_V5, run_id, label, :cm_upper, find_smallest, delta, W, draw_seed,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            0, Float64[], :anchored, :equal, :cumulative, :dense_reference, cm_gradient_backend,
            :cm_only, 0, 0, :direct, 0,
            distribution_restriction, K_mean_r, K_pair_r, power_target_layout, layout_D(layout), ORIGINZC_MOMENT_LAYOUT_VERSION,
            w_current[1], copy(zfree_now), copy(eta_now), logA_full, copy(dual_warm_src), copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version)
        path = joinpath(ckpt_dir, "$(label)_latest.jls")
        save_cm_checkpoint(path, ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:D2_econ], pe)
        νvec = exp.(w[D2_econ+1:end])
        local base, verify
        try
            _, base, verify = cm_originzc_production_value_verified(xf, νvec, pcx)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_originzc_upper_checkpointed($label): infeasible/failed inner solve at this point")
        end
        Δ = verify.Delta_dual
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base, verify = verify)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        verified = is_verified_success(verify)
        is_new_best = feasible && verified && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = prior_wall + (time() - t_start))
            do_checkpoint(:new_best, collect(w))
        elseif time() - last_ckpt_t[] >= checkpoint_interval_s
            do_checkpoint(:wall_interval, collect(w))
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            lp("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:D2_econ], pe)
        νvec = exp.(w[D2_econ+1:end])
        shared = last_F_state[]
        matched = shared !== nothing && shared.w == w
        base = matched ? shared.base : nothing
        verify_c = matched ? shared.verify : nothing
        gfull, meta = if cm_gradient_backend == :cplus
            cm_originzc_production_gradient_cplus(xf, νvec, pcx, ctx, pe, cplus_pool, cplus_ws;
                base = base, verify = verify_c, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        else
            cm_originzc_production_gradient(xf, νvec, pcx, ctx, pe;
                base = base, verify = verify_c, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        end
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= gfull
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))

    xsol_v = collect(xsol)
    xf_final = x_free_from_w(xsol_v[1:D2_econ], pe)
    local verify_final
    try
        νvec_final = exp.(xsol_v[D2_econ+1:end])
        _, _, verify_final = cm_originzc_production_value_verified(xf_final, νvec_final, pcx)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        verify_final = (inner_status = -300,)
    end
    final_ckpt = if is_verified_success(verify_final)
        do_checkpoint(:stage_complete, collect(xsol))
    else
        lp("[", label, "] WARNING: terminal point failed verification -- checkpointing as :stage_complete_unverified.")
        do_checkpoint(:stage_complete_unverified, collect(xsol))
    end
    return (knitro_status = nStatus, wall = wall_ext, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end
