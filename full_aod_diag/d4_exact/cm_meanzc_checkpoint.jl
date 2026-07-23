# ============================================================================
# CM+moments(+ZC) production checkpoint schema (task brief 9.1): extends the
# current canonical checkpoint DISCIPLINE (CMCheckpoint, cm_checkpoint.jl --
# atomic write, schema-version hard-refusal on mismatch, best-verified-
# incumbent tracked separately from the terminal iterate, draw-provenance
# validation on resume) rather than forking a new one-off driver -- this is
# the exact hazard the archived prototype's own schema=1 driver
# (c40_meanzc_checkpoint.jl, audit finding 3.3) is flagged for.
#
# A genuinely NEW struct type (not a bump of CM_CHECKPOINT_SCHEMA on the
# existing CMCheckpoint) because the field set differs (K_mean/K_pair/
# meanzc_basis/nu_bounds/eta_nu have no CM-only analog) and Julia's
# Serialization module is not layout-tolerant across struct field changes --
# bumping CMCheckpoint's OWN schema would make every already-written
# schema-2 checkpoint (including the just-completed 2026-07-22 CM campaign's)
# undeserializable outright instead of cleanly refusing with an actionable
# message. Distinguished by TYPE, matching the same reasoning the archived
# prototype used for its own (differently-motivated) choice.
#
# Provenance validation uses context_fingerprint(ctx) (oracle.jl, AUD-08) --
# already handles both real D=20 contexts (via ctx.draw_meta's checksums) and
# D=4 test contexts (falls back to hashing ctx.U directly) uniformly, so this
# checkpoint schema and its round-trip tests work at either scale without a
# separate D=4 code path.
# ============================================================================
using Serialization, Dates

const CM_MEANZC_CHECKPOINT_SCHEMA = 1

"""
    CMMeanZCCheckpoint

CM+moments(+ZC)-production analogue of `CMCheckpoint`. Covers everything a
resume needs to (a) validate it is reconstructing the IDENTICAL problem
instance and (b) restart the outer KNITRO search from the best exact-feasible
incumbent found so far -- plus the extension-specific configuration
(`K_mean`, `K_pair`, `meanzc_basis`, `nu_bounds`) and outer-point state
(`eta_nu`, appended to `g`/`zfree` per the task brief's "all exact keys,
fingerprints, checkpoints, bounds, cold-verification files ... must include
that coordinate" requirement).
"""
struct CMMeanZCCheckpoint
    schema::Int
    run_id::String
    label::String
    branch::Symbol                 # :cm_meanzc_upper (only direction wired today)
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    ctx_fingerprint::String        # context_fingerprint(ctx) -- AUD-08 pattern, works at any D
    # ---- CM configuration (same fields as CMCheckpoint) ----
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    # ---- meanzc-specific configuration ----
    K_mean::Int
    K_pair::Int
    meanzc_basis::Symbol
    nu_bounds::Vector{NTuple{2,Float64}}   # eta_nu-space (log nu) box, one per level
    # ---- outer-point / incumbent state ----
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}        # length K_mean -- the NEW trailing outer coordinates
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any             # NamedTuple or nothing
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol      # :new_best | :wall_interval | :stage_complete | :stage_complete_unverified
    knitro_version::String
end

"Atomic-ish checkpoint write, same discipline as save_cm_checkpoint: serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint."
function save_cm_meanzc_checkpoint(path::AbstractString, ckpt::CMMeanZCCheckpoint)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

function load_cm_meanzc_checkpoint(path::AbstractString)
    ckpt = deserialize(path)::CMMeanZCCheckpoint
    ckpt.schema == CM_MEANZC_CHECKPOINT_SCHEMA ||
        error("load_cm_meanzc_checkpoint($path): schema=$(ckpt.schema), expected $(CM_MEANZC_CHECKPOINT_SCHEMA) -- " *
              "refusing to resume from an incompatible checkpoint. Start a fresh run instead.")
    return ckpt
end

"""
    run_cm_meanzc_upper_checkpointed(ctx_builder, w0_ext=nothing; W, delta, draw_design, draw_seed,
        L, contrasts, probs, K_mean, K_pair, meanzc_basis=:direct, nu_bounds=nothing,
        cm_hessian_backend=:structured, cm_grid_rule=:equal, maxtime_real=180.0,
        opt_file="csw_outer_wallclock_sr1.opt", z_halfwidth=30.0,
        ckpt_dir, run_id=string(now()), label="cm_meanzc_upper", checkpoint_interval_s=90.0,
        resume_from=nothing, verbose=true, heartbeat_interval_s=nothing) -> NamedTuple

Checkpointed CM+moments(+ZC) outer loop, same objective/constraint/gradient
path as `run_cm_meanzc_upper` (cm_meanzc_outer_driver.jl, UNCHANGED, reused
not reimplemented) but with `CMCheckpoint`-grade resume support.

`ctx_builder` is a `(; W, δ, find_smallest, draw_design, draw_seed) -> ctx`
function -- production callers pass `d20_real_setup_design` (matching the
current canonical D=20 driver, cm_production_stage_runner.jl); tests can pass
a lightweight adapter (see test_cm_meanzc_checkpoint.jl) so the checkpoint
schema/resume MECHANICS are testable at D=4 speed without requiring a real
D=20 solve for every gate. `resume_from` (a `CMMeanZCCheckpoint` path)
rebuilds `ctx` from the checkpoint's own recorded provenance and REFUSES to
resume (hard error) if the regenerated `context_fingerprint` doesn't match.
"""
function run_cm_meanzc_upper_checkpointed(ctx_builder::Function, w0_ext::Union{Nothing,Vector{Float64}} = nothing;
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        L::Int = 10, contrasts::Symbol = :anchored, probs::Union{Nothing,AbstractVector{Float64}} = nothing,
        K_mean::Int = 1, K_pair::Int = 0, meanzc_basis::Symbol = :direct,
        nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
        cm_hessian_backend::Symbol = :structured, cm_grid_rule::Symbol = :equal,
        maxtime_real::Float64 = 180.0, opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        ckpt_dir::AbstractString, run_id::String = string(Dates.now()), label::String = "cm_meanzc_upper",
        checkpoint_interval_s::Float64 = 90.0, resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true, heartbeat_interval_s::Union{Nothing,Float64} = nothing)
    lp(xs...) = (println(xs...); flush(stdout))
    mkpath(ckpt_dir)

    resumed = resume_from === nothing ? nothing : load_cm_meanzc_checkpoint(resume_from)
    find_smallest = true   # :cm_meanzc_upper is the only wired direction today

    if resumed !== nothing
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        L = resumed.cm_L; probs = resumed.cm_probs; contrasts = resumed.cm_contrasts
        cm_hessian_backend = resumed.cm_hessian_backend; cm_grid_rule = resumed.cm_grid_rule
        K_mean = resumed.K_mean; K_pair = resumed.K_pair; meanzc_basis = resumed.meanzc_basis
        nu_bounds = resumed.nu_bounds
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
        if resumed.checkpoint_reason == :stage_complete_unverified
            lp("WARNING: resuming from a checkpoint whose terminal point FAILED verification at write time ",
               "-- best_feasible[] remains the correct scientific incumbent; only the raw terminal solver-state fields are suspect.")
        end
    end

    ctx = ctx_builder(; W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed)
    pe = build_pivot_elimination(ctx)

    if resumed !== nothing
        context_fingerprint(ctx) == resumed.ctx_fingerprint ||
            error("run_cm_meanzc_upper_checkpointed($label): context fingerprint MISMATCH on resume -- " *
                  "regenerated context (draw_design=:$(draw_design), seed=$(draw_seed)) does not match the " *
                  "checkpoint's own recorded fingerprint. Refusing to resume from a different problem instance.")
        w0_ext = vcat(resumed.g, resumed.zfree, resumed.eta_nu)
    elseif w0_ext === nothing
        error("run_cm_meanzc_upper_checkpointed($label): w0_ext required for a fresh (non-resumed) run")
    end

    probs === nothing && error("run_cm_meanzc_upper_checkpointed($label): probs required (exact cutpoints, not re-derived from L)")
    cfg = CMMeanZCConfig(cm = CMConfig(cm_grid_size = L, cm_grid_rule = cm_grid_rule, cm_basis = :cumulative,
                                        cm_hessian_backend = cm_hessian_backend, contrasts = contrasts),
                          cm_extension = :cm_plus_moments, meanzc_K_mean = K_mean, meanzc_K_pair = K_pair,
                          meanzc_basis = meanzc_basis,
                          meanzc_nu_bounds = nu_bounds === nothing ? meanzc_default_nu_bounds(ctx, K_mean) : nu_bounds)
    _meanzc_validate(cfg)
    pcx = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        contrasts = contrasts, meanzc_basis = meanzc_basis, probs = probs)
    nub = cfg.meanzc_nu_bounds

    D2ext = length(w0_ext)
    D2 = D2ext - K_mean
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo = vcat(gp_lo, w0_ext[2:D2] .- z_halfwidth, first.(nub))
    w_hi = vcat(gp_hi, w0_ext[2:D2] .+ z_halfwidth, last.(nub))

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2ext)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0_ext)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    n_grad = Ref(resumed !== nothing ? resumed.n_grad : 0)
    trace = NamedTuple[]
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    fp = context_fingerprint(ctx)
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = w_current[2:D2]
        eta_nu_now = w_current[D2+1:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = CMMeanZCCheckpoint(CM_MEANZC_CHECKPOINT_SCHEMA, run_id, label, :cm_meanzc_upper, find_smallest, delta,
            W, draw_seed, draw_design, fp,
            L, collect(probs), contrasts, cm_grid_rule, :cumulative, cm_hessian_backend,
            K_mean, K_pair, meanzc_basis, nub,
            w_current[1], copy(zfree_now), copy(eta_nu_now), logA_full, copy(ctx.obj.x), Dict{Int,Float64}(),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version)
        path = joinpath(ckpt_dir, "$(label)_latest.jls")
        save_cm_meanzc_checkpoint(path, ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w_meanzc(w, pe, K_mean)
        νvec = nu_from_w_meanzc(w, K_mean)
        local base, verify
        try
            _, base, verify = cm_meanzc_production_value_verified(xf, νvec, pcx)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_cm_meanzc_upper_checkpointed($label): infeasible/failed inner solve at this point")
        end
        Δ = verify.Delta_dual
        evalResult.obj[1] = w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base)
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
            lp("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " nu=", νvec, " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w_meanzc(w, pe, K_mean)
        νvec = nu_from_w_meanzc(w, K_mean)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        gext, meta = cm_meanzc_production_gradient(xf, νvec, pcx, ctx, pe; base = base)
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac .= gext
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2ext), jacIndexVars = xIndices)

    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))

    xf_final = x_free_from_w_meanzc(collect(xsol), pe, K_mean)
    νvec_final = nu_from_w_meanzc(collect(xsol), K_mean)
    local verify_final
    try
        _, _, verify_final = cm_meanzc_production_value_verified(xf_final, νvec_final, pcx)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        verify_final = (inner_status = -300,)
    end
    if is_verified_success(verify_final)
        final_ckpt = do_checkpoint(:stage_complete, collect(xsol))
    else
        lp("[", label, "] WARNING: terminal point failed verification -- checkpointing as :stage_complete_unverified.")
        final_ckpt = do_checkpoint(:stage_complete_unverified, collect(xsol))
    end
    return (knitro_status = nStatus, wall = wall_ext, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"))
end
