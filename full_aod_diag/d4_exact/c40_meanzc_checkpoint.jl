# ============================================================================
# Section 9 (D=20 short outer-loop trial) checkpoint schema + checkpointed driver
# for the two extended arms. Exact structural analog of cm_checkpoint.jl's
# CMCheckpoint/run_cm_upper_checkpointed, extended by exactly the eta_nu
# scalar (appended to the outer state, never inserted -- w = [g; zfree; eta_nu]).
#
# This is a NEW checkpoint type (CM_MEANZC_CHECKPOINT_SCHEMA starts at 1, not a
# "bump" of CM_CHECKPOINT_SCHEMA, which stays at 2 and is untouched -- cm_only
# checkpoints and this file's checkpoints are never compatible/interchangeable,
# distinguished by TYPE not just a schema-version int).
# ============================================================================
using Serialization, Dates

const CM_MEANZC_CHECKPOINT_SCHEMA = 1

"""
    CMMeanZCCheckpoint

CM+mean(+ZC) analog of `CMCheckpoint`. Carries everything a resume needs to (a) validate it is
reconstructing the IDENTICAL problem instance (draw design/checksums, W, delta, CM grid) and (b)
restart the outer KNITRO search from the best exact-feasible incumbent, INCLUDING its eta_nu
state -- resuming with `eta_nu` reset to some default would silently restart the ZC/mean profile
from scratch, discarding real search progress.
"""
struct CMMeanZCCheckpoint
    schema::Int
    run_id::String
    label::String
    branch::Symbol                 # :meanzc_upper
    cm_extension::Symbol           # :cm_plus_mean | :cm_plus_mean_zero_covariance
    meanzc_basis::Symbol           # :direct | :anchored
    nu_parameterization::Symbol    # :log
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
    # ---- outer-point / incumbent state ----
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Float64
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any              # NamedTuple (gp, w, Delta, nu, n_eval, t) or nothing
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    checkpoint_reason::Symbol       # :new_best | :wall_interval | :stage_complete | :stage_complete_unverified
    knitro_version::String
end

function save_cm_meanzc_checkpoint(path::AbstractString, ckpt::CMMeanZCCheckpoint)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

function load_cm_meanzc_checkpoint(path::AbstractString)
    ckpt = deserialize(path)::CMMeanZCCheckpoint
    ckpt.schema == CM_MEANZC_CHECKPOINT_SCHEMA ||
        error("load_cm_meanzc_checkpoint($path): schema=$(ckpt.schema), expected $(CM_MEANZC_CHECKPOINT_SCHEMA)")
    return ckpt
end

"""
    run_meanzc_upper_checkpointed(w0=nothing; W, delta, draw_design, draw_seed, L, cm_extension,
        meanzc_basis=:direct, eta_nu_bounds, maxtime_real=900.0, ckpt_dir, run_id, label,
        checkpoint_interval_s=90.0, resume_from=nothing, verbose=true) -> NamedTuple

Checkpointed outer loop for `cm_extension in (:cm_plus_mean, :cm_plus_mean_zero_covariance)`.
`w0 = [g; zfree; eta_nu]` for a fresh run (required unless `resume_from` is given).
`eta_nu_bounds = (lo, hi)` sets the KNITRO box bounds for the appended coordinate -- callers
should pass `nu_feasible_interval(ctx.U)` log-transformed (Section 3), NOT an arbitrary guess.
Mirrors `run_cm_upper_checkpointed`'s resume/verification/AUD-04/AUD-10 discipline exactly.
"""
function run_meanzc_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        L::Int = 10, cm_extension::Symbol, meanzc_basis::Symbol = :direct, contrasts::Symbol = :anchored,
        probs::Union{Nothing,AbstractVector{Float64}} = nothing,
        maxtime_real::Float64 = 900.0, opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0, eta_nu_bounds::Union{Nothing,Tuple{Float64,Float64}} = nothing,
        gp_lo::Union{Nothing,Float64} = nothing, gp_hi::Union{Nothing,Float64} = nothing,
        ckpt_dir::AbstractString, run_id::String = string(Dates.now()), label::String = "meanzc_upper",
        checkpoint_interval_s::Float64 = 90.0, resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true)
    cm_extension in (:cm_plus_mean, :cm_plus_mean_zero_covariance) ||
        error("run_meanzc_upper_checkpointed: cm_extension must be :cm_plus_mean or :cm_plus_mean_zero_covariance, got $cm_extension")
    lp(xs...) = (println(xs...); flush(stdout))
    mkpath(ckpt_dir)

    resumed = resume_from === nothing ? nothing : load_cm_meanzc_checkpoint(resume_from)
    find_smallest = true

    if resumed !== nothing
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        L = resumed.cm_L; probs = resumed.cm_probs; contrasts = resumed.cm_contrasts
        cm_extension = resumed.cm_extension; meanzc_basis = resumed.meanzc_basis
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s, eta_nu=", resumed.eta_nu, ")")
        if resumed.checkpoint_reason == :stage_complete_unverified
            lp("WARNING: resuming from a checkpoint whose terminal point FAILED verification at write time.")
        end
    end

    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed)
    pe = build_pivot_elimination(ctx)

    if resumed !== nothing
        if ctx.draw_meta.checksum_uniform != resumed.draw_checksum_uniform ||
           ctx.draw_meta.checksum_transformed != resumed.draw_checksum_transformed
            error("run_meanzc_upper_checkpointed($label): draw checksum MISMATCH on resume -- refusing.")
        end
        w0 = vcat(resumed.g, resumed.zfree, resumed.eta_nu)
    elseif w0 === nothing
        error("run_meanzc_upper_checkpointed($label): w0 required for a fresh (non-resumed) run")
    end
    probs === nothing && error("run_meanzc_upper_checkpointed($label): probs required (exact cutpoints)")

    nu_ref = Ref(1.0)
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = cm_extension, contrasts = contrasts,
        meanzc_basis = meanzc_basis, probs = probs, nu_ref = nu_ref)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    bins = cm_bin_indices_for(ctx, aug)
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    pcx = (ctx_cm = ctx_cm, aug = aug, bins = bins, cctx = cctx)

    nu_lo, nu_hi = eta_nu_bounds === nothing ? log.(nu_feasible_interval(ctx.U)) : eta_nu_bounds
    gplo = gp_lo === nothing ? ctx.bounds.γp_lo : gp_lo
    gphi = gp_hi === nothing ? ctx.bounds.γp_hi : gp_hi

    D2 = length(w0)   # 1(g) + n_free_econ(zfree) + 1(eta_nu)
    w_lo = vcat(gplo, w0[2:end-1] .- z_halfwidth, nu_lo)
    w_hi = vcat(gphi, w0[2:end-1] .+ z_halfwidth, nu_hi)

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
    bandwidth_cache = resumed !== nothing ? copy(resumed.bandwidth_cache) : Dict{Int,Float64}()
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    knitro_version = try KNITRO.KN_get_release() catch; "unknown" end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = w_current[2:end-1]
        η_now = w_current[end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = CMMeanZCCheckpoint(CM_MEANZC_CHECKPOINT_SCHEMA, run_id, label, :meanzc_upper,
            cm_extension, meanzc_basis, :log, find_smallest, delta, W, draw_seed,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            L, collect(probs), contrasts,
            w_current[1], copy(zfree_now), η_now, logA_full, copy(ctx.obj.x), copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start), reason, knitro_version)
        path = joinpath(ckpt_dir, "$(label)_latest.jls")
        save_cm_meanzc_checkpoint(path, ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:end-1], pe)
        η_ν = w[end]
        aug.nu_ref[] = exp(η_ν)
        local base, verify
        try
            _, base, verify = cm_meanzc_production_value_verified(xf, pcx)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_meanzc_upper_checkpointed($label): infeasible/failed inner solve (eta_nu=$η_ν)")
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
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, nu = exp(η_ν), n_eval = n_eval[], t = prior_wall + (time() - t_start))
            do_checkpoint(:new_best, collect(w))
        elseif time() - last_ckpt_t[] >= checkpoint_interval_s
            do_checkpoint(:wall_interval, collect(w))
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], eta_nu = η_ν, nu = exp(η_ν),
                       Delta = Δ, feasible = feasible, verified = verified))
        if verbose && (n_eval[] <= 3 || n_eval[] % 10 == 0)
            lp("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1],
               " nu=", exp(η_ν), " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = x_free_from_w(w[1:end-1], pe)
        shared = last_F_state[]
        match = (shared !== nothing && shared.w == w)
        base = match ? shared.base : nothing
        verify = match ? shared.verify : nothing
        if base === nothing || verify === nothing
            aug.nu_ref[] = exp(w[end])
            _, base, verify = cm_meanzc_production_value_verified(xf, pcx)
        end
        gfull, meta = cm_meanzc_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
            h_mode = :cached, bandwidth_cache = bandwidth_cache)
        n_grad[] += 1
        d_eta = d_delta_dual_d_eta_nu(base.λstar, aug, aug.nu_ref[]; mean_m = verify.m_mean)
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = 1.0
        evalResult.jac[1:end-1] .= gfull
        evalResult.jac[end] = d_eta
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

    xf_final = x_free_from_w(collect(xsol)[1:end-1], pe)
    aug.nu_ref[] = exp(xsol[end])
    local verify_final
    try
        _, _, verify_final = cm_meanzc_production_value_verified(xf_final, pcx)
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
