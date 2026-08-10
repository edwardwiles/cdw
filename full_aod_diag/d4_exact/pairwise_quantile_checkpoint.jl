# ================================================================================================
# Checkpointed OUTER driver for the pairwise-quantile-independence restriction family.
#
# SEPARATE function from every other family's driver, by the same explicit-opt-in discipline
# `run_originzc_upper_checkpointed` follows: a caller must name THIS function to reach this
# restriction at all. Structural analog of `run_originzc_upper_checkpointed`
# (cm_originzc_checkpoint.jl:510-1035), step for step, minus everything ZC-specific (no
# nu/eta/K_mean/K_pair/power_target_layout -- this family's outer restriction coordinates are the
# `n_raw(layout) = (L-1)*D` raw cutoff coordinates) and plus this family's own two required
# modelling choices, `L` (number of quantile bins) and `min_crossed` (the cutoff gradient's secant
# bandwidth).
#
# OUTER COORDINATE LAYOUT (load-bearing; the gradient is assembled to match index-for-index):
#     w = vcat( gp , zfree , raw_cutoffs )
#           |    |      |         |
#           |    |      |         +-- n_raw(layout) = (L-1)*D  cutoff coordinates, origin-major
#           |    |      +------------ D*Ddest - 1 pivot-reduced economic coordinates
#           |    +------------------- w[1], the gp coordinate
#           +------------------------ D2_econ = D*Ddest, the economic block width
# mirroring origin-ZC's own `w = vcat(gp, zfree, eta)`. `pairwise_quantile_production_gradient`
# returns `vcat(g_econ, g_cut)` with exactly these widths.
#
# NO DEFAULTS ON SCIENTIFIC PARAMETERS. Per CLAUDE.md's standing rule ("no function anywhere in the
# EK/Ricardo FULL/REDUCED call chain may give a default value to a parameter that changes what
# economic problem is being solved"), every parameter of that kind is a bare keyword with no `=`,
# so Julia raises `UndefKeywordError` before the body runs if a caller omits it: sigma, W, delta,
# draw_design, draw_seed, destination_sample, the gravity exclusions, inner_lower_limit,
# find_smallest, z_halfwidth -- plus this family's own `L` and `min_crossed`. This is a NEW entry
# point, so there are no legacy callers for that strictness to break; it deliberately does NOT copy
# the defaults `run_originzc_upper_checkpointed` still carries on this branch.
#
# Requires pairwise_quantile_outer_production.jl (and its include chain), cm_originzc_checkpoint.jl
# (for the shared checkpoint I/O discipline this mirrors), and c10_d20_production_driver.jl
# (x_free_from_w) to already be included.
# ================================================================================================

using Serialization: serialize, deserialize
using LinearAlgebra: BLAS
import Dates

"""
    PairwiseQuantileCheckpointV1

This family's checkpoint schema. Deliberately a FRESH schema (no faked version history) carrying
the same STRUCTURAL field groups every other family's checkpoint carries -- run identity, draw
provenance, outer-point/incumbent state -- plus this family's own fields (`L`, `min_crossed`,
`raw_cutoffs`, `n_raw`).

The NAME is deliberately unique across the whole repo, and that is not cosmetic: Julia's
`Serialization.deserialize` resolves types by NAME, and this codebase has already been bitten once
by two unrelated `struct`s both called `CMCheckpointV10` in different files, where whichever
`include` ran last silently redefined the binding and the other family's `do_checkpoint` then
constructed against the wrong field layout (see `OriginZCCheckpointV10`'s own docstring, and memory
`feedback-julia-serialization-type-name-collision`). Verified unique by grep before adding.
"""
struct PairwiseQuantileCheckpointV1
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    # ---- draw provenance ----
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    # ---- this family's own restriction identity ----
    L::Int
    min_crossed::Int
    n_raw::Int
    D::Int
    # ---- outer point ----
    g::Float64
    zfree::Vector{Float64}
    raw_cutoffs::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    # ---- progress/incumbent ----
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
    # ---- scientific context identity ----
    destination_sample::Symbol
    row_idx::Union{Nothing,Int}
    D_dest::Int
    A_coordinate_mode::Symbol
    sigma::Float64
end

const PAIRWISE_QUANTILE_CHECKPOINT_SCHEMA_V1 = 1

"Atomic-ish checkpoint write: serialize to `<path>.tmp`, then `mv` onto `path`. Identical discipline to `save_cm_checkpoint`'s own V5/V7/V10 methods -- plain `Serialization`, NOT JLD2/JSON, matching every other checkpoint in this codebase."
function save_pairwise_quantile_checkpoint(path::AbstractString, ckpt::PairwiseQuantileCheckpointV1)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Loader. No fallback/upgrade chain by design: this is schema 1, there is no earlier schema of this family's checkpoint to upgrade FROM, and inventing one would be fiction."
function load_pairwise_quantile_checkpoint(path::AbstractString)
    ckpt = deserialize(path)
    ckpt isa PairwiseQuantileCheckpointV1 ||
        error("load_pairwise_quantile_checkpoint($path): deserialized a $(typeof(ckpt)), not a " *
              "PairwiseQuantileCheckpointV1 -- refusing to resume from a foreign checkpoint.")
    return ckpt
end

"""
    pairwise_quantile_start_cutoffs(ctx, layout) -> Vector{Float64}

Raw cutoff coordinates placing each origin's cutoffs at that origin's OWN empirical quantiles
(`q_{o,r}` = the `r/L` empirical quantile of `ctx.U[:,o]`), inverted through the softplus-ordered
transform. This is the family's canonical starting point, and the direct analog of origin-ZC's
`companion_implied_nu_originzc` (multistart_seed_generator.jl:536-547) -- a data-derived "good
starting value" for the restriction's own outer coordinates, not a hand-tuned constant.

Worth knowing when reading a run's trace, confirmed live at real D=4 KNITRO (16/16 cutoffs,
debug_pq_cutoff_sign_isolate.jl EXPERIMENT 2): this point is a local MINIMUM of `Delta*` in every
cutoff coordinate, because the restriction's targets are the fixed constants `1/L` and `1/L^2` and
cutoffs AT the empirical quantiles are exactly where the unweighted draws already come closest to
meeting them. So the cutoff block of the outer gradient is genuinely near zero at the start of a
run, and early outer iterations move mostly in the economic block. That is the expected behaviour,
not a stalled or mis-scaled gradient.
"""
function pairwise_quantile_start_cutoffs(ctx, layout::PairwiseQuantileCutoffLayout)
    D = layout.D; L = layout.L
    raw = zeros(n_raw(layout))
    for o in 1:D
        Uo = sort(@view ctx.U[:, o])
        n = length(Uo)
        q = [Uo[clamp(round(Int, (r / L) * n), 1, n)] for r in 1:L-1]
        b = raw_index(layout, o, 1)
        all(>(0), q) || error("pairwise_quantile_start_cutoffs: non-positive empirical quantile at origin=$o")
        raw[b] = log(q[1])
        for k in 2:L-1
            gap = log(q[k]) - log(q[k-1])
            gap > 0 ||
                error("pairwise_quantile_start_cutoffs: empirical quantiles $(k-1) and $k coincide at " *
                      "origin=$o (gap=$gap) -- the draws are too coarse for L=$L bins at this W.")
            raw[b+k-1] = log(expm1(gap))
        end
    end
    return raw
end

"""
    run_pairwise_quantile_upper_checkpointed(w0=nothing; kwargs...) -> NamedTuple

Checkpointed outer loop for the pairwise-quantile-independence family. See this file's header for
the outer coordinate layout and the no-defaults rule.

Returns a `NamedTuple` carrying (at minimum) `.knitro_status::Int`, `.n_eval`, `.n_grad`, and
`.best` -- either `nothing` or `NamedTuple(gp=..., w=..., Delta=..., n_eval=..., t=...)` -- which is
exactly the shape `paper_upper_v1_orchestrator/family_start_chain.jl` reads
(`:157`, `:166-175`, `:232-238`).
"""
function run_pairwise_quantile_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        # ---- scientific parameters: ALL REQUIRED, no defaults (CLAUDE.md) ----
        find_smallest::Bool,
        W::Int, delta::Float64, draw_design::Symbol, draw_seed::Int,
        σHat::Float64, inner_lower_limit::Float64, z_halfwidth::Float64,
        destination_sample::Symbol,
        exclude_diagonal_gravity::Bool,
        gravity_exclude_cells::AbstractVector{<:Tuple{Int,Int}},
        # ---- this family's own modelling choices: ALSO REQUIRED ----
        L::Int,             # number of quantile bins per origin
        min_crossed::Int,   # cutoff-gradient secant bandwidth, in draws crossed
        # ---- run plumbing ----
        ckpt_dir::AbstractString,
        run_id::String = string(Dates.now()),
        label::String = "pairwise_quantile_upper",
        maxtime_real::Float64 = 180.0,
        opt_file::String = "csw_outer_wallclock_sr1.opt",
        checkpoint_interval_s::Float64 = 90.0,
        resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        # ---- Stage B / R0 fixed-gp restoration, same mechanism as every other family ----
        objective_mode::Symbol = :min_gp,
        gp_fixed::Union{Nothing,Float64} = nothing,
        # ---- solver/backend options (behaviour-preserving defaults, not scientific) ----
        A_coordinate_mode::Symbol = :powered_aspace,
        outer_direct_hessopt::Union{Nothing,Symbol} = nothing,
        pin_outer_algorithm::Bool = false,
        blas_threads::Union{Nothing,Int} = nothing,
        use_exact_cache::Bool = true,
        cutoff_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
        )
    lp(xs...) = (println(xs...); flush(stdout))

    # ---- 1. absolute ckpt_dir BEFORE any real-data setup (a setup file further down the include
    # chain calls cd() as a side effect; a relative path would otherwise resolve against whatever
    # cwd happened to be at each later joinpath) ----
    ckpt_dir = abspath(ckpt_dir)
    mkpath(ckpt_dir)

    # ---- 2. hard-error validation of enumerated kwargs ----
    destination_sample in (:exclude_row, :all_legacy) ||
        error("run_pairwise_quantile_upper_checkpointed($label): destination_sample must be :exclude_row|:all_legacy, got :$destination_sample")
    A_coordinate_mode in (:legacy_z, :powered_aspace) ||
        error("run_pairwise_quantile_upper_checkpointed($label): A_coordinate_mode must be :legacy_z|:powered_aspace, got :$A_coordinate_mode")
    objective_mode in (:min_gp, :min_delta_fixed_gp) ||
        error("run_pairwise_quantile_upper_checkpointed($label): objective_mode must be :min_gp|:min_delta_fixed_gp, got :$objective_mode")
    objective_mode == :min_delta_fixed_gp && gp_fixed === nothing &&
        error("run_pairwise_quantile_upper_checkpointed($label): objective_mode=:min_delta_fixed_gp requires gp_fixed " *
              "(this mode has no meaning with gp free -- it minimizes Delta* AT a fixed gp).")
    L >= 2 || error("run_pairwise_quantile_upper_checkpointed($label): L (quantile bins) must be >= 2, got $L")
    min_crossed >= 1 ||
        error("run_pairwise_quantile_upper_checkpointed($label): min_crossed must be >= 1, got $min_crossed")
    min_crossed < W ||
        error("run_pairwise_quantile_upper_checkpointed($label): min_crossed=$min_crossed must be < W=$W")
    lp("[", label, "] pairwise-quantile-independence family: L=", L, " bins, min_crossed=", min_crossed,
       " of W=", W, "  A_coordinate_mode=", A_coordinate_mode, "  objective_mode=", objective_mode)

    # ---- 4. resume load ----
    resumed = resume_from === nothing ? nothing : load_pairwise_quantile_checkpoint(resume_from)

    # ---- 5. resume-mismatch guards: hard errors, never a silent override ----
    if resumed !== nothing
        resumed.find_smallest == find_smallest ||
            error("run_pairwise_quantile_upper_checkpointed($label): direction MISMATCH on resume -- checkpoint " *
                  "has find_smallest=$(resumed.find_smallest), this call requests $find_smallest.")
        resumed.L == L ||
            error("run_pairwise_quantile_upper_checkpointed($label): L MISMATCH on resume -- checkpoint has L=$(resumed.L), " *
                  "this call requests L=$L -- refusing to resume under a different bin count (the entire " *
                  "moment layout and outer coordinate width depend on it).")
        resumed.destination_sample == destination_sample ||
            error("run_pairwise_quantile_upper_checkpointed($label): destination_sample MISMATCH on resume -- " *
                  "checkpoint has :$(resumed.destination_sample), this call requests :$destination_sample.")
        resumed.sigma == σHat ||
            error("run_pairwise_quantile_upper_checkpointed($label): sigma MISMATCH on resume -- checkpoint has " *
                  "sigma=$(resumed.sigma), this call requests σHat=$σHat.")
        # min_crossed is a gradient-bandwidth choice, not part of the problem's identity: changing it
        # on resume is legitimate (it changes the search's step quality, not what is being solved).
        # Logged rather than refused, deliberately -- unlike every field above it.
        resumed.min_crossed == min_crossed ||
            lp("[", label, "] NOTE: min_crossed differs from checkpoint (", resumed.min_crossed, " -> ",
               min_crossed, "); allowed -- it is a gradient bandwidth, not part of the problem identity.")
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
    end

    # ---- 6. real-data economic context ----
    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design,
        draw_seed = draw_seed, destination_sample = destination_sample,
        exclude_diagonal_gravity = exclude_diagonal_gravity, gravity_exclude_cells = gravity_exclude_cells,
        σHat = σHat, inner_lower_limit = inner_lower_limit)
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
    pe = build_pivot_elimination(ctx)
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

    # ---- 7. coordinate-mode closures ----
    if A_coordinate_mode == :powered_aspace
        isdefined(Main, :cm_fixed_theta) ||
            error("run_pairwise_quantile_upper_checkpointed($label): A_coordinate_mode=:powered_aspace requires " *
                  "cm_aspace_coordinate.jl to be included (cm_fixed_theta/precompute_cm_aspace_xy/cm_z_from_a/cm_a_from_z).")
        theta_cm = cm_fixed_theta(ctx)
        xy_cm = precompute_cm_aspace_xy(ctx)
    else
        theta_cm = NaN
        xy_cm = nothing
    end
    xf_from_w_econ(w_econ) = A_coordinate_mode == :powered_aspace ?
        x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe) :
        x_free_from_w(w_econ, pe)
    zfree_from_w_econ(w_econ) = A_coordinate_mode == :powered_aspace ?
        cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe) : w_econ[2:end]

    # ---- 8. this family's layout + bounds ----
    layout = PairwiseQuantileCutoffLayout(D, L)
    n_cut = n_raw(layout)
    bounds = cutoff_bounds === nothing ? default_raw_cutoff_bounds(ctx, layout) : cutoff_bounds
    length(bounds) == n_cut ||
        error("run_pairwise_quantile_upper_checkpointed($label): cutoff_bounds length=$(length(bounds)) != n_raw(layout)=$n_cut")
    D2_econ = D * Ddest
    lp("[", label, "] outer layout: D2_econ=", D2_econ, " (gp + ", D2_econ - 1, " zfree) + n_raw=", n_cut,
       " cutoff coords = ", D2_econ + n_cut, " total")

    # ---- 9. (no profiled/eliminated-coordinate handling: this family has no analog of origin-ZC's
    #         focal k=(sigma-1) row omission -- its restriction rows are bin indicators, none of
    #         which is exactly collinear with the autarky/counterfactual moment.) ----

    # ---- 10. w0 reconstruction on resume, else require a fresh w0 ----
    if resumed !== nothing
        resumed.D == D ||
            error("run_pairwise_quantile_upper_checkpointed($label): D MISMATCH on resume -- checkpoint D=$(resumed.D), current D=$D.")
        (ctx.draw_meta.checksum_uniform == resumed.draw_checksum_uniform &&
         ctx.draw_meta.checksum_transformed == resumed.draw_checksum_transformed) ||
            error("run_pairwise_quantile_upper_checkpointed($label): draw checksum MISMATCH on resume -- regenerated " *
                  "draws (design=:$(draw_design), seed=$(draw_seed)) do not match the checkpoint's recorded checksums.")
        length(resumed.raw_cutoffs) == n_cut ||
            error("run_pairwise_quantile_upper_checkpointed($label): checkpoint raw_cutoffs length=$(length(resumed.raw_cutoffs)) != n_raw=$n_cut")
        # resumed.zfree is ALWAYS canonical z-space regardless of which A_coordinate_mode wrote it,
        # so resuming under a different coordinate mode is safe -- same reasoning as
        # run_originzc_upper_checkpointed's identical block.
        resumed.A_coordinate_mode == A_coordinate_mode ||
            lp("[", label, "] A_coordinate_mode differs from checkpoint (:", resumed.A_coordinate_mode,
               " -> :", A_coordinate_mode, ") -- safe (checkpoint zfree is canonical z-space), reconstructing w0.")
        A_native0 = A_coordinate_mode == :powered_aspace ? cm_a_from_z(resumed.zfree, theta_cm, xy_cm, pe) : resumed.zfree
        w0 = vcat(resumed.g, A_native0, resumed.raw_cutoffs)
    elseif w0 === nothing
        error("run_pairwise_quantile_upper_checkpointed($label): w0 required for a fresh (non-resumed) run -- must be " *
              "vcat(gp, zfree, raw_cutoffs) with length(raw_cutoffs)==$n_cut. " *
              "pairwise_quantile_start_cutoffs(ctx, layout) builds the canonical cutoff block.")
    end
    length(w0) == D2_econ + n_cut ||
        error("run_pairwise_quantile_upper_checkpointed($label): length(w0)=$(length(w0)) != D2_econ+n_raw=$(D2_econ + n_cut)")

    # ---- 11. production context (hardcoded :operator bundle via prepare_production_run) ----
    prepared = prepare_production_run(:pairwise_quantile, "run_pairwise_quantile_upper_checkpointed",
        () -> build_pairwise_quantile_production_context(ctx, layout; min_crossed = min_crossed))
    pcx = prepared.ctx.inner
    pcx = with_screen_counters(pcx)
    ctx_cm = pcx.ctx_cm
    exact_cache = use_exact_cache ? cm_production_exact_cache() : nothing
    blas_threads !== nothing && BLAS.set_num_threads(blas_threads)
    print_active_layout_banner(ctx, "pairwise_quantile")
    print_screen_startup_banner("pairwise_quantile")
    th = ctx_cm.obj.threshold_state
    println("[threshold-config] mode=pairwise_quantile requested_delta=", delta,
            " resolved_active_threshold=", th.threshold)
    flush(stdout)
    write_backend_manifest_atomic(prepared.manifest, joinpath(ckpt_dir, "$(label)_backend_manifest.json"))
    econ_ws = get_or_build_econ_a_grad_ws(W)

    # ---- 12. outer variable bounds ----
    D2 = length(w0)
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo = vcat(gp_lo, w0[2:D2_econ] .- z_halfwidth, [bounds[k][1] for k in 1:n_cut])
    w_hi = vcat(gp_hi, w0[2:D2_econ] .+ z_halfwidth, [bounds[k][2] for k in 1:n_cut])
    if gp_fixed !== nothing
        gp_lo <= gp_fixed <= gp_hi ||
            error("run_pairwise_quantile_upper_checkpointed($label): gp_fixed=$gp_fixed outside the gp box [$gp_lo, $gp_hi]")
        isapprox(w0[1], gp_fixed; atol = 1e-10) ||
            error("run_pairwise_quantile_upper_checkpointed($label): gp_fixed=$gp_fixed but w0[1]=$(w0[1]) -- the caller " *
                  "must construct w0 with the SAME pinned gp, not rely on KNITRO to move it there.")
        w_lo[1] = gp_fixed
        w_hi[1] = gp_fixed
    end
    all(w_lo .<= w0 .<= w_hi) ||
        error("run_pairwise_quantile_upper_checkpointed($label): w0 violates its own box at coordinate(s) " *
              "$(findall(.!(w_lo .<= w0 .<= w_hi))) -- refusing to start KNITRO from an infeasible point.")

    # ---- 13. KNITRO outer setup ----
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    if outer_direct_hessopt !== nothing
        pin_outer_algorithm && error("run_pairwise_quantile_upper_checkpointed($label): outer_direct_hessopt and pin_outer_algorithm are mutually exclusive.")
        outer_direct_hessopt in (:sr1, :bfgs) || error("run_pairwise_quantile_upper_checkpointed($label): outer_direct_hessopt must be :sr1|:bfgs, got :$outer_direct_hessopt")
        set_outer_algorithm_direct!(kc, outer_direct_hessopt === :sr1 ? KNITRO_HESSOPT_SR1 : KNITRO_HESSOPT_BFGS)
    end
    pin_outer_algorithm && set_production_outer_algorithm!(kc)
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = objective_mode == :min_gp ? KNITRO.KN_add_cons(kc, 1) : Int32[]
    objective_mode == :min_gp && KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    # ---- 14. mutable outer-loop state ----
    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    n_grad = Ref(resumed !== nothing ? resumed.n_grad : 0)
    trace = NamedTuple[]
    bandwidth_cache = resumed !== nothing ? copy(resumed.bandwidth_cache) : Dict{Int,Float64}()
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    # ---- 15. checkpoint closure ----
    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = zfree_from_w_econ(w_current[1:D2_econ])   # ALWAYS canonical z-space
        cut_now = w_current[D2_econ+1:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = PairwiseQuantileCheckpointV1(PAIRWISE_QUANTILE_CHECKPOINT_SCHEMA_V1, run_id, label,
            (find_smallest ? :pq_upper : :pq_lower), find_smallest, delta,
            W, draw_seed, draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            L, min_crossed, n_cut, D,
            w_current[1], collect(zfree_now), collect(cut_now), logA_full, copy(ctx_cm.obj.x),
            copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version,
            destination_sample, ctx.row_idx, Ddest, A_coordinate_mode, ctx.σ)
        save_pairwise_quantile_checkpoint(joinpath(ckpt_dir, "$(label)_latest.jls"), ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    # ---- 16. objective/constraint callback ----
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        cuts = collect(w[D2_econ+1:end])
        local base, verify
        # The shared CMProductionEvalKey's `nu` slot carries this family's raw cutoff vector and its
        # `L` slot the bin count -- both are generic "the restriction's own outer parameters"
        # fields, and `family_tag=:pairwise_quantile` plus a FRESH per-run cache prevent any
        # cross-family collision. K_mean/K_pair/contrasts have no meaning here and take their
        # off sentinels.
        cache_key = exact_cache === nothing ? nothing :
            CMProductionEvalKey(collect(xf), cuts, delta, find_smallest, ctx_cm.obj.inner_loop_opt,
                :pairwise_quantile, L, :none, 0, 0, A_coordinate_mode, context_fingerprint(ctx_cm))
        try
            base, verify = cm_cache_lookup_or_compute!(exact_cache, cache_key, () -> begin
                _, b, v = pairwise_quantile_production_value_verified_screened(xf, cuts, pcx;
                    counters = pcx.screen_counters, eval_id = n_eval[])
                return b, v
            end)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_pairwise_quantile_upper_checkpointed($label): infeasible/failed inner solve at this point")
        end
        Δ = verify.Delta_dual
        if objective_mode == :min_gp
            evalResult.obj[1] = find_smallest ? w[1] : -w[1]
            evalResult.c[1] = Δ
        else
            evalResult.obj[1] = Δ
        end
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base, verify = verify)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        verified = is_verified_success(verify)
        is_new_best = if objective_mode == :min_gp
            feasible && verified && is_better_polish(w[1], best_feasible[] === nothing ? nothing : best_feasible[].gp, find_smallest)
        else
            verified && is_better_profile(Δ, best_feasible[] === nothing ? nothing : best_feasible[].Delta)
        end
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = prior_wall + (time() - t_start))
            do_checkpoint(:new_best, collect(w))
        elseif time() - last_ckpt_t[] >= checkpoint_interval_s
            do_checkpoint(:wall_interval, collect(w))
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible,
                      verified = verified,
                      max_marginal_cum_resid = get(verify, :max_marginal_cumulative_residual, NaN),
                      max_joint_cum_resid = get(verify, :max_cumulative_residual, NaN)))
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            lp("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1],
               " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        return 0
    end

    # ---- 17. gradient callback ----
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        cuts = collect(w[D2_econ+1:end])
        shared = last_F_state[]
        matched = shared !== nothing && shared.w == w
        base = matched ? shared.base : nothing
        verify_c = matched ? shared.verify : nothing
        gfull, meta = pairwise_quantile_production_gradient(xf, cuts, pcx, ctx, pe;
            base = base, verify = verify_c, econ_ws = econ_ws, min_crossed = min_crossed,
            threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        n_grad[] += 1
        # gfull's economic block is ALWAYS the z-space gradient; rescale it by the constant scalar
        # -theta_cm when the outer search is actually in a-space -- same block as every other
        # family's driver. The cutoff block (indices D2_econ+1:end) is coordinate-mode-INDEPENDENT
        # by construction (the cutoff transform has nothing to do with the A-space reparametrization)
        # and is deliberately NOT rescaled.
        if A_coordinate_mode == :powered_aspace
            gfull[2:D2_econ] .*= -theta_cm
        end
        if objective_mode == :min_gp
            evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
            evalResult.jac .= gfull
        else
            evalResult.objGrad .= gfull
        end
        return 0
    end

    # ---- 18. register callbacks ----
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    if objective_mode == :min_gp
        KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)
    else
        KNITRO.KN_set_cb_grad(kc, cb, cb_G!)
    end

    # ---- 19. algorithm-pin assertions ----
    if outer_direct_hessopt !== nothing
        assert_outer_algorithm_direct!(kc, outer_direct_hessopt === :sr1 ? KNITRO_HESSOPT_SR1 : KNITRO_HESSOPT_BFGS;
            context = "run_pairwise_quantile_upper_checkpointed($label)")
    end
    pin_outer_algorithm && assert_outer_algorithm_explicit!(kc; context = "run_pairwise_quantile_upper_checkpointed($label)")

    # ---- 20. solve ----
    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    # ---- 21. derived summary scalar ----
    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))

    # ---- 22. final re-verification at the solution ----
    xsol_v = collect(xsol)
    local verify_final
    try
        _, _, verify_final = pairwise_quantile_production_value_verified_screened(
            xf_from_w_econ(xsol_v[1:D2_econ]), xsol_v[D2_econ+1:end], pcx; counters = pcx.screen_counters)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        verify_final = (inner_status = -300,)
    end
    final_ckpt = if is_verified_success(verify_final)
        do_checkpoint(:stage_complete, xsol_v)
    else
        lp("[", label, "] WARNING: terminal point failed verification -- checkpointing as :stage_complete_unverified.")
        do_checkpoint(:stage_complete_unverified, xsol_v)
    end
    print_screen_summary(pcx; label = label)

    # ---- 23. return ----
    return (knitro_status = nStatus, wall = wall_ext, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = xsol_v, trace = trace, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"),
            L = L, min_crossed = min_crossed, n_raw = n_cut,
            screen_summary = as_namedtuple(pcx.screen_counters))
end

"""
    run_pairwise_quantile_lower_checkpointed(w0=nothing; label="pairwise_quantile_lower", kwargs...)

Thin wrapper with `find_smallest=false` (the lower-kappa direction, per direction_bounds.jl's own
audit), mirroring `run_originzc_lower_checkpointed`. Forwards every other argument unchanged.
"""
function run_pairwise_quantile_lower_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        label::String = "pairwise_quantile_lower", kwargs...)
    return run_pairwise_quantile_upper_checkpointed(w0; find_smallest = false, label = label, kwargs...)
end
