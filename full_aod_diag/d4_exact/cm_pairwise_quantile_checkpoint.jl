# ================================================================================================
# Checkpointed OUTER driver for the CM + pairwise-quantile family (family #7, 2026-08-12) -- the
# entry point the five-family campaign orchestrator calls.
#
# Ported from `pairwise_quantile_checkpoint.jl` (version B) section for section. Read the two side
# by side; every structural block below has the same number and the same job. FOUR differences, all
# of them consequences of what this family IS rather than of how it is wired:
#
#   1. OUTER COORDINATES: `w = vcat(gp, zfree, raw_masses)` with `length(raw_masses) == L-1`, not
#      `D*(L-1)`. At D=20/L=5 that is 4 mass coordinates instead of 80 -- the entire reason the
#      family exists, and the one number to sanity-check in a run's opening banner.
#   2. NO `cutoff_source`. The standalone family chooses where its fixed cutoffs sit; this family
#      does not get to choose -- the cutoffs are SELECTED from CM's own threshold array so that each
#      PQ bin is a union of CM grid cells, which is what makes the dropped per-origin marginal rows
#      exactly implied. `:empirical_quantile` cutoffs would put each origin's cutoffs at its own
#      empirical quantiles, which are not CM grid points and differ across origins, so the superset
#      identity fails and a shared `mu` is not even well defined. The checkpoint therefore records
#      the CM grid (`G`, `n_families`, `contrasts`) and the DERIVED cutoffs, not a source symbol.
#   3. THE INNER OPTION FILE IS REQUIRED. `ek_inner_cmpq.opt` (`hessopt exact`) is not a tuning
#      preference here: measured at production `n_x=412`, the FG-only path hits `-400` after 16,734
#      evaluations while the exact-Hessian path reaches `nStatus=0` in twelve. Passing a non-exact
#      file is now a hard error inside `inner_loop_KNITRO_cmpairwisequantile_operator` itself.
#   4. `gradient_backend` defaults to `:cplus`, the factorized economic-block representation
#      (`cm_pairwise_quantile_cplus.jl`), exactly as the standalone family's driver does. `:dense`
#      remains available and is the reference the C+ gate compares against.
#
# NO SCIENTIFIC PARAMETER IS DEFAULTED. Every kwarg that changes what economic problem is solved is
# a bare Julia keyword, so omitting it raises `UndefKeywordError` before the body runs (CLAUDE.md).
# ================================================================================================

using Serialization: serialize, deserialize
using LinearAlgebra: BLAS
import Dates

"""
    CMPairwiseQuantileCheckpointV1

This family's checkpoint schema. The NAME is deliberately unique across the whole repo, and that is
not cosmetic: Julia's `Serialization.deserialize` resolves types by NAME, and this codebase has
already been bitten once by two unrelated `struct`s sharing one (memory
`feedback-julia-serialization-type-name-collision`). Verified unique by grep before adding.

RECORDS THE RESTRICTION'S IDENTITY, not just how it was requested. `L`/`G`/`n_families`/`contrasts`
say what was asked for; `cm_thresholds` and `pq_cutoffs` are the numbers that request produced
against these exact draws. Both are needed -- the request alone does not pin the numbers, and the
numbers alone do not record the intent -- and both are re-checked bit-identically on resume, because
different cutoffs are a DIFFERENT restriction, not a different search over the same one.
"""
struct CMPairwiseQuantileCheckpointV1
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
    L::Int                       # PQ bins
    G::Int                       # CM grid size (cm_grid_size); CM levels are G-1
    n_families::Int              # 1 = eq.35 only, 2 = eq.35 + eq.36
    contrasts::Symbol
    cm_thresholds::Vector{Float64}
    pq_cutoffs::Matrix{Float64}
    min_bin_count::Int
    mass_start::Symbol
    n_raw::Int
    D::Int
    # ---- outer point ----
    g::Float64
    zfree::Vector{Float64}
    raw_masses::Vector{Float64}
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

const CM_PAIRWISE_QUANTILE_CHECKPOINT_SCHEMA_V1 = 1

"Atomic-ish checkpoint write: serialize to `<path>.tmp`, then `mv` onto `path`. Same discipline as
every other checkpoint in this codebase -- plain `Serialization`, not JLD2/JSON."
function save_cm_pairwise_quantile_checkpoint(path::AbstractString, ckpt::CMPairwiseQuantileCheckpointV1)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Loader. No upgrade chain by design: this is schema 1, there is no earlier schema of THIS checkpoint
to upgrade from, and a standalone-PQ checkpoint is not a degraded version of it (its outer mass
block is `D*(L-1)` long and means something different), so a foreign type is rejected outright."
function load_cm_pairwise_quantile_checkpoint(path::AbstractString)
    ckpt = deserialize(path)
    ckpt isa CMPairwiseQuantileCheckpointV1 ||
        error("load_cm_pairwise_quantile_checkpoint($path): deserialized a $(typeof(ckpt)), not a " *
              "CMPairwiseQuantileCheckpointV1 -- refusing to resume from a foreign checkpoint.")
    return ckpt
end

"""
    run_cm_pairwise_quantile_upper_checkpointed(w0=nothing; kwargs...) -> NamedTuple

Checkpointed outer loop for the CM + pairwise-quantile family.

Returns a `NamedTuple` carrying `.knitro_status::Int`, `.n_eval`, `.n_grad` and `.best` -- either
`nothing` or `NamedTuple(gp=..., w=..., Delta=..., n_eval=..., t=...)` -- which is exactly the shape
`paper_upper_v1_orchestrator/family_start_chain.jl` reads.

`w0` is `vcat(gp, zfree, raw_masses)` with `length(raw_masses) == L-1`.
`cmpq_uniform_mass_raw(L)` builds the canonical `mu = 1/L` block; the context's own
`raw_start` (from `mass_start`) is the data-derived alternative.
"""
function run_cm_pairwise_quantile_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        # ---- scientific parameters: ALL REQUIRED, no defaults (CLAUDE.md) ----
        find_smallest::Bool,
        W::Int, delta::Float64, draw_design::Symbol, draw_seed::Int,
        σHat::Float64, inner_lower_limit::Float64, z_halfwidth::Float64,
        destination_sample::Symbol,
        exclude_diagonal_gravity::Bool,
        gravity_exclude_cells::AbstractVector{<:Tuple{Int,Int}},
        # ---- this family's own modelling choices: ALSO REQUIRED ----
        L::Int,                    # PQ quantile bins on the SHARED reference marginal
        cm_grid_size::Int,         # CM's grid size G; L must divide it
        cm_moment_families::Int,   # 1 (eq.35) or 2 (eq.35 + eq.36)
        contrasts::Symbol,         # :anchored | :orthonormal
        min_bin_count::Int,        # non-degeneracy floor per marginal bin AND per joint cell
        mass_start::Symbol,        # :uniform | :empirical -- the canonical starting masses
        # ---- inner solver: REQUIRED, see this file's header difference #3 ----
        inner_opt::AbstractString,
        # ---- run plumbing ----
        ckpt_dir::AbstractString,
        run_id::String = string(Dates.now()),
        label::String = "cm_pairwise_quantile_upper",
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
        mass_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
        # Economic-block backend for the OUTER gradient. `:cplus` is the FACTORIZED representation
        # (lfix_factorized.jl) that never materializes the W x D x Ddest price tensors; `:dense` is
        # the older path, kept as the reference `test_cm_pairwise_quantile_cplus_gate.jl` compares
        # against. NOT a scientific parameter -- it changes how the same gradient is computed, not
        # what problem is solved -- so it carries a default, like A_coordinate_mode.
        gradient_backend::Symbol = :cplus,
        )
    lp(xs...) = (println(xs...); flush(stdout))

    # ---- 1. absolute ckpt_dir BEFORE any real-data setup (a setup file further down the include
    # chain calls cd() as a side effect) ----
    ckpt_dir = abspath(ckpt_dir)
    mkpath(ckpt_dir)

    # ---- 2. hard-error validation of enumerated kwargs ----
    destination_sample in (:exclude_row, :all_legacy) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): destination_sample must be :exclude_row|:all_legacy, got :$destination_sample")
    A_coordinate_mode in (:legacy_z, :powered_aspace) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): A_coordinate_mode must be :legacy_z|:powered_aspace, got :$A_coordinate_mode")
    objective_mode in (:min_gp, :min_delta_fixed_gp) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): objective_mode must be :min_gp|:min_delta_fixed_gp, got :$objective_mode")
    gradient_backend in (:cplus, :dense) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): gradient_backend must be :cplus|:dense, got :$gradient_backend")
    objective_mode == :min_delta_fixed_gp && gp_fixed === nothing &&
        error("run_cm_pairwise_quantile_upper_checkpointed($label): objective_mode=:min_delta_fixed_gp requires gp_fixed.")
    L >= 2 || error("run_cm_pairwise_quantile_upper_checkpointed($label): L (PQ bins) must be >= 2, got $L")
    # `L | G` is the family's defining structural condition (each PQ bin a union of CM grid cells).
    # `resolve_cm_pairwise_quantile_config` also enforces it with the full explanation; checked here
    # too so a campaign arm fails at argument-validation time rather than after a real-data setup.
    cm_grid_size % L == 0 ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): L=$L does not divide " *
              "cm_grid_size=$cm_grid_size. Each PQ bin must be a UNION of CM grid cells, which is " *
              "what makes the dropped per-origin marginal rows exactly implied; without it the " *
              "family is not the family.")
    cm_moment_families in (1, 2) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): cm_moment_families must be 1|2, got $cm_moment_families")
    contrasts in (:anchored, :orthonormal) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): contrasts must be :anchored|:orthonormal, got :$contrasts")
    mass_start in (:uniform, :empirical) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): mass_start must be :uniform|:empirical, got :$mass_start")
    min_bin_count >= 1 ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): min_bin_count must be >= 1, got $min_bin_count")
    min_bin_count * L^2 <= W ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): min_bin_count=$min_bin_count with " *
              "L=$L needs at least $(min_bin_count * L^2) draws for the JOINT cells, but W=$W -- the " *
              "floor is unsatisfiable by arithmetic before any data is looked at. (This family gates " *
              "joint cells as well as marginal bins, so the binding count is L^2, not L.)")
    inner_opt_path = isabspath(inner_opt) ? String(inner_opt) :
        joinpath(D4X_ROOT, "full_aod_diag", String(inner_opt))
    isfile(inner_opt_path) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): inner_opt=$inner_opt resolves to " *
              "$inner_opt_path, which does not exist")
    # And it must actually request an exact Hessian. `inner_loop_KNITRO_cmpairwisequantile_operator`
    # already refuses this combination, but it only gets the chance from INSIDE the outer KNITRO
    # callback -- after a ~80s real-data context build, and where a thrown Julia error may surface as
    # an opaque KN_RC_CALLBACK_ERR instead of this message (memory
    # `feedback-archC-verified-state-direct-call-knitro-callback-err`). Checked here instead, in
    # milliseconds, using KNITRO's OWN parser on a throwaway context rather than by grepping the
    # file: the .opt format has comments and aliases, and a textual check would be the kind of
    # almost-right validation that passes when it should not.
    let kc_probe = KNITRO.KN_new()
        try
            KNITRO.KN_load_param_file(kc_probe, inner_opt_path)
            KNITRO.KN_get_int_param(kc_probe, "hessopt") == 1 ||
                error("run_cm_pairwise_quantile_upper_checkpointed($label): inner_opt=$inner_opt does " *
                      "not request hessopt=exact. This family's inner solve does not converge at " *
                      "production n without the exact Hessian -- measured at n_x=412, the FG-only " *
                      "path reached -400 after 16,734 evaluations while the exact-Hessian path " *
                      "reached nStatus=0 in twelve. Pass ek_inner_cmpq.opt.")
        finally
            KNITRO.KN_free(kc_probe)
        end
    end
    lp("[", label, "] CM + pairwise-quantile family: L=", L, " PQ bins on a SHARED reference marginal, ",
       "CM grid G=", cm_grid_size, " (", cm_grid_size - 1, " levels, ", cm_moment_families,
       " family/families, contrasts=:", contrasts, "), min_bin_count=", min_bin_count, " of W=", W,
       "  mass_start=:", mass_start, "  A_coordinate_mode=", A_coordinate_mode,
       "  objective_mode=", objective_mode)
    lp("[", label, "] inner KNITRO opt file = ", inner_opt_path)

    # ---- 4. resume load ----
    resumed = resume_from === nothing ? nothing : load_cm_pairwise_quantile_checkpoint(resume_from)

    # ---- 5. resume-mismatch guards: hard errors, never a silent override ----
    if resumed !== nothing
        resumed.find_smallest == find_smallest ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): direction MISMATCH on resume.")
        resumed.L == L ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): L MISMATCH on resume -- checkpoint L=$(resumed.L), " *
                  "requested L=$L. The entire moment layout and outer coordinate width depend on it.")
        resumed.G == cm_grid_size ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): cm_grid_size MISMATCH on resume -- " *
                  "checkpoint G=$(resumed.G), requested $cm_grid_size. A different CM grid is a different restriction.")
        resumed.n_families == cm_moment_families ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): cm_moment_families MISMATCH on resume -- " *
                  "checkpoint $(resumed.n_families), requested $cm_moment_families.")
        resumed.contrasts == contrasts ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): contrasts MISMATCH on resume -- " *
                  "checkpoint :$(resumed.contrasts), requested :$contrasts.")
        resumed.destination_sample == destination_sample ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): destination_sample MISMATCH on resume.")
        resumed.sigma == σHat ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): sigma MISMATCH on resume -- checkpoint " *
                  "sigma=$(resumed.sigma), requested σHat=$σHat.")
        # min_bin_count and mass_start are NOT part of the problem's identity: the first is a
        # validation floor, the second only chooses where a fresh run STARTS (and a resume does not
        # start fresh). Logged rather than refused, unlike every field above.
        resumed.min_bin_count == min_bin_count ||
            lp("[", label, "] NOTE: min_bin_count differs from checkpoint (", resumed.min_bin_count,
               " -> ", min_bin_count, "); allowed -- a validation floor, not part of the identity.")
        resumed.mass_start == mass_start ||
            lp("[", label, "] NOTE: mass_start differs from checkpoint (:", resumed.mass_start, " -> :",
               mass_start, "); allowed -- it only chooses a FRESH run's start, and this is a resume.")
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
    end

    # ---- 6. real-data economic context ----
    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest,
        draw_design = draw_design, draw_seed = draw_seed, destination_sample = destination_sample,
        exclude_diagonal_gravity = exclude_diagonal_gravity, gravity_exclude_cells = gravity_exclude_cells,
        σHat = σHat, inner_lower_limit = inner_lower_limit, inner_loop_opt = inner_opt_path)
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
    pe = build_pivot_elimination(ctx)
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

    # ---- 7. coordinate-mode closures ----
    if A_coordinate_mode == :powered_aspace
        isdefined(Main, :cm_fixed_theta) ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): A_coordinate_mode=:powered_aspace requires " *
                  "cm_aspace_coordinate.jl to be included.")
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

    # ---- 8. this family's outer layout: ONE shared simplex ----
    n_cut = n_cmpq_raw(L)
    D2_econ = D * Ddest
    lp("[", label, "] outer layout: D2_econ=", D2_econ, " (gp + ", D2_econ - 1, " zfree) + n_raw=",
       n_cut, " shared mass coords = ", D2_econ + n_cut, " total  (the standalone pairwise-quantile ",
       "family would need ", (L - 1) * D, " mass coords here)")

    # ---- 10. w0 reconstruction on resume, else require a fresh w0 ----
    if resumed !== nothing
        resumed.D == D ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): D MISMATCH on resume -- checkpoint D=$(resumed.D), current D=$D.")
        (ctx.draw_meta.checksum_uniform == resumed.draw_checksum_uniform &&
         ctx.draw_meta.checksum_transformed == resumed.draw_checksum_transformed) ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): draw checksum MISMATCH on resume -- regenerated " *
                  "draws (design=:$(draw_design), seed=$(draw_seed)) do not match the checkpoint's recorded checksums.")
        length(resumed.raw_masses) == n_cut ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): checkpoint raw_masses length=$(length(resumed.raw_masses)) != n_raw=$n_cut")
        resumed.A_coordinate_mode == A_coordinate_mode ||
            lp("[", label, "] A_coordinate_mode differs from checkpoint (:", resumed.A_coordinate_mode,
               " -> :", A_coordinate_mode, ") -- safe (checkpoint zfree is canonical z-space), reconstructing w0.")
        A_native0 = A_coordinate_mode == :powered_aspace ? cm_a_from_z(resumed.zfree, theta_cm, xy_cm, pe) : resumed.zfree
        w0 = vcat(resumed.g, A_native0, resumed.raw_masses)
    elseif w0 === nothing
        error("run_cm_pairwise_quantile_upper_checkpointed($label): w0 required for a fresh (non-resumed) run -- " *
              "must be vcat(gp, zfree, raw_masses) with length(raw_masses)==$n_cut. " *
              "cmpq_uniform_mass_raw(L) builds the canonical mu=1/L block.")
    end
    length(w0) == D2_econ + n_cut ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): length(w0)=$(length(w0)) != D2_econ+n_raw=$(D2_econ + n_cut)")

    # ---- 11. production context (hardcoded :operator bundle via prepare_production_run) ----
    cfg = CMPairwiseQuantileConfig(L = L, cm_grid_size = cm_grid_size,
                                   cm_moment_families = cm_moment_families, contrasts = contrasts,
                                   min_bin_count = min_bin_count, mass_start = mass_start)
    prepared = prepare_production_run(:cm_pairwise_quantile, "run_cm_pairwise_quantile_upper_checkpointed",
        () -> build_cm_pairwise_quantile_production_context(ctx, cfg; inner_opt = inner_opt_path))
    pcx = prepared.ctx.inner
    pcx = with_screen_counters(pcx)
    ctx_cm = pcx.ctx_cm
    cmpq = pcx.cmpq
    exact_cache = use_exact_cache ? cm_production_exact_cache() : nothing
    blas_threads !== nothing && BLAS.set_num_threads(blas_threads)
    print_active_layout_banner(ctx, "cm_pairwise_quantile")
    print_screen_startup_banner("cm_pairwise_quantile")
    th = ctx_cm.obj.threshold_state
    lp("[threshold-config] mode=cm_pairwise_quantile requested_delta=", delta,
       " resolved_active_threshold=", th.threshold)
    write_backend_manifest_atomic(prepared.manifest, joinpath(ckpt_dir, "$(label)_backend_manifest.json"))
    econ_ws = get_or_build_econ_a_grad_ws(W)
    # Backend C+ workspaces: built ONCE for the whole run, never per gradient call -- that reuse is
    # the entire point of the backend.
    cplus_pool, cplus_ws = gradient_backend === :cplus ?
        cm_pairwise_quantile_cplus_workspaces(ctx_cm) : (nothing, nothing)
    lp("[", label, "] outer-gradient economic backend: :", gradient_backend)

    # ---- 11a. the DERIVED cutoffs: report them, and on resume verify them EXACTLY ----
    z_cm = cmpq.z_cm
    Qfixed = cmpq.op.Q
    bin_counts = cmpq.bin_counts
    lp("[", label, "] CM thresholds (k/G grid, ", length(z_cm), " levels): first=", round(z_cm[1], digits = 8),
       " last=", round(z_cm[end], digits = 8),
       "   PQ z-cutoffs, origin 1 = ", round.(Qfixed[:, 1], digits = 6))
    lp("[", label, "] bin gates: (draw,origin) cells cross-checked=", cmpq.gates.bin_cells_checked,
       ", cutoffs bit-identical to CM's own array=", cmpq.gates.cutoffs_bit_identical,
       ", min marginal=", cmpq.gates.min_marginal_count, ", min joint=", cmpq.gates.min_joint_count)
    if resumed !== nothing
        (length(resumed.cm_thresholds) == length(z_cm) && all(resumed.cm_thresholds .== z_cm)) ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): CM THRESHOLD MISMATCH on resume -- " *
                  "the grid regenerated here is not bit-identical to the checkpoint's.")
        (size(resumed.pq_cutoffs) == size(Qfixed) && all(resumed.pq_cutoffs .== Qfixed)) ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): PQ CUTOFF MISMATCH on resume -- " *
                  "different cutoffs define a DIFFERENT restriction; refusing to resume across them.")
    end

    # ---- 12. outer variable bounds ----
    # The mass box is DATA-DERIVED from the bins' own occupancy, so it can only be built now, after
    # the context has decoded them.
    bounds = mass_bounds === nothing ? cmpq.raw_bounds : mass_bounds
    length(bounds) == n_cut ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): mass_bounds length=$(length(bounds)) != n_raw=$n_cut")
    D2 = length(w0)
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo = vcat(gp_lo, w0[2:D2_econ] .- z_halfwidth, [bounds[k][1] for k in 1:n_cut])
    w_hi = vcat(gp_hi, w0[2:D2_econ] .+ z_halfwidth, [bounds[k][2] for k in 1:n_cut])
    if gp_fixed !== nothing
        gp_lo <= gp_fixed <= gp_hi ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): gp_fixed=$gp_fixed outside the gp box [$gp_lo, $gp_hi]")
        isapprox(w0[1], gp_fixed; atol = 1e-10) ||
            error("run_cm_pairwise_quantile_upper_checkpointed($label): gp_fixed=$gp_fixed but w0[1]=$(w0[1]) -- the " *
                  "caller must construct w0 with the SAME pinned gp, not rely on KNITRO to move it there.")
        w_lo[1] = gp_fixed; w_hi[1] = gp_fixed
    end
    all(w_lo .<= w0 .<= w_hi) ||
        error("run_cm_pairwise_quantile_upper_checkpointed($label): w0 violates its own box at coordinate(s) " *
              "$(findall(.!(w_lo .<= w0 .<= w_hi))) -- refusing to start KNITRO from an infeasible point.")

    # ---- 13. KNITRO outer setup ----
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    if outer_direct_hessopt !== nothing
        pin_outer_algorithm && error("run_cm_pairwise_quantile_upper_checkpointed($label): outer_direct_hessopt and pin_outer_algorithm are mutually exclusive.")
        outer_direct_hessopt in (:sr1, :bfgs) || error("run_cm_pairwise_quantile_upper_checkpointed($label): outer_direct_hessopt must be :sr1|:bfgs, got :$outer_direct_hessopt")
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
        mass_now = w_current[D2_econ+1:end]
        logA_full = pivot_expand(zfree_now, pe)
        ckpt = CMPairwiseQuantileCheckpointV1(CM_PAIRWISE_QUANTILE_CHECKPOINT_SCHEMA_V1, run_id, label,
            (find_smallest ? :cmpq_upper : :cmpq_lower), find_smallest, delta,
            W, draw_seed, draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            L, cm_grid_size, cm_moment_families, contrasts, copy(z_cm), copy(Qfixed),
            min_bin_count, mass_start, n_cut, D,
            w_current[1], collect(zfree_now), collect(mass_now), logA_full, copy(ctx_cm.obj.x),
            copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version,
            destination_sample, ctx.row_idx, Ddest, A_coordinate_mode, ctx.σ)
        save_cm_pairwise_quantile_checkpoint(joinpath(ckpt_dir, "$(label)_latest.jls"), ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    # ---- 16. objective/constraint callback ----
    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        masses = collect(w[D2_econ+1:end])
        local base, verify
        # The shared CMProductionEvalKey's `nu` slot carries this family's raw MASS vector and its
        # `L` slot the PQ bin count; `contrasts` and `K_mean` carry the CM grid's own identity, so
        # two arms differing only in G or in family count cannot collide in one cache. `family_tag`
        # plus a FRESH per-run cache prevent any cross-family collision.
        cache_key = exact_cache === nothing ? nothing :
            CMProductionEvalKey(collect(xf), masses, delta, find_smallest, ctx_cm.obj.inner_loop_opt,
                :cm_pairwise_quantile, L, contrasts, cm_grid_size, cm_moment_families,
                A_coordinate_mode, context_fingerprint(ctx_cm))
        try
            base, verify = cm_cache_lookup_or_compute!(exact_cache, cache_key, () -> begin
                _, b, v = cm_pairwise_quantile_production_value_verified_screened(xf, masses, pcx;
                    counters = pcx.screen_counters, eval_id = n_eval[])
                return b, v
            end)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_cm_pairwise_quantile_upper_checkpointed($label): infeasible/failed inner solve at this point")
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
                      max_level_cum_resid = get(verify, :max_level_cumulative_residual, NaN),
                      max_implied_cum_resid = get(verify, :max_implied_cumulative_residual, NaN),
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
        masses = collect(w[D2_econ+1:end])
        shared = last_F_state[]
        matched = shared !== nothing && shared.w == w
        base = matched ? shared.base : nothing
        verify_c = matched ? shared.verify : nothing
        gfull, meta = if gradient_backend === :cplus
            cm_pairwise_quantile_production_gradient_cplus(xf, masses, pcx, ctx, pe, cplus_pool,
                cplus_ws; base = base, verify = verify_c,
                threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        else
            cm_pairwise_quantile_production_gradient(xf, masses, pcx, ctx, pe;
                base = base, verify = verify_c, econ_ws = econ_ws,
                threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        end
        n_grad[] += 1
        # gfull's economic block is ALWAYS the z-space gradient; rescale by the constant -theta_cm
        # when the outer search is in a-space. The mass block is coordinate-mode-INDEPENDENT by
        # construction (the simplex transform has nothing to do with the A-space reparametrization)
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
            context = "run_cm_pairwise_quantile_upper_checkpointed($label)")
    end
    pin_outer_algorithm && assert_outer_algorithm_explicit!(kc; context = "run_cm_pairwise_quantile_upper_checkpointed($label)")

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
        _, _, verify_final = cm_pairwise_quantile_production_value_verified_screened(
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
            L = L, G = cm_grid_size, n_families = cm_moment_families, contrasts = contrasts,
            cm_thresholds = copy(z_cm), pq_cutoffs = copy(Qfixed),
            min_bin_count = min_bin_count, mass_start = mass_start, n_raw = n_cut,
            gradient_backend = gradient_backend,
            screen_summary = as_namedtuple(pcx.screen_counters))
end

"""
    run_cm_pairwise_quantile_lower_checkpointed(w0=nothing; label=..., kwargs...)

Thin wrapper with `find_smallest=false` (the lower-kappa direction), mirroring every other family's.
Forwards every other argument unchanged.
"""
function run_cm_pairwise_quantile_lower_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        label::String = "cm_pairwise_quantile_lower", kwargs...)
    return run_cm_pairwise_quantile_upper_checkpointed(w0; find_smallest = false, label = label, kwargs...)
end
