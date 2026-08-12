# ================================================================================================
# Production wiring for the CM + pairwise-quantile family (family #7, 2026-08-12): the campaign-
# lifetime context builder, the per-outer-point inner-solve entry point, and the KNITRO registration.
#
# Mirrors `pairwise_quantile_production.jl` (standalone PQ) and `cm_meanzc_production.jl` (CM+ZC)
# structurally, so the five-family campaign runner sees the same shape it already knows:
#
#   build_cm_pairwise_quantile_context(ctx, cfg)   -- ONCE per campaign: cutoffs, bins, operator,
#                                                     CM grid metadata, the no-H OperatorPsiBundle,
#                                                     and every structural gate this family needs.
#   archCMPQ_base_state(x_free0, raw_masses, econ_ctx, ctx_cm)
#                                                  -- ONCE per outer point: prime, decode masses,
#                                                     run the real KNITRO inner dual solve.
#
# EVERY SCIENTIFIC PARAMETER COMES FROM `cfg` (a `CMPairwiseQuantileConfig`, all fields required) OR
# FROM `ctx` (sigma, muHat, U, D, refIndex1). Nothing here defaults one. `inner_opt` is the KNITRO
# OPTION FILE, which sets tolerances rather than the economic problem, and is therefore allowed to
# fall back to the economic bundle's own -- documented at its keyword.
#
# WHAT THE CONTEXT BUILDER DELIBERATELY DOES NOT DO: it never materializes CM's `W x ncm` matrix.
# `precalc_common_marginals_cdf` would (1.5 GB at D=20/W=100k/G=50/two families), and this family
# has no use for it -- CM enters the FG through its bin-lookup kernels and the Hessian through its
# contingency tables. Only CM's THRESHOLD array and bin indices are needed, and those are O(G) and
# O(W*D). The threshold expression here (`theoretical_u_threshold.(probs)`) is the same single line
# `precalc_common_marginals_cdf` runs internally, and the D=4 oracle checks the two agree exactly
# (`CM thresholds are the theoretical k/G ones`), so the duplication cannot drift silently.
# ================================================================================================

isdefined(Main, :CMPairwiseQuantileOperatorState) ||
    include(joinpath(@__DIR__, "cm_pairwise_quantile_lookup_kernels.jl"))
isdefined(Main, :CMPQCoreHessCtx) ||
    include(joinpath(@__DIR__, "cm_pairwise_quantile_hessian_assembly.jl"))

"""
    build_cm_pairwise_quantile_context(ctx, cfg::CMPairwiseQuantileConfig; inner_opt=nothing) -> NamedTuple

ONCE per campaign. `ctx` is the ORIGINAL, unaugmented economic context (`d4_exact_setup` /
`d20_real_setup`-shaped: `obj`, `U`, `D`, `muHat`, `sigma`, `gamma.refIndex1`).

Returns a NamedTuple with
  `obj_cmpq`     the no-H `OperatorPsiBundle` for `[economic | level | pair | CM-grid]`
  `ncore_econ`   `ctx.obj.d` (economic width INCLUDING the outer-only gravity column)
  `ncore1`       `ncore_econ - 1`, the economic inner lambda length
  `op`           the `PairwiseQuantileOperator` (cutoffs + campaign-constant bins + T3/T4 registries)
  `mass_state`   the single shared `CMPQMassState`
  `core_cf_ref`  the `Ref{Any}` `prime_operator!` publishes the `CompressedFactual` into
  `Bidx`, `z_cm`, `origins`, `nO`, `Lcm`, `ncm`, `R`, `Pow`, `refIndex1`, `bin_counts`
  `n_restr`      `n_cmpq_restr_rows(D,L)`
  `raw_start`    the `L-1` raw mass coordinates implied by `cfg.mass_start`
  `raw_bounds`   the data-derived box for those coordinates
  `gates`        what the structural gates actually measured, for the run record

STRUCTURAL GATES RUN HERE, not deferred (each is a hard error):
  * `L | cm_grid_size`                          (`resolve_cm_pairwise_quantile_config`)
  * PQ cutoffs are bit-identical selections from CM's own threshold array
  * PQ bins agree between the z-space route and CM's integer bin map, every (draw,origin) cell
    (`assert_cm_pq_bin_consistency`) -- this is the superset property the dropped per-origin
    marginal rows depend on
  * every PQ marginal bin and joint cell holds >= `cfg.min_bin_count` draws
    (`assert_pairwise_quantile_bins_nondegenerate`)
"""
function build_cm_pairwise_quantile_context(ctx, cfg::CMPairwiseQuantileConfig;
                                            inner_opt::Union{Nothing,String} = nothing)
    rc = resolve_cm_pairwise_quantile_config(cfg)
    obj0 = ctx.obj
    D = ctx.D
    L = rc.L
    W = size(ctx.U, 1)
    refIndex1 = ctx.γ.refIndex1
    1 <= refIndex1 <= D || error("build_cm_pairwise_quantile_context: ctx.γ.refIndex1=$refIndex1 outside 1:$D")

    # ---- CM grid: thresholds and bins only, never the dense matrix (see file header) ------------
    z_cm = theoretical_u_threshold.(rc.cm_probs)
    origins = [o for o in 1:D if o != refIndex1]
    nO = length(origins)
    Lcm = rc.n_cm_levels
    ncm = n_cm_moments(D, Lcm; include_truncated_moment = (rc.n_families == 2))
    ncm == rc.n_families * nO * Lcm ||
        error("build_cm_pairwise_quantile_context: ncm=$ncm != n_families*nO*Lcm=$(rc.n_families*nO*Lcm)")
    Bidx = compute_bin_indices(ctx.U, z_cm)
    R = rc.contrasts === :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    # eq.36 is a POWER-weighted moment and must be built from the Frechet productivity draw
    # z = U^(-mu), not from U -- via the same `frechet_power_feature` every other family uses.
    Pow = rc.n_families == 2 ? frechet_power_feature(ctx.U, ctx.σ - 1, Float64(ctx.μHat)) : nothing

    # ---- PQ side: cutoffs SELECTED from CM's own thresholds, bins cross-checked ------------------
    c_u = cm_pq_u_cutoffs_from_cm_grid(z_cm, rc.G, L)
    Q = cm_pq_z_cutoffs_from_u(c_u, D; mu_frechet = Float64(ctx.μHat))
    Zfeat = pairwise_quantile_frechet_features(ctx.U, Float64(ctx.μHat))
    op = PairwiseQuantileOperator(Zfeat, L, Q)
    bit_ok = all(c_u[r] === z_cm[cm_pq_grid_index(rc.G, L, r)] for r in 1:(L-1))
    bit_ok || error("build_cm_pairwise_quantile_context: the PQ U-cutoffs are not bit-identical " *
                    "selections from CM's own threshold array -- they must be selected, never " *
                    "recomputed (see cm_pairwise_quantile_config.jl's header).")
    bincheck = assert_cm_pq_bin_consistency(op, Bidx, rc.G)
    occ = assert_pairwise_quantile_bins_nondegenerate(op; min_bin_count = cfg.min_bin_count)

    # ---- the no-H operator bundle ---------------------------------------------------------------
    ncore_econ = obj0.d
    n_restr = n_cmpq_restr_rows(D, L)
    outer_constr_index_new = obj0.outer_constr_index + n_restr + ncm
    core_cf_ref = Ref{Any}(nothing)
    obj_cmpq = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, l = obj0.l, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        inner_loop_opt = inner_opt === nothing ? obj0.inner_loop_opt : inner_opt)

    # ---- starting point + box for the L-1 shared mass coordinates --------------------------------
    bin_counts = occ.bin_counts
    raw_start = rc.mass_start === :uniform ? cmpq_uniform_mass_raw(L) :
                cmpq_empirical_mass_raw(bin_counts, refIndex1, L)
    raw_bounds = cmpq_default_raw_mass_bounds(bin_counts, refIndex1, L)

    mass_state = CMPQMassState(L)
    return (obj_cmpq = obj_cmpq, ncore_econ = ncore_econ, ncore1 = ncore_econ - 1, op = op,
            mass_state = mass_state, core_cf_ref = core_cf_ref, Bidx = Bidx, z_cm = z_cm,
            origins = origins, nO = nO, Lcm = Lcm, ncm = ncm, R = R, Pow = Pow,
            refIndex1 = refIndex1, bin_counts = bin_counts, n_restr = n_restr, L = L, G = rc.G,
            n_families = rc.n_families, contrasts = rc.contrasts, raw_start = raw_start,
            raw_bounds = raw_bounds,
            gates = (bin_cells_checked = bincheck.n_checked, cutoffs_bit_identical = bit_ok,
                     min_marginal_count = occ.min_marginal_count, min_joint_count = occ.min_joint_count,
                     min_joint_cell = occ.min_joint_cell))
end

"""
    cm_pairwise_quantile_fg_state(cmpq, ctx_cm) -> CMPairwiseQuantileOperatorState

The per-inner-solve FG state, built from the campaign context. Separated from
`archCMPQ_base_state` so a test/diagnostic can drive the FG functor directly without a KNITRO solve
(which is exactly what the FG gates do).
"""
function cm_pairwise_quantile_fg_state(cmpq, ctx_cm)
    return CMPairwiseQuantileOperatorState(ctx_cm.obj, cmpq.ncore1, cmpq.op, cmpq.mass_state,
        cmpq.core_cf_ref, cmpq.ncm, cmpq.Lcm, cmpq.origins, cmpq.refIndex1, cmpq.Bidx, cmpq.R;
        Pow = cmpq.Pow)
end

"""
    inner_loop_KNITRO_cmpairwisequantile_operator(obj, st; hess_cb_builder) -> (nStatus, objSol, x, lambda, n_fg, n_hess)

KNITRO registration, mirroring `inner_loop_KNITRO_pairwisequantile_operator` exactly. The Hessian
callback is registered ONLY when the loaded option file asks for an exact Hessian
(`hessopt == 1`), so the same code path serves both the FG-only gate (`ek_inner_cmpq_fgonly.opt`,
`hessopt lbfgs`) and production (`ek_inner_cmpq.opt`, `hessopt exact`).

`hess_cb_builder = nothing` means "no exact-Hessian callback is available"; combined with an option
file that requests one, that is a hard error rather than a silent downgrade to a quasi-Newton solve
-- a silent downgrade would look like a working family that is merely slow.
"""
function inner_loop_KNITRO_cmpairwisequantile_operator(obj, st::CMPairwiseQuantileOperatorState;
                                                       hess_cb_builder)
    CS.guard_enter_inner_solve!()
    health = CallbackHealthRecord()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        x_initial = CS.inner_loop_initial_values(obj)
        KNITRO.KN_set_var_primal_init_values_all(kc, x_initial)

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[],
            callback_health_guard(_callbackEvalFG_inner_cmpairwisequantile!, health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        n_hess = Ref(0)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_cb_builder === nothing &&
                error("inner_loop_KNITRO_cmpairwisequantile_operator: the option file " *
                      "$(obj.inner_loop_opt) requests hessopt=exact but no Hessian callback builder " *
                      "was supplied. Refusing to run a silent quasi-Newton solve under a " *
                      "production option file -- pass a builder, or use an option file whose " *
                      "hessopt is not `exact`.")
            hess_cb_raw = hess_cb_builder(obj)
            hess_cb_adapted = callback_health_guard((kc2, cb2, evalRequest, evalResult, userParams) -> begin
                r = hess_cb_raw(kc2, cb2, evalRequest, evalResult, userParams.obj)
                n_hess[] += 1
                return r
            end, health)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb_adapted)
        end

        KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        assert_no_fake_success!("inner_loop_KNITRO_cmpairwisequantile_operator", health, nStatus,
                                st.n_fg_calls, x_initial, x)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        n_fg = st.n_fg_calls
        KNITRO.KN_free(kc)
        return nStatus, objSol, x, lambda_, n_fg, n_hess[]
    finally
        CS.guard_exit_inner_solve!()
    end
end

"""
    archCMPQ_base_state(x_free0, raw_masses, econ_ctx, ctx_cm; hess_cb_builder)
        -> (nStatus, x, obj, n_fg, n_hess)

ONCE per outer point: build `theta_econ`, prime the economic state (`prime_operator!`), decode this
family's shared masses (`reset_for_solve!`), and run the real KNITRO inner dual solve.

`hess_cb_builder` is REQUIRED, not defaulted: pass `cmpq_hess_builder_for(ctx_cm)` for a production
(`hessopt=exact`) solve, or an explicit `nothing` for an FG-only option file. Omitting it would make
"which Hessian is this solve using" an invisible property of a call site, which is exactly the class
of silent-default this repo does not allow.

`econ_ctx` MUST be the ORIGINAL, unaugmented economic context -- NOT `ctx_cm`, whose `.obj` is the
restriction-augmented `OperatorPsiBundle`. `cf_build`/`prime_operator!` read dimensionality off
`ctx.obj` internally, and passing the augmented context corrupts `cf.oci` (confirmed live for the
standalone family: it gave `cf.oci-1 = 129` instead of the correct economic-only 17 -- a real bug
caught by a D=4 KNITRO run, see `archPQ_base_state`'s own docstring).
"""
function archCMPQ_base_state(x_free0::AbstractVector, raw_masses::AbstractVector{Float64},
                             econ_ctx, ctx_cm; hess_cb_builder)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    prime_operator!(obj, θ_econ0, econ_ctx, ctx_cm.cmpq_core_cf_ref)

    st = ctx_cm.cmpq_fg_state
    reset_for_solve!(st, raw_masses)
    nStatus, objSol, x, lambda_, n_fg, n_hess =
        inner_loop_KNITRO_cmpairwisequantile_operator(obj, st; hess_cb_builder = hess_cb_builder)
    return nStatus, x, obj, n_fg, n_hess
end

"""
    cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx) -> ctx_cm

Merges the family's campaign state onto the economic context, giving the `ctx_cm` shape the rest of
the family (and `archCMPQ_base_state`) expects: `.obj` is the augmented bundle, and the family's own
handles are namespaced under `cmpq_*` so nothing collides with the standalone PQ family's `pq_*`
fields if both are ever attached to the same context in a comparison harness.

`build_hessian_ctx` decides whether `ctx_cm.cmpq_hess_ctx` is populated (`CMPQCoreHessCtx`, needed
for `hessopt=exact`) or left `nothing`. It is a WIRING switch, not a scientific one -- it changes
only which buffers exist, never what problem is solved -- but it is required rather than defaulted
anyway, because building it is not free (CM's `Ttab`/`CT` family plus this family's `X` tables) and
a caller doing an FG-only diagnostic should have to say so out loud. Note that populating it does
NOT by itself register a Hessian callback: `archCMPQ_base_state` still takes the builder explicitly,
so "production option file + no builder" stays the hard error it is meant to be.
"""
function cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx::Bool)
    ctx_cm = merge(ctx, (obj = cmpq.obj_cmpq, cmpq_op = cmpq.op, cmpq_mass_state = cmpq.mass_state,
                         cmpq_core_cf_ref = cmpq.core_cf_ref, cmpq_ctx = cmpq))
    st = cm_pairwise_quantile_fg_state(cmpq, ctx_cm)
    ctx_cm = merge(ctx_cm, (cmpq_fg_state = st,))
    octx = build_hessian_ctx ? build_cmpq_hess_ctx(ctx, cmpq, st) : nothing
    return merge(ctx_cm, (cmpq_hess_ctx = octx,))
end

"""
    cmpq_hess_builder_for(ctx_cm) -> Function

The `hess_cb_builder` argument `archCMPQ_base_state`/`inner_loop_KNITRO_cmpairwisequantile_operator`
expect, for a `ctx_cm` attached with `build_hessian_ctx=true`. Hard-errors (rather than returning
`nothing` and letting the registration silently fall through to a quasi-Newton solve) if the context
was attached without one.
"""
function cmpq_hess_builder_for(ctx_cm)
    octx = ctx_cm.cmpq_hess_ctx
    octx === nothing &&
        error("cmpq_hess_builder_for: this context was attached with build_hessian_ctx=false, so " *
              "there is no CMPQCoreHessCtx to build a callback from. Re-attach with " *
              "build_hessian_ctx=true.")
    return _ -> cmpq_hess_cb_builder(octx)
end
