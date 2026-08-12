# ================================================================================================
# Inner FG (forward/transpose) callback for the CM + pairwise-quantile family (family #7,
# 2026-08-12): a true no-H `OperatorPsiBundle` state composing THREE already-validated blocks, with
# no new algebra and no dense `G` anywhere.
#
#   E  economic     -- shared `economic_forward!`/`economic_transpose!` (economic_operator.jl)
#                      against the `core_cf_ref[]` `CompressedFactual` `prime_operator!` publishes.
#   R  restriction  -- this family's own `cm_pq_forward!`/`cm_pq_transpose!`
#                      (cm_pairwise_quantile_moments.jl): `L-1` reference-LEVEL rows + the pair rows,
#                      keyed on the SINGLE shared mu.
#   C  CM grid      -- CM's own production bin-lookup kernels (`apply_contrast!`/`suffix_sums!`/
#                      `cumulative_forward_contribution!`/`build_weighted_histogram!`/`prefix_sums!`/
#                      `cumulative_backward_gradient_from_prefix!`, cm_lookup_kernels.jl), REUSED
#                      VERBATIM -- the same functions `CMLookupState` and `CMMeanZCOperatorState`
#                      call. CM is a CONTINGENCY-TABLE/bin-lookup block, never a dense W x ncm matrix.
#
# This file is the direct analogue of `cm_meanzc_lookup_kernels.jl` (CM+ZC), and deliberately mirrors
# its structure field for field: same `[E | restriction | CM-grid]` column order, same `ncore1`
# semantics, same two-family (eq.35/eq.36) handling, same `hw_cache` same-point publication so a
# same-point Hessian callback reuses `r` instead of recomputing it. Read the two side by side; the
# only differences are which restriction kernel sits in the middle block and its row count.
#
# LAYOUT: `x = [zeta; lambda_E(ncore1); lambda_L(L-1); lambda_P((L-1)^2*npair); lambda_CM(ncm)]`.
# The restriction rows sit BEFORE the CM-grid block, matching CM+ZC's own
# `[economic|mean|pair|CM-grid|gravity]` (gravity is the outer-only suffix and is never an inner
# variable). `reshape_cmpq_duals` is the ONE place that arithmetic lives.
#
# ONE DELIBERATE IMPROVEMENT over the standalone PQ family's own state
# (`PairwiseQuantileOperatorState`, pairwise_quantile_production.jl): that one calls
# `build_pairwise_quantile_thread_scratch(...)` and `PairwiseQuantileTransposeScratch(...)` INSIDE its
# FG functor, i.e. it allocates the thread-local histogram scratch on EVERY FG callback (~608 KB per
# call at D=20/L=5/16 threads). Here both are campaign-lifetime fields, allocated once, per this
# codebase's own established "no per-callback vectors, matrices, closures" discipline
# (cm_lookup_kernels.jl's Phase 5.5 remediation). Same numbers, no per-call allocation. The standalone
# family's copy is left alone -- fixing it there is a separate, gated change to a family currently in
# production, not something to smuggle in here.
#
# Requires: economic_operator.jl, operator_psi_bundle.jl, cm_lookup_kernels.jl,
# operator_hessian_weights.jl, pairwise_quantile_bin_context.jl, pairwise_quantile_operator.jl,
# cm_pairwise_quantile_config.jl, cm_pairwise_quantile_moments.jl.
# ================================================================================================

isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
isdefined(Main, :HessianWeightCache) || include(joinpath(@__DIR__, "operator_hessian_weights.jl"))
isdefined(Main, :apply_contrast!) || include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
isdefined(Main, :CallbackHealthRecord) || include(joinpath(@__DIR__, "cm_callback_health.jl"))

"""
    CMPairwiseQuantileOperatorState

Per-inner-solve mutable FG state. Fields group as `[shared/economic | this family's restriction |
CM-grid block | scratch]`.

`refIndex1` serves BOTH roles on purpose: it is CM's contrast anchor AND the origin whose bins carry
this family's level rows. They must be the same origin -- that is what makes the dropped per-origin
marginal rows exactly implied (cm_pairwise_quantile_config.jl's header) -- so there is one field, not
two that could drift apart.
"""
mutable struct CMPairwiseQuantileOperatorState
    obj::Any
    ncore1::Int                      # economic (pre-gravity) inner lambda length = ncore_econ - 1
    # ---- this family's restriction block ----
    op::PairwiseQuantileOperator
    mass_state::CMPQMassState
    tls::PairwiseQuantileThreadScratch          # campaign-lifetime (see header)
    pq_scratch::PairwiseQuantileTransposeScratch
    g_L::Vector{Float64}                        # (L-1) transpose output, copied into g
    g_P::Array{Float64,3}                       # (L-1)x(L-1)xnpair transpose output
    # ---- economic ----
    core_cf_ref::Ref{Any}
    econ_ws::Any
    econ_ws_for::Any
    # ---- CM-grid block (mirrors CMMeanZCOperatorState's own field list) ----
    ncm::Int
    Lcm::Int                         # CM's number of grid LEVELS (= cm_grid_size - 1), not PQ's L
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    bins::Matrix{<:Unsigned}
    R::Union{Nothing,Matrix{Float64}}
    nbins::Int
    λmat_ext::Matrix{Float64}
    cm_contrib::Vector{Float64}
    λmat_block::Matrix{Float64}
    hist_partials::Vector{Matrix{Float64}}
    hist_h::Matrix{Float64}
    Hpre::Matrix{Float64}
    g_block::Matrix{Float64}
    g_stored::Matrix{Float64}
    # eq.36 companions -- `Pow === nothing` for a single-family (eq.35-only) state
    Pow::Union{Nothing,Matrix{Float64}}
    cm_contrib2::Vector{Float64}
    λmat_ext2::Matrix{Float64}
    λmat_block2::Matrix{Float64}
    hist_partials2::Vector{Matrix{Float64}}
    hist_h2::Matrix{Float64}
    Hpre2::Matrix{Float64}
    g_block2::Matrix{Float64}
    g_stored2::Matrix{Float64}
    # ---- shared scratch ----
    arg0::Vector{Float64}
    arg1::Vector{Float64}
    econ_buf::Vector{Float64}
    n_fg_calls::Int
    hw_cache::HessianWeightCache
end

function CMPairwiseQuantileOperatorState(obj, ncore1::Int, op::PairwiseQuantileOperator,
        mass_state::CMPQMassState, core_cf_ref::Ref{Any}, ncm::Int, Lcm::Int, origins::Vector{Int},
        refIndex1::Int, bins::Matrix{<:Unsigned}, R::Union{Nothing,Matrix{Float64}};
        nthreads_use::Int = Threads.nthreads(), Pow::Union{Nothing,Matrix{Float64}} = nothing)
    D = op.D; L = op.L; nc = L - 1; npair = op.npair
    W_ = op.W
    nO = length(origins)
    nbins = Lcm + 1
    D_bins = size(bins, 2)
    D_bins == D || error("CMPairwiseQuantileOperatorState: size(bins,2)=$D_bins != op.D=$D")
    size(bins, 1) == W_ || error("CMPairwiseQuantileOperatorState: size(bins,1)=$(size(bins,1)) != op.W=$W_")
    nt = max(1, min(nthreads_use, W_))
    if Pow !== nothing
        ncm == 2 * nO * Lcm ||
            error("CMPairwiseQuantileOperatorState: Pow given (two-family CM) but ncm=$ncm != " *
                  "2*nO*Lcm=$(2*nO*Lcm) -- a two-family state needs the doubled CM width")
        size(Pow) == (W_, D) ||
            error("CMPairwiseQuantileOperatorState: size(Pow)=$(size(Pow)) != (W,D)=($W_,$D)")
    else
        ncm == nO * Lcm ||
            error("CMPairwiseQuantileOperatorState: single-family CM but ncm=$ncm != nO*Lcm=$(nO*Lcm)")
    end
    n_x = 1 + ncore1 + n_cmpq_restr_rows(D, L) + ncm
    return CMPairwiseQuantileOperatorState(obj, ncore1,
        op, mass_state, build_pairwise_quantile_thread_scratch(D, npair, L),
        PairwiseQuantileTransposeScratch(D, npair, L), zeros(nc), zeros(nc, nc, npair),
        core_cf_ref, nothing, nothing,
        ncm, Lcm, nO, origins, refIndex1, bins, R, nbins,
        zeros(nO, Lcm + 1), zeros(W_), zeros(nO, Lcm),
        [zeros(D, nbins) for _ in 1:nt], zeros(D, nbins), zeros(D, Lcm),
        zeros(nO, Lcm), zeros(nO, Lcm),
        Pow, zeros(W_), zeros(nO, Lcm + 1), zeros(nO, Lcm),
        [zeros(D, nbins) for _ in 1:nt], zeros(D, nbins), zeros(D, Lcm),
        zeros(nO, Lcm), zeros(nO, Lcm),
        zeros(W_), zeros(W_), zeros(W_), 0, HessianWeightCache(n_x))
end

"""
    reset_for_solve!(st::CMPairwiseQuantileOperatorState, raw_masses) -> st

Call ONCE per outer point, BEFORE the KNITRO solve starts: decode the `L-1` raw outer coordinates
into the shared free masses (`set_cmpq_masses!`). Same lifecycle contract as every other family's
`reset_for_solve!` -- never called from inside an FG or Hessian callback, because everything
downstream reads `mass_state.mu` as a constant of the inner problem.

The BIN assignment is not refreshed here and never is: the cutoffs are campaign constants derived
once from CM's own grid, so `op.bin` was built once in the operator's constructor.
"""
function reset_for_solve!(st::CMPairwiseQuantileOperatorState, raw_masses::AbstractVector{Float64})
    set_cmpq_masses!(st.mass_state, raw_masses)
    st.n_fg_calls = 0
    return st
end

"""
    dual_index!(st::CMPairwiseQuantileOperatorState, x) -> st.arg0

Computes `st.arg0 = r = -zeta - E*lambda_E - G_R*lambda_R - G_CM*lambda_CM` in place. Extracted so
the FG functor and the Hessian-weight prep (`operator_prep_for_hessian!`) provably run the identical
code path, exactly as `CMLookupState`/`CMMeanZCOperatorState` do.

Every one of the three blocks SUBTRACTS into `arg0`, which is the sign convention every derivation in
this family is stated against (see cm_pairwise_quantile_moments.jl's header).
"""
function dual_index!(st::CMPairwiseQuantileOperatorState, x::AbstractVector{Float64})
    ncore1 = st.ncore1
    ζ = x[1]
    λ_E = @view x[2:1+ncore1]
    λ_L, λ_P, λ_CM = reshape_cmpq_duals(x, st.op, ncore1)
    length(λ_CM) == st.ncm ||
        error("dual_index!(CMPairwiseQuantileOperatorState): CM dual slice is $(length(λ_CM)) long, " *
              "expected ncm=$(st.ncm) -- the inner variable vector does not match this state's layout")

    fill!(st.arg0, -ζ)

    # (E) economic -- operator only. There is NO dense fallback branch here on purpose: this family
    # is operator-native (memories `feedback-no-dense-reduced-ever-anywhere`,
    # `feedback-never-silently-fall-back-to-dense-reference`), so a missing CompressedFactual is a
    # hard error rather than a silent switch to a dense read.
    cf = st.core_cf_ref[]
    cf isa CompressedFactual ||
        error("dual_index!(CMPairwiseQuantileOperatorState): core_cf_ref[] is not a CompressedFactual " *
              "-- prime_operator! was not called for this outer point. This family has no dense " *
              "economic fallback by design.")
    if st.econ_ws === nothing || st.econ_ws_for !== cf
        st.econ_ws = economic_operator_workspace(cf)
        st.econ_ws_for = cf
    end
    economic_forward!(st.econ_buf, λ_E, cf, st.econ_ws)
    st.arg0 .-= st.econ_buf

    # (R) this family's restriction: level + pair rows on the shared mu
    cm_pq_forward!(st.arg0, λ_L, λ_P, st.op, st.mass_state, st.refIndex1)

    # (C) CM grid, via CM's own lookup kernels (cumulative/:suffix basis, CM's production default)
    ncm_cdf = st.nO * st.Lcm
    λ_cdf = st.Pow === nothing ? λ_CM : (@view λ_CM[1:ncm_cdf])
    apply_contrast!(st.λmat_block, reshape(λ_cdf, st.nO, st.Lcm), st.R)
    suffix_sums!(st.λmat_ext, st.λmat_block)
    cumulative_forward_contribution!(st.cm_contrib, st.bins, st.refIndex1, st.origins, st.λmat_ext)
    st.arg0 .-= st.cm_contrib

    if st.Pow !== nothing
        λ_pow = @view λ_CM[ncm_cdf+1:2*ncm_cdf]
        apply_contrast!(st.λmat_block2, reshape(λ_pow, st.nO, st.Lcm), st.R)
        suffix_sums!(st.λmat_ext2, st.λmat_block2)
        # eq.36's indicator is `1{U>c}`, not `1{U<=c}` -- the suffix table's "total at index 1"
        # property supplies the reflection with no new table. See
        # `interval_forward_contribution_pow!`'s own docstring for the derivation and the bug it fixed.
        cumulative_forward_contribution_pow!(st.cm_contrib2, st.bins, st.refIndex1, st.origins,
                                             st.λmat_ext2, st.Pow)
        st.arg0 .-= st.cm_contrib2
    end
    return st.arg0
end

"""
    (st::CMPairwiseQuantileOperatorState)(x, g=Float64[]) -> f

FG evaluator with the same signature/semantics as `obj(x, g)`:
`f = (1/M) sum_w Psi(r_w) + zeta`, `g = d f / d(zeta, lambda)`.
"""
function (st::CMPairwiseQuantileOperatorState)(x::AbstractVector{Float64},
                                               g::AbstractVector{Float64} = Float64[])
    obj = st.obj
    M = length(st.arg0)
    ncore1 = st.ncore1
    op = st.op
    D = op.D; npair = op.npair; L = op.L; nc = L - 1

    ζ = x[1]
    cf = st.core_cf_ref[]
    dual_index!(st, x)

    obj.Psi!(st.arg1, st.arg0)
    f = sum(st.arg1) / M + ζ

    if length(g) > 0
        obj.dPsi!(st.arg1, st.arg0)
        g[1] = 1.0 - sum(st.arg1) / M

        # (E)
        g_E = @view g[2:1+ncore1]
        economic_transpose!(g_E, st.arg1, cf, st.econ_ws)
        g_E .*= -(1.0 / M)

        # (R) -- writes into persistent buffers, then copies into g's own slices. The copy is
        # deliberate: `g_P`'s `(nc,nc,npair)` shape is what `cm_pq_transpose!` writes, and a
        # `reshape` of a `@view` into `g` would alias fine but makes the offset arithmetic appear in
        # two places instead of one (`reshape_cmpq_duals`). At D=20/L=5 the copy is 3044 doubles.
        cm_pq_transpose!(st.g_L, st.g_P, st.arg1, op, st.mass_state, st.refIndex1, st.tls, st.pq_scratch)
        off = 1 + ncore1
        nL = n_cmpq_level_rows(L)
        @inbounds for a in 1:nL
            g[off+a] = st.g_L[a]
        end
        gP_flat = reshape(@view(g[off+nL+1 : off+nL+nc*nc*npair]), nc, nc, npair)
        copyto!(gP_flat, st.g_P)

        # (C) CM grid, via CM's own lookup kernels
        cm_off = off + n_cmpq_restr_rows(D, L)
        ncm_cdf = st.nO * st.Lcm
        build_weighted_histogram!(st.hist_h, st.hist_partials, st.bins, st.arg1, D, st.nbins)
        prefix_sums!(st.Hpre, st.hist_h, st.Lcm)
        cumulative_backward_gradient_from_prefix!(st.g_block, st.Hpre, st.refIndex1, st.origins, st.Lcm, M)
        apply_contrast!(st.g_stored, st.g_block, st.R)
        @views g[cm_off+1:cm_off+ncm_cdf] .= vec(st.g_stored)
        if st.Pow !== nothing
            build_weighted_histogram_pow!(st.hist_h2, st.hist_partials2, st.bins, st.arg1, st.Pow, D, st.nbins)
            prefix_sums!(st.Hpre2, st.hist_h2, st.Lcm)
            # Reflect the prefix sums in place for eq.36's `1{U>c}` indicator -- `Total - prefix`,
            # reusing `cumulative_backward_gradient_from_prefix!` unchanged (see
            # cm_lookup_kernels.jl::cm_transpose_into_g!'s own note).
            @inbounds for o in 1:D
                total_o = sum(@view st.hist_h2[o, :])
                for l in 1:st.Lcm
                    st.Hpre2[o, l] = total_o - st.Hpre2[o, l]
                end
            end
            cumulative_backward_gradient_from_prefix!(st.g_block2, st.Hpre2, st.refIndex1, st.origins, st.Lcm, M)
            apply_contrast!(st.g_stored2, st.g_block2, st.R)
            @views g[cm_off+ncm_cdf+1:cm_off+st.ncm] .= vec(st.g_stored2)
        end
    end

    obj.arg0 .= st.arg0
    _publish_dual_index_cache!(st, x)   # let a same-point Hessian call reuse this r
    st.n_fg_calls += 1
    return f
end

"KNITRO FG callback -- mirrors `_callbackEvalFG_inner_pairwisequantile!` exactly (generic on any
callable `st`, no family-specific assumption)."
function _callbackEvalFG_inner_cmpairwisequantile!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    x = evalRequest.x
    f = st(x, evalResult.objGrad)
    evalResult.obj[1] = f <= st.obj.lower_limit ? -KNITRO.KN_INFINITY : f
    return 0
end
