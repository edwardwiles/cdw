# D=20 profiling task (flexible_cm/common_frechet, 2026-07-28): this file's Hessian callbacks use
# the opt-in `@cmhess_prof` sub-block timing macro (default off, zero overhead) -- self-include the
# defining file if a caller hasn't already loaded it, same defensive pattern this codebase's own
# cm_checkpoint.jl already uses for its own optional-dependency includes. Requires instrumentation.jl
# (PROF_ENABLED/prof_record!) to already be loaded -- true of every existing caller of this file,
# since this file's OWN pre-existing `@prof` uses already assumed that ordering.
isdefined(Main, :CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED) || include(joinpath(@__DIR__, "cm_hessian_subblock_profiling.jl"))

# ============================================================================
# Continuation (branch diag/fullA-d4-exact-cm-hessian-arch): Hessian
# architecture comparison for the CM-augmented inner CC dual solve, D=4.
#
# BASELINE, unchanged (Architecture A): `build_cm_augmented_obj`'s resulting
# `obj_cm` fed straight into the EXISTING generic dense-BLAS
# `cc_algo/PsiObjectiveBundle.jl::hessian!` -- that function already operates
# on whatever `H`/`d`/`outer_constr_index` the bundle carries, so it needs NO
# code change to work for the CM-augmented moment layout; this file does not
# redefine it, only re-exercises it via the standard KNITRO wiring
# (`inner_loop_KNITRO_profiled` from oracle_fast.jl, included by callers).
#
# This file adds THREE additive architectures (B, C, D) that all produce the
# SAME packed upper-triangular Hessian `cc_algo/PsiObjectiveBundle.jl::hessian!`
# does (validated in c13_validate_hessian_archs.jl), for the SAME CM-augmented
# `obj_cm`, so any of them can be swapped in as a KNITRO Hessian callback with
# zero change to FG callback, bounds, or complementarity wiring.
#
# ---- Notation (see docs/fullA_cm_hessian_architecture_report.md sec 2) ----
# The inner Newton Hessian is w.r.t. x=(zeta,lambda), dimension
# n = outer_constr_index = NCORE + ncm, where:
#   NCORE = aug.ncore  (E-block width: 1 "ones"/zeta column + (NCORE-1)
#           pregrav economic moment columns -- gravity itself is excluded,
#           it is the sole OUTER-only column, see common_marginals_moments.jl)
#   ncm   = aug.ncm    (common-marginals block width, (D-1)*L for
#           include_truncated_moment=false)
# E = H[:, 2:1+NCORE]         (W x NCORE)   -- "existing economic moments" (+intercept)
# C = H[:, 2+NCORE:1+NCORE+ncm]  (W x ncm)  -- common-marginals block
# H_partition = [[H_EE H_EC];[H_EC' H_CC]], H_EE=(1/M)E'DE, H_EC=(1/M)E'DC,
# H_CC=(1/M)C'DC, D=diag(w), w=ddPsi!(arg0) (obj.arg2 after `ddPsi!(arg2,arg0)`).
# ============================================================================

using LinearAlgebra: BLAS, mul!

# Allocation/Hessian port task §5: compressed winner-form core-moment representation
# (build_compressed_factual/materialize_dense_factual_structured!/fill_K_directgp!/
# compressed_gravity_raw) -- not previously needed by the CM include stack. Self-include-guarded
# (this codebase's own convention, e.g. infeasibility_screen.jl's winner_certificate.jl guard) so
# this file works regardless of which driver's include stack pulls it in. Dependency order matches
# c10_d20_production_driver.jl's own (compressed_moments.jl -> structured_moment_build.jl ->
# compressed_live.jl, the last of which requires oracle_fast.jl already loaded -- true in every
# real caller of this file).
isdefined(Main, :CompressedFactual) || include(joinpath(@__DIR__, "compressed_moments.jl"))
isdefined(Main, :materialize_dense_factual_structured!) || include(joinpath(@__DIR__, "structured_moment_build.jl"))
# Profiled all-families completion task (2026-08-01): the reduced/anchor-omitting H_EE kernel
# _fill_cm_HEE! now optionally dispatches to (build_reduced_homogeneous_winner_pair_ctx/
# reduced_homogeneous_winner_pair_hessian!, only reached when a caller opts in via
# build_cm_bin_ctx(...; profiled_layout=...)) -- guarded includes for that dependency chain, same
# order test_reduced_homogeneous_hessian_2026-08-01.jl already establishes as canonical
# (active_layout.jl -> relative_a_coordinate_2026-07-31.jl -> profiled_economic_moment_layout_2026-08-01.jl
# -> reduced_homogeneous_hessian_2026-08-01.jl; the FG-only reduced_homogeneous_contraction_2026-08-01.jl
# is NOT needed here, this task only touches the Hessian side).
isdefined(Main, :dest_slot) || include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
isdefined(Main, :AnchorSpec) || include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
isdefined(Main, :ProfiledEconomicMomentLayout) || include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
isdefined(Main, :ReducedHomogeneousWinnerPairHessCtx) || include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
# reduced_homogeneous_dual_contraction -- needed by materialize_homogeneous_dense_G_reduced!
# (profiled_restricted_family_base_2026-08-01.jl), the FG counterpart of the homogeneous H_EE/H_EC
# kernels above (formulation-consistency bugfix, 2026-08-01).
isdefined(Main, :reduced_homogeneous_dual_contraction) || include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
# build_reduced_base_obj_for_family/materialize_dense_factual_structured_reduced! -- used by
# build_cm_augmented_obj_archB's base_obj kwarg and wrap_moments_with_cm_archB's profiled_layout
# kwarg respectively (both additive, both nothing by default).
isdefined(Main, :materialize_dense_factual_structured_reduced!) || include(joinpath(@__DIR__, "profiled_restricted_family_base_2026-08-01.jl"))
# port/shared-inner-fg-operator-and-verification-2026-07-26: compressed_live.jl (as of Addendum
# Part A, fbb7d79) references EconomicFGWorkspace/compressed_cc_value_grad! from
# compressed_cc_inner.jl but never includes it itself (every pre-existing caller happened to
# include compressed_cc_inner.jl separately before compressed_live.jl) -- this include stack didn't,
# so add the same self-guard here rather than relying on caller discipline.
isdefined(Main, :EconomicFGWorkspace) || include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
isdefined(Main, :compressed_gravity_raw) || include(joinpath(@__DIR__, "compressed_live.jl"))
# Allocation/Hessian port task §6.1/6.2: threaded Architecture-C Hessian (ThreadLocalBinScratch/
# build_thread_local_scratch/hessian_cm_structured_v2!). Safe to include here despite
# cm_hessian_threaded.jl's own functions referencing CMBinHessCtx-shaped objects: neither file's
# top-level code CALLS into the other (only type/function DEFINITIONS), so there is no genuine
# circular load-order requirement -- only genuine requirement is both are loaded before either is
# actually CALLED, which this guard (placed before CMBinHessCtx's own struct, which now carries a
# tls::ThreadLocalBinScratch field) guarantees.
isdefined(Main, :ThreadLocalBinScratch) || include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
# Winner-aware H_ER phase (2026-07-27): winner_pair_cross_hessian.jl's WinnerBinCrossScratch is
# referenced by CMBinHessCtx's own struct definition below, so this include must run before that
# struct is parsed -- same self-guard convention every other dependency in this file already uses.
isdefined(Main, :WinnerBinCrossScratch) || include(joinpath(@__DIR__, "winner_pair_cross_hessian.jl"))
# Default-flips task (2026-07-27), Task C: MOMENT_REPRESENTATION selector + record_generic_dense_fg!/
# record_dense_cm_g! counters, same self-guard convention as every other dependency above.
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))
# CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): ZCCenteredScratch is referenced by
# CMBinHessCtx's own struct definition below (the shared, obj.H-independent H_ZZ/H_CZ state), so
# this include must run before that struct is parsed -- same self-guard convention every other
# dependency in this file already uses.
isdefined(Main, :ZCCenteredScratch) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))
# optimize/structured-cross-hessian-ZC-CM-2026-07-28: threaded (output-ownership) variants of the
# four raw-table-fill cross-Hessian primitives above -- see that file's own header for why. Must
# load after both includes above (references WinnerBinCrossScratch/WinnerZCCrossScratch/
# BinZCrossScratch/ZCCenteredScratch).
isdefined(Main, :cross_hessian_chunk_ranges) || include(joinpath(@__DIR__, "threaded_cross_hessian.jl"))
# optimize/structured-cross-hessian-ZC-CM-2026-07-28 ADDENDUM: H_ZZ BLAS/threaded candidates
# (SYRK/GEMM/threaded_packed), built directly from the immutable raw Phi -- must load after
# threaded_cross_hessian.jl (uses cross_hessian_chunk_ranges/resolve_cross_hessian_workers_default)
# and zc_restriction_operator.jl (uses ZCRestrictionOperator/ZCCenteredScratch/zc_restriction_gram!).
isdefined(Main, :ZCRawWeightedWorkspace) || include(joinpath(@__DIR__, "zc_gram_blas_candidates.jl"))
# ZC Hessian backend production integration (2026-08-01): H_CZ/H_EZ candidates dispatched from
# CMBinHessCtx.hcz_prep_backend/zc_ez_backend and OriginZCCoreHessCtx.zc_ez_backend below --
# previously only reachable via ad-hoc includes in diagnostic/benchmark scripts, never through this
# file's own self-guarded dependency chain (the same chain campaign_cm_family_runner.jl and every
# real caller of cm_checkpoint.jl/cm_originzc_checkpoint.jl already loads). Moved here so a clean
# checkout, with no benchmark script involved, resolves every backend Symbol this file dispatches
# on. hcz_drawchunk_candidate_2026-07-29.jl defines the PRE-EXISTING default H_CZ backend
# (:draw_chunk_thread_local) and was itself missing this same guard -- fixed as part of this task,
# not a new dependency.
isdefined(Main, :HCZ_PREP_BACKEND_DEFAULT) || include(joinpath(@__DIR__, "hcz_drawchunk_candidate_2026-07-29.jl"))
isdefined(Main, :bin_zc_cross_hessian_fill_drawchunk_reordered!) || include(joinpath(@__DIR__, "hcz_reordered_candidate_2026-08-01.jl"))
isdefined(Main, :ZC_EZ_BACKEND_DEFAULT) || include(joinpath(@__DIR__, "hez_drawmajor_candidate_2026-08-01.jl"))
isdefined(Main, :winner_pair_cross_hessian_zc_block_drawmajor_v2!) || include(joinpath(@__DIR__, "hez_drawmajor_v2_candidate_2026-08-01.jl"))
# No-moments/no-composite-G task (2026-07-28): operator_prep_for_hessian!/HessianWeightCache, used
# by archC_hess_cb_builder/archA_partitioned_hess_cb_builder below in place of the dense
# _archC_prep_for_hessian! fallback.
isdefined(Main, :HessianWeightCache) || include(joinpath(@__DIR__, "operator_hessian_weights.jl"))

# ----------------------------------------------------------------------------
# Shared: per-draw bin indices w.r.t. the SAME thresholds `z` that
# `precalc_common_marginals_cdf` uses (reused, not re-derived -- see
# `build_cm_bin_ctx` below, which calls `precalc_common_marginals_cdf` itself
# so `z` is byte-identical to what produced the reference dense CM matrix).
# bin(u) = searchsortedfirst(z, u) in {1,...,L+1}; satisfies, for l in 1:L,
# 1{u<=z_l} == (bin(u) <= l)  (z sorted ascending, verified in the header
# comment derivation, docs/fullA_cm_hessian_architecture_report.md sec 2).
# ----------------------------------------------------------------------------
function compute_bin_indices(U::AbstractMatrix{Float64}, z::AbstractVector{Float64})
    W, D = size(U)
    Bidx = Matrix{Int}(undef, W, D)
    @inbounds for x in 1:D, s in 1:W
        Bidx[s, x] = searchsortedfirst(z, U[s, x])
    end
    return Bidx
end

# ============================================================================
# ARCHITECTURE B: chunked/cached common-block materialization.
#
# Avoids (1) `wrap_moments_with_cm`'s fresh-`similar`-every-call `G_tmp`
# (cached and reused across calls instead) and (2) storing/copying from a
# persistent dense W x ncm CM matrix (built fresh from bin indices, in row
# chunks, directly into the destination G view instead). The Hessian
# CONTRACTION itself is untouched -- Architecture B's obj still dispatches to
# the same generic `hessian!` as Architecture A; only moment CONSTRUCTION
# differs. Mathematically identical G matrix content to Architecture A's (up
# to floating point summation order in the chunked recompute vs the
# precalc'd-once-then-copied path) -- validated, not assumed.
# ============================================================================

"""
Fill `Gdest` (a W x ncm view, threshold-major layout matching precalc_common_marginals_cdf)
directly from bin indices, in row-chunks of `chunk_size`. Never materializes a persistent W x ncm
matrix.

Allocation/Hessian port task §4.1: `Gdest[rows,cols] .= bview * R` previously allocated a fresh
`n x nO` product EVERY (chunk, threshold-level) iteration -- `(W/chunk_size)*L` times per call
(e.g. 40 chunks x L=50 at real D=20/W=80,000), reproducing the audit's own reported ~608 MB/call.
`prod_scratch`, when given (sized `chunk_size x nO`, built once per outer-solve process and
reused across every call -- see `wrap_moments_with_cm_archB` below), lets this use `mul!` into a
persistent buffer instead. Falls back to the original fresh-allocation behavior, unchanged, when
`prod_scratch === nothing`.
"""
function fill_cm_columns_from_bins!(Gdest::AbstractMatrix{Float64}, Bidx::AbstractMatrix{Int},
                                     origins::Vector{Int}, refIndex1::Int, L::Int,
                                     R::Union{Nothing,Matrix{Float64}}; chunk_size::Int = 2000,
                                     prod_scratch::Union{Nothing,Matrix{Float64}} = nothing)
    W = size(Gdest, 1)
    nO = length(origins)
    @assert size(Gdest, 2) == L * nO
    cs = min(chunk_size, W)
    buf = Matrix{Float64}(undef, cs, nO)
    use_scratch = R !== nothing && prod_scratch !== nothing && size(prod_scratch, 1) >= cs && size(prod_scratch, 2) == nO
    start = 1
    @inbounds while start <= W
        stop = min(start + cs - 1, W)
        rows = start:stop
        n = length(rows)
        bview = @view buf[1:n, :]
        for l in 1:L
            for (oi, o) in enumerate(origins)
                for (ridx, s) in enumerate(rows)
                    bview[ridx, oi] = Float64(Bidx[s, o] <= l) - Float64(Bidx[s, refIndex1] <= l)
                end
            end
            cols = (l - 1) * nO + 1 : l * nO
            if R === nothing
                @views Gdest[rows, cols] .= bview
            elseif use_scratch
                pview = @view prod_scratch[1:n, :]
                mul!(pview, bview, R)
                @views Gdest[rows, cols] .= pview
            else
                @views Gdest[rows, cols] .= bview * R
            end
        end
        start = stop + 1
    end
    return nothing
end

"""
    fill_gravity_column_into!(Gcol, grav_raw, ctx, d)

Generic form of `compressed_live.jl`'s `fill_gravity_column!(obj, grav_raw)`, parameterized on an
arbitrary target column view and column index `d` (rather than assuming `obj.H`/`obj.d`) --
needed because CM's compressed-core swap (`wrap_moments_with_cm_archB` below) writes gravity into
`Gtmp[:, ncore_full]`, a plain temporary matrix, at the BASE (pre-CM-augmentation) object's own
column index `ncore_full` -- not `obj_cm.d` (the CM-augmented object's, which is larger by `ncm`).
Same formula/post-processing as `fill_gravity_column!` (SamplingWeights/NormalizeMoments/usePMM);
verified bit-identical to it directly in test_cm_compressed_core.jl.
"""
function fill_gravity_column_into!(Gcol::AbstractVector{Float64}, grav_raw::Float64, ctx, d::Int)
    γo = ctx.γ
    W = length(Gcol)
    ind = γo.indicators
    nrm_g = (ind.NormalizeMoments == 1 && !(d in γo.moments_without_var)) ? 1.0 / γo.σ_Moments[d] : 1.0
    pmm_g = ind.usePMM == 1 ? γo.PMM[d] : 0.0
    @views @. Gcol = γo.SamplingWeights[1:W] * nrm_g * (grav_raw - pmm_g)
    return nothing
end

"""
    wrap_moments_with_cm_archB(core_moments!, ncore_full, Bidx, origins, refIndex1, L, R, ctx; chunk_size, use_compressed_core)

Architecture B analogue of `wrap_moments_with_cm` (common_marginals_moments.jl).
Same external contract (a `moments!`-signature closure), same column layout,
but (a) caches the `G_tmp` scratch buffer across calls (keyed on `n=size(U,1)`,
reallocated only if `n` changes) instead of `similar`-ing a fresh one every
call, and (b) builds the CM columns fresh from bin indices in row-chunks
(`fill_cm_columns_from_bins!`) instead of copying from a persistent dense CM
matrix.

Allocation/Hessian port task §5 (largest remaining allocation change): when `use_compressed_core=
true` (default), the "pregrav" (economic) + gravity columns of `Gtmp` -- previously built by
calling the DENSE `core_moments!` (`EK_moments_gammanorm_directgp!` in production, an O(W*D^2)
`hFunction!`/`hFunctionCounter!` pass, ~734 MB/call per the audit) -- are instead built via the
SAME compressed winner-form representation the unrestricted family already uses
(`build_compressed_factual`/`materialize_dense_factual_structured!`/`fill_K_directgp!`/
`compressed_gravity_raw`). Provably the same quantity: `pregrav == cf.oci-1 == D*Ddest+1` exactly
(`cf.oci = ctx.obj.outer_constr_index`, and the base/pre-CM ctx satisfies `ctx.obj.
outer_constr_index == ctx.obj.d == D*Ddest+2`, matching `EK_moments_gammanorm_directgp!`'s own
`simple_end = D*Ddest+1` -- see docs/CM_COMPRESSED_CORE_PORT_2026-07-25.md §1 for the full
derivation). Requires `ctx` to be the PLAIN, pre-CM-augmentation context (`ctx.obj.
outer_constr_index` must equal `ncore_full`, NOT `obj_cm`'s own larger `outer_constr_index`) --
callers already have this available (it is the same `ctx` `Bidx`/`R` are derived from).

Falls back to the ORIGINAL dense `core_moments!` call, for that one point only, on a
`TiedWinnerError` (an exact price tie the compressed winner-argmin does not tolerate but the
dense `hFunction!`/`MinInd!` convention handles silently) -- matches the unrestricted family's own
documented fallback discipline (compressed_live.jl's `COMPRESSED_FALLBACK_COUNT`). Set
`use_compressed_core=false` to force the original dense path unconditionally (kept for
correctness comparison / emergency revert; byte-identical to pre-port production).
"""
function wrap_moments_with_cm_archB(core_moments!::Function, ncore_full::Int,
                                     Bidx::Matrix{Int}, origins::Vector{Int}, refIndex1::Int, L::Int,
                                     R::Union{Nothing,Matrix{Float64}}, ctx; chunk_size::Int = 2000,
                                     use_compressed_core::Bool = true,
                                     core_cf_ref::Ref{Any} = Ref{Any}(nothing),
                                     theta_ref::Ref{Any} = Ref{Any}(nothing),   # profiled all-families completion task (2026-08-01): shared box, published alongside core_cf_ref, needed by the reduced H_EE kernel (see CMBinHessCtx's own profiled_theta_ref field docstring). Unused/harmless for every caller that doesn't opt into a profiled_layout.
                                     profiled_layout::Any = nothing,   # profiled all-families completion task (2026-08-01), FG-callback gap closure: when set, G is ALWAYS fully filled (skip_fill is ignored/irrelevant for a profiled context -- there is no separate ":cm_lookup priming" concept here), using the REDUCED economic materialization (materialize_dense_factual_structured_reduced!, profiled_restricted_family_base_2026-08-01.jl) instead of the full one. `nothing` (every existing caller) preserves every line below byte-for-byte.
                                     skip_fill::Bool = false)   # skip_cm_fill_ref removal (2026-07-27):
                                     # was a caller-toggled `Ref{Bool}` read at CALL time (a mutable
                                     # shared box archC_base_state/archC_verified_state set/reset around
                                     # each inner solve); now a plain, immutably-captured boolean baked
                                     # into THIS closure at BUILD time -- when `true`, this closure
                                     # unconditionally skips materializing the dense CM columns
                                     # (`G[:, cm_cols]`); when `false` (default), it always fills them.
                                     # Because the skip decision genuinely varies PER CALL (archC_
                                     # base_state wants it conditionally true, archC_verified_state
                                     # always wants it false -- see that function's own docstring), the
                                     # production builder (`build_cm_production_context`) calls this
                                     # function TWICE -- once with `skip_fill=false` (installed as
                                     # `ctx_cm.obj.moments!`, used by every non-skip path) and once with
                                     # `skip_fill=true` (installed as `cctx.moments_skip!`, used ONLY by
                                     # `inner_loop_internal_cmlookup_production`'s priming call when its
                                     # OWN caller explicitly threads `skip_fill=true` through) -- both
                                     # closures share the SAME `core_cf_ref` box (passed explicitly by
                                     # the caller) so the Hessian callback sees an identical publish
                                     # regardless of which variant ran. `skip_fill=false` (the default)
                                     # preserves the ORIGINAL unconditional-fill behavior for every
                                     # caller that doesn't pass this (e.g. the archB diagnostic builder
                                     # below, c13_*/c14_* benchmark scripts).
    pregrav = ncore_full - 1
    nO = length(origins)
    Gtmp_cache = Ref{Matrix{Float64}}(Matrix{Float64}(undef, 0, 0))
    # Allocation/Hessian port task §4.1: persistent bview*R product scratch, built once (per
    # closure lifetime -- this closure itself is built once per outer-solve process, see
    # build_cm_production_context) and reused across every fill_cm_columns_from_bins! call.
    cs = min(chunk_size, size(Bidx, 1))
    prod_scratch = R === nothing ? nothing : Matrix{Float64}(undef, cs, nO)
    return function (K, G, θ, U, obj)
        n = size(U, 1)
        if size(Gtmp_cache[], 1) != n
            Gtmp_cache[] = Matrix{Float64}(undef, n, ncore_full)
        end
        Gtmp = Gtmp_cache[]
        if use_compressed_core
            # No-moments/no-composite-G task (2026-07-28): `check_ties=false` -- was `true`, which
            # threw `TiedWinnerError` on a literal (machine-precision) price tie and fell back to a
            # dense `core_moments!`/`core_cf_ref[]=:tied_winner` reconstruction that the Hessian
            # callback would then also have to serve from dense `obj.H`. Per direct read of the
            # winner-assignment code (compressed_factual_buffer_reuse.jl/compressed_moments.jl): the
            # winner (`winner[w,s] = bo`, the argmin) is computed UNCONDITIONALLY regardless of
            # `check_ties` -- the tie check is a pure, side-effect-free diagnostic scan bolted on
            # AFTER the winner is already assigned, it does not change which winner gets picked.
            # Since a literal machine-precision tie is a measure-zero event for continuous draws and
            # the assignment is already deterministic (first-argmin) either way, there is no reason
            # to special-case it: `core_cf_ref[]` is now ALWAYS a valid `CompressedFactual` in
            # operator mode, never a `:tied_winner` Symbol, which makes the dense H_EC/H_ER fallback
            # branches in the Hessian callbacks provably unreachable in production (not merely rare)
            # -- the precondition this task's G/H storage elimination relies on.
            cf = cf_build(θ, ctx; check_ties = false)   # Phase E remediation (2026-07-26): reuses ctx.cf_workspace when attached
            # No-moments/no-composite-G task (2026-07-28), RESOLVED this session: the economic
            # block fill is now genuinely skippable under `skip_fill`. A prior attempt at exactly
            # this (2026-07-28, same day) was reverted after test_shared_core_hessian_d4_gates.jl
            # caught what LOOKED like a real H_EE regression (max|Δ|=0.0336, then an outright
            # nStatus=-400 inner-solve failure once the guard below was tried). Root-caused this
            # session, empirically (not by static reading): it was NEVER a production correctness
            # bug. Two compounding bugs, both now fixed:
            #   1. `archC_base_state`'s `skip_fill_safe` gate (cm_production_bundle.jl) checked
            #      `cctx.cm_cross_hessian_backend` (irrelevant to this fill) instead of
            #      `cctx.core_hessian_backend` (the flag that actually determines whether THIS
            #      context's Hessian path will read dense H) -- so a context explicitly built to
            #      want `:dense_reference` ground truth (only ever done by this repo's own
            #      comparison test harnesses, never in real production) got the skip applied to it
            #      too, leaving its own required dense H unfilled. Fixed by adding
            #      `&& cctx.core_hessian_backend !== :dense_reference` to the gate.
            #   2. Independently, `test_shared_core_hessian_d4_gates.jl`'s own `full_hessian` test
            #      helper called the legacy dense-only `_archC_prep_for_hessian!` directly for
            #      BOTH its comparison contexts, bypassing the real production dispatcher
            #      (`_prep_dual_index_for_archC!`) -- which happens to also need dense H, even for
            #      an operator-mode context, purely as an artifact of the test's own methodology.
            #      Fixed in the test file to route through the dispatcher for non-dense-reference
            #      contexts.
            # With both fixed, skip_fill_safe correctly differs per-context, and the economic fill
            # is safe to skip in every real production configuration (which never sets
            # core_hessian_backend=:dense_reference) -- validated D=4, all 4 sections,
            # test_shared_core_hessian_d4_gates.jl, 40/40 PASS including the real-solved-point arm.
            if profiled_layout !== nothing
                # Formulation-consistency bugfix (2026-08-01): MUST be the HOMOGENEOUS-formulation
                # G (matching reduced_homogeneous_winner_pair_hessian!/winner_pair_cross_hessian_
                # cm_block!'s use_profiled_correction=true path), NOT the structured-formulation one
                # -- see materialize_homogeneous_dense_G_reduced!'s own docstring for the full
                # derivation (confirmed via autodiff: mixing the two gave a real, ~30-95%-off H_EC
                # block and a KNITRO solve that ground through the iteration limit instead of
                # converging in ~4 iterations like the full model).
                materialize_homogeneous_dense_G_reduced!(@view(Gtmp[:, 1:pregrav]), cf, ctx, θ, profiled_layout)
            elseif !skip_fill
                materialize_dense_factual_structured!(@view(Gtmp[:, 1:pregrav]), cf)
            end
            grav_raw = compressed_gravity_raw(θ, ctx)
            fill_gravity_column_into!(@view(Gtmp[:, ncore_full]), grav_raw, ctx, ncore_full)
            fill_K_directgp!(K, θ, ctx)
            # port/shared-winner-pair-core-hessian-production-2026-07-25: publish the
            # freshly-built `cf` for `hessian_cm_structured!`/`_v2!` (via `cctx.core_cf_ref`,
            # the SAME shared box) to pick up -- KNITRO always calls the FG/moments! callback
            # at a new point before the first Hessian call there, so this is set before any
            # Hessian callback that needs it runs.
            core_cf_ref[] = cf
            # Profiled all-families completion task (2026-08-01): publish theta alongside cf, same
            # box/lifecycle discipline -- `copy(θ)`, not an alias, since θ is a KNITRO-owned buffer
            # that may be mutated/reused before the Hessian callback (which reads this ref) runs.
            theta_ref[] = copy(θ)
        else
            core_moments!(K, Gtmp, θ, U, obj)
            core_cf_ref[] = :compressed_state_unavailable   # use_compressed_core=false: no winner-form cf built this call, Hessian must fall back to dense
        end
        # skip_fill=true also skips the copy into G -- Gtmp's economic columns were never written
        # this call (undef-backed scratch, not zeroed), so copying them would propagate stale
        # memory into G for no reason; nothing on the production path reads G's economic columns
        # when skip_fill=true (see the root-cause note above).
        if profiled_layout !== nothing || !skip_fill
            @views G[:, 1:pregrav] .= Gtmp[:, 1:pregrav]
        end
        @views G[:, end] .= Gtmp[:, end]
        if profiled_layout !== nothing || !skip_fill
            cm_cols = pregrav + 1 : pregrav + L * nO
            fill_cm_columns_from_bins!(@view(G[:, cm_cols]), Bidx, origins, refIndex1, L, R;
                                        chunk_size = chunk_size, prod_scratch = prod_scratch)
            record_dense_cm_g!()   # Task C (2026-07-27): fires exactly when the dense CM-column fill
            # actually executes -- 0 for the `skip_fill=true` closure variant (installed as
            # cctx.moments_skip!, used only when archC_base_state's own skip_fill_safe -- moment_
            # representation[]==:operator AND inner_fg_backend==:cm_lookup AND cm_cross_hessian_backend
            # ==:winner_bin -- is true), nonzero for the `skip_fill=false` variant (every other case,
            # including archC_verified_state's own inner solve, which always uses the fill variant).
        end
        return nothing
    end
end

"""
    build_cm_augmented_obj_archB(ctx, CS; L, contrasts=:anchored) -> (obj_cm=..., ...)

Architecture-B analogue of `build_cm_augmented_obj`: same returned obj shape
and same `outer_constr_index`/`d` bookkeeping, but the `moments!` closure is
`wrap_moments_with_cm_archB` instead of `wrap_moments_with_cm`. `z`/`origins`
are taken from a throwaway `precalc_common_marginals_cdf` call (so `z` is
IDENTICAL to what Architecture A's reference CM matrix uses -- required for
the correctness comparison to be apples-to-apples) but the dense CM matrix it
returns is discarded immediately (never stored) -- Architecture B's whole
point is to not carry that persistent buffer.

Profiled all-families completion task (2026-08-01): `base_obj` (optional,
default `nothing` = old behavior exactly) lets a caller supply an alternate
`ctx.obj`-shaped object to read `ncore`/`outer_constr_index` bookkeeping from
instead of `ctx.obj` itself -- e.g. `build_reduced_base_obj_for_family(ctx,
layout, CS)` (profiled_restricted_family_base_2026-08-01.jl), whose `.d` is
the profiled/reduced economic width `1+layout.total_reduced_economic_moments`
rather than the full `D*Ddest+2`. Every other input (`Bidx`/`z`/`origins`/
`ctx` itself, used for the moments!-closure fallback and `precalc_common_
marginals_cdf`) is UNCHANGED regardless of `base_obj` -- only the economic
dual-dimension bookkeeping (`ncore`, hence `d_new`/`outer_constr_index_new`,
hence `wrap_moments_with_cm_archB`'s own `pregrav`/`cm_cols` offset) shrinks.
No other line in this function or `wrap_moments_with_cm_archB` needed to
change for this: both already read the economic width as a plain parameter,
never a hardcoded `D*Ddest`-shaped constant (confirmed by direct read, not
assumed -- see profiled_restricted_family_base_2026-08-01.jl's own header for
the full derivation of why this is safe on the production `skip_fill=true`
path).
"""
function build_cm_augmented_obj_archB(ctx, CS; L::Int, contrasts::Symbol = :anchored,
                                       refIndex1::Int = ctx.γ.refIndex1, chunk_size::Int = 2000,
                                       base_obj = nothing, profiled_layout = nothing)
    obj0 = base_obj === nothing ? ctx.obj : base_obj
    ncore = obj0.d
    _CM_throwaway, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; contrasts = contrasts)
    ncm = L * length(origins)
    @assert ncm == n_cm_moments(ctx.D, L)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(ctx.D) : nothing
    # Profiled all-families completion task (2026-08-01): found while exercising this function
    # DIRECTLY (a new, direct `base_obj`-keyword call site this task adds; the only pre-existing
    # caller, build_cm_production_context, apparently sidesteps this) -- explicit `Matrix{Int}(...)`
    # coercion, was a bare `compute_bin_indices(ctx.U, z)`. Two methods of that name exist
    # (common_marginals_interval.jl:72, `z::Vector{Float64}` -- MORE specific, returns a compact
    # `Matrix{bin_index_dtype(L)}` i.e. UInt8/UInt16; cm_hessian_architectures.jl:104,
    # `z::AbstractVector{Float64}`, returns `Matrix{Int}`); `z` here is a plain `Vector{Float64}`
    # (from `precalc_common_marginals_cdf`), so Julia's dispatch picks the FIRST, narrower-dtype
    # method whenever both files are included -- which then fails `wrap_moments_with_cm_archB`'s
    # declared `Bidx::Matrix{Int}` parameter with a `MethodError` (confirmed live). Pre-existing,
    # latent, unrelated to this task's own `base_obj` addition; harmless value-preserving widening.
    Bidx = Matrix{Int}(compute_bin_indices(ctx.U, z))

    d_new = ncore + ncm
    outer_constr_index_new = obj0.outer_constr_index + ncm
    # port/shared-winner-pair-core-hessian-production-2026-07-25: shared box the moments! closure
    # publishes its freshly-built `cf` into, for `build_cm_bin_ctx`/`hessian_cm_structured!` to
    # pick up -- see CMBinHessCtx's own `core_cf_ref` field docstring for the full rationale.
    core_cf_ref = Ref{Any}(nothing)
    # Profiled all-families completion task (2026-08-01): sibling shared box, same lifecycle,
    # publishing the current outer theta -- see CMBinHessCtx's own `profiled_theta_ref` docstring.
    theta_ref = Ref{Any}(nothing)
    moments_cm! = wrap_moments_with_cm_archB(obj0.moments!, ncore, Bidx, origins, refIndex1, L, R, ctx;
        chunk_size = chunk_size, core_cf_ref = core_cf_ref, theta_ref = theta_ref, profiled_layout = profiled_layout)

    obj_cm = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, (moments!) = moments_cm!, moments_jacobian! = error,
        d = d_new, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,   # 2026-07-24 release fix: was defaulting to Inf (disabled) on every rebuild
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
    @assert obj_cm.outer_constr_index == obj_cm.d

    return (obj_cm = obj_cm, z = z, origins = origins, ncore = ncore, ncm = ncm, L = L,
            contrasts = contrasts, refIndex1 = refIndex1, Bidx = Bidx, core_cf_ref = core_cf_ref,
            theta_ref = theta_ref)
end

# ============================================================================
# ARCHITECTURE C: exact structured Hessian via weighted bin contingency
# tables. See docs/fullA_cm_hessian_architecture_report.md sec 2 for the
# full derivation; summary:
#   bin(u) = searchsortedfirst(z,u) in 1:(L+1);  1{u<=z_l} == (bin(u)<=l)
#   T_xy(k,h) = sum_s w_s 1{bin(U[s,x])=k} 1{bin(U[s,y])=h}     (D x D bin tables)
#   S_x(j,k)  = sum_s w_s E[s,j] 1{bin(U[s,x])=k}                (per-origin,
#                                                                  per-econ-col)
#   CT_xy(l,l') = sum_{k<=l,h<=l'} T_xy(k,h)   (2D prefix sum, O(L^2) per pair)
#   CS_x(j,l)   = sum_{k<=l} S_x(j,k)          (1D prefix sum)
# raw (anchored, R=I) blocks:
#   H_EE       = (1/M) E'DE                        (small dense BLAS, as usual)
#   H_EC[j,(o,l)]     = (1/M)[CS_o(j,l) - CS_ref(j,l)]
#   H_CC[(o,l),(p,l')] = (1/M)[CT_op(l,l') - CT_o,ref(l,l') - CT_ref,p(l,l') + CT_ref,ref(l,l')]
# orthonormal contrasts: CM = raw*R within each threshold block (nO x nO
# right-multiply), so H_EC_final[:,block_l] = H_EC_raw[:,block_l]*R and
# H_CC_final[block_l,block_l'] = R' * H_CC_raw[block_l,block_l'] * R
# (congruence, per threshold-block pair).
#
# Per-Hessian-call cost: O(W*(D*NCORE + D^2)) to build T/S tables (bin
# indices Bidx are PRECOMPUTED once, theta/weight-independent) + O(D^2*L^2 +
# D*NCORE*L) to prefix-sum + assemble the dense (NCORE+ncm)^2 output (this
# LAST step is unavoidably O((NCORE+ncm)^2) since KNITRO's dense callback API
# demands the full matrix -- Architecture C only cheapens the INGREDIENT
# computation, not the final materialization).
# ============================================================================

mutable struct CMBinHessCtx
    L::Int
    D::Int
    nO::Int
    origins::Vector{Int}
    refIndex1::Int
    z::Vector{Float64}
    Bidx::Matrix{Int}          # W x D
    NCORE::Int
    ncm::Int
    contrasts::Symbol
    R::Union{Nothing,Matrix{Float64}}
    # scratch, rebuilt every call (sized once)
    Ttab::Array{Float64,4}     # D x D x (L+1) x (L+1)
    Stab::Array{Float64,3}     # D x NCORE x (L+1)
    CT::Array{Float64,4}       # D x D x L x L  (prefix-summed, 1:L only)
    CScum::Array{Float64,3}    # D x NCORE x L
    Ews::Matrix{Float64}       # W x NCORE scratch for sqrt(w)-scaled E
    Hfull::Matrix{Float64}     # (NCORE+ncm) x (NCORE+ncm) scratch
    # Allocation/Hessian port task §4.2: NCORE x nO scratch for the H_EC raw block and its
    # (optional) R-congruence product -- were `Matrix{Float64}(undef,...)` (Hraw_EC) allocated
    # once per hessian_cm_structured! call, and `Hraw_EC * cctx.R` (block_ec) allocated FRESH on
    # every one of the L=50 threshold-block iterations within that same call. `block_ec` is
    # `nothing` when `R === nothing` (that branch uses Hraw_EC directly, no product needed).
    Hraw_EC::Matrix{Float64}
    block_ec::Union{Nothing,Matrix{Float64}}
    # Same fix, found live while investigating §6.1's threaded Hessian port: `Hraw_CC` (nO x nO)
    # was ALSO reallocated once per call (missed in the original §4.2 pass, which only named
    # Hraw_EC/block_ec), and its OWN R-congruence product `cctx.R' * Hraw_CC * cctx.R` allocated
    # TWO fresh matrices (the R'*Hraw_CC intermediate, then the final product) on EVERY one of the
    # L^2=2500 (l,l') threshold-block-PAIR iterations within that same call -- 50x more iterations
    # than H_EC's own L=50, making this potentially the single largest CM Hessian-callback
    # allocation site, larger than the one actually named in the original audit.
    Hraw_CC::Matrix{Float64}
    RtHraw_CC::Union{Nothing,Matrix{Float64}}
    block_cc::Union{Nothing,Matrix{Float64}}
    # Allocation/Hessian port task §6.1/6.2: threaded Architecture-C Hessian, validated (see
    # test_cm_threaded_hessian.jl) to agree with the serial Hessian to ~1e-13 and measured 3.52x
    # faster at a real, hard D=20/W=80,000/L=50 point (5.39s/call serial -> 1.53s/call threaded).
    # `tls` is `nothing` (and use_threaded_bins is false) only if a caller explicitly opts out via
    # build_cm_bin_ctx(...; threaded_bins=false) -- production default is true.
    tls::Union{Nothing,ThreadLocalBinScratch}
    use_threaded_bins::Bool
    # port/shared-winner-pair-core-hessian-production-2026-07-25 (task §4.2/§4.3): H_EE now goes
    # through the shared exact winner-pair backend instead of a small dense BLAS gemm/syrk. The
    # `CompressedFactual` needed to build the winner-pair workspace is theta-DEPENDENT (winner
    # assignments change every outer point) but `CMBinHessCtx` itself is built ONCE PER CAMPAIGN
    # (`build_cm_production_context`'s own docstring: "reused for every subsequent inner solve...
    # bin indices are fixed once the draws U are fixed -- independent of theta"), so the workspace
    # cannot be precomputed here. Instead `core_cf_ref` is a SHARED box also closed over by
    # `wrap_moments_with_cm_archB`'s moments! closure (same pattern this codebase already uses to
    # pass FG-computed state to the Hessian callback, e.g. compressed_live.jl's
    # `obj.arg0 .= q` written by the FG callback for the Hessian callback to read) -- every moments!
    # call (which KNITRO always issues before the first Hessian call at a new point) refreshes
    # `core_cf_ref[]` with the freshly-built `cf` for the CURRENT theta; `hessian_cm_structured!`/
    # `_v2!` below rebuild `core_ws` only when the `cf` object identity changes (`core_ws_for`).
    # `core_cf_ref[]` is `nothing` (falls back to dense BLAS unconditionally) whenever the caller
    # built `aug` with `use_compressed_core=false` or via the non-archB `build_cm_augmented_obj`.
    core_cf_ref::Ref{Any}
    core_ws::Union{Nothing,CoreExactHessianWorkspace}
    core_ws_for::Any
    core_hessian_backend::Symbol
    core_hessian_workers::Int
    core_hessian_storage::Symbol
    # CM+mean/ZC widens NCORE to include the mean/pair columns IN THE SAME dense "economic" block
    # (cm_meanzc_moments.jl's own column layout -- mean/pair are NOT a separate CM-grid-style
    # restriction block there). The true winner-pair core is only the first `ncore_core` of those
    # `NCORE` columns; `ncore_core == NCORE` for plain flexible CM (no widening).
    ncore_core::Int
    # Remediation task Phase B1 (production-audit continuation, 2026-07-26): which inner FG
    # (forward/backward) callback the KNITRO inner dual solve registers. :dense_reference (default
    # -- unchanged, byte-identical to every pre-existing production run) | :cm_lookup (the
    # validated O(W*(D-1)) lookup kernel, cm_lookup_kernels.jl/cm_lookup_production.jl -- ONLY
    # valid for plain flexible CM, marginal_restriction=:common_flexible + cm_extension=:cm_only;
    # callers must not set this for common_frechet or meanzc configs, see
    # cm_lookup_production.jl's own header). Hessian math is completely unaffected either way --
    # this field only selects the FG callback, archC_hess_cb_builder(cctx) is reused unmodified.
    inner_fg_backend::Symbol
    # Phase 5.5 remediation (2026-07-26, matrix-free operator FG allocation fix): lazily-built
    # CMLookupState (cm_lookup_kernels.jl), reused across EVERY subsequent :cm_lookup inner solve
    # at this context (bins/origins/refIndex1/R/NCORE/ncm are all fixed once `cctx` is built --
    # same "built once per campaign, reused every inner solve" invariant this struct's own
    # `core_ws`/`core_ws_for` fields already rely on above). `nothing` until the first :cm_lookup
    # inner solve; typed `Any` (not `CMLookupState`) purely to avoid a forward type reference,
    # since cm_lookup_kernels.jl is included after this file.
    cmlookup_st::Any
    # skip_cm_fill_ref removal (2026-07-27): was a mutable `Ref{Bool}` toggled true/false around
    # each inner solve by archC_base_state/archC_verified_state (cm_production_bundle.jl). Replaced
    # by this field, which holds the SECOND, `skip_fill=true`-baked-in `wrap_moments_with_cm_archB`
    # (flexible-CM) / `wrap_moments_with_cm_frechet_archB` (common-Fréchet) closure -- built, but as
    # of 2026-07-28 never actually invoked in production for EITHER family (a real D=20/W=80,000
    # multi-point re-test found the skip reproduces a genuine nStatus=-400 solve failure at
    # non-calibration points for both, see cm_production_bundle.jl::archC_base_state's own HISTORY
    # comment) -- a plain immutable function reference, not a mutable shared box. `inner_loop_internal_cmlookup_production` /
    # `inner_loop_internal_cmfrechetlookup_production`'s priming call selects between `obj.moments!`
    # (the default, `skip_fill=false` closure, always used) and this field (used ONLY when its own
    # caller explicitly threads `skip_fill=true` through, i.e. archC_base_state/
    # archC_frechet_base_state under their own `skip_fill_safe`) -- an explicit per-call argument,
    # not a read of ambient mutable state. `nothing` when this cctx's aug wasn't built with an archB
    # skip variant at all (dense/non-archB paths, CM+ZC -- its own `wrap_moments_with_cm_meanzc` is a
    # separate closure that never builds a skip variant; see build_cm_bin_ctx's own hasproperty
    # fallback below).
    moments_skip!::Union{Nothing,Function}
    # port/shared-inner-fg-operator-and-verification-2026-07-26: CM+ZC's operator FG
    # (`inner_fg_backend=:operator`, cm_meanzc_lookup_kernels.jl/cm_meanzc_lookup_production.jl)
    # needs its own `ZCRestrictionOperator`/`SharedByPowerLayout` (built once, campaign-lifetime,
    # from `aug.Zraw_all`/`Zpairraw_all`/`K_mean`/`K_pair`) -- `Any`-typed for the same load-order
    # reason `cmlookup_st` is (this file has no dependency on zc_restriction_operator.jl unless a
    # caller actually requests `:operator`). `nothing` for plain CM and common Frechet (their
    # `build_cm_bin_ctx`/`build_cm_bin_ctx`-via-Frechet callers never set these).
    meanzc_zc_op::Any
    meanzc_zc_layout::Any
    # Winner-aware H_ER phase (2026-07-27, task Section 2): which backend fills the economic x
    # CM-restriction cross block (H_EC). :dense_reference (default until this phase's own gates
    # pass) | :winner_bin (winner_pair_cross_hessian.jl, reuses the SAME WinnerPairHessCtx H_EE
    # already builds via core_ws -- no separate winner-pair precompute). Only ever taken when
    # ncore_core == NCORE (plain flexible CM / common-Frechet's CM-grid block widening does NOT
    # apply here -- CM+mean/pair-ZC's ncore_core < NCORE widened layout is explicitly out of this
    # backend's validated scope, see task Section 4).
    cm_cross_hessian_backend::Symbol
    # Persistent scratch for the :winner_bin backend, rebuilt only on a (ncolI,D,L) size change
    # (campaign-lifetime constant in practice) -- NOT per Hessian callback, matching this file's
    # existing "built once, reused every call" discipline for core_ws/tls.
    cross_scratch::Union{Nothing,WinnerBinCrossScratch}
    # Winner-aware H_ER phase (2026-07-27, task Section 4): which backend fills the economic x
    # mean/pairwise-ZC cross block H_EM (CM+ZC's own `HEM = HEE[1:ncore_core, ncore_core+1:NCORE]`,
    # `_fill_cm_HEE!`'s `ncore < NCORE` branch). :dense_reference (default until this section's own
    # gates pass) | :winner_bin (winner_pair_cross_hessian_zc_block!, winner_pair_cross_hessian.jl
    # -- reuses the SAME core_ws/wctx H_EE already builds). Only ever taken when `ncore_core <
    # NCORE` (there IS a widened mean/pair block to fill at all -- plain flexible CM / common-
    # Frechet never reach this branch, `ncore_core == NCORE` there). Always `:dense_reference` for
    # plain CM (`build_cm_bin_ctx`, no ZC widening ever exists there).
    zc_cross_hessian_backend::Symbol
    # Persistent scratch for the ZC :winner_bin backend (WinnerZCCrossScratch, distinct struct from
    # the CM-grid's WinnerBinCrossScratch -- no L-threshold binning, see winner_pair_cross_hessian.jl
    # header), rebuilt only on a (W, n_restr) size change.
    zc_cross_scratch::Union{Nothing,WinnerZCCrossScratch}
    # CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): DEDICATED raw-ZC-feature state
    # for the NEW direct H_CZ/H_ZZ primitives -- intentionally SEPARATE objects from
    # `meanzc_zc_op`/`meanzc_zc_layout` above (those are gated on `inner_fg_backend===:operator` and
    # several call sites use `meanzc_zc_op !== nothing` as a "was this built with :operator"
    # prerequisite check; repurposing them for the Hessian's own use, which must be available
    # regardless of inner_fg_backend, would silently defeat those checks). `hzz_zc_op`/
    # `hzz_zc_layout` built ONCE (campaign-lifetime, `build_cm_meanzc_bin_ctx`, ALWAYS -- unlike
    # `meanzc_zc_op`), `nothing` only for plain CM/common-Frechet (`build_cm_bin_ctx`, no ZC block
    # at all). `hzz_zc_ws` (ZCRestrictionWorkspace) holds the CURRENT outer point's refreshed
    # targets, `nu_ref` is the shared box `archC_meanzc_base_state`/`_verified_state`
    # (cm_meanzc_production.jl) publish the current νvec into (mirrors `core_cf_ref`'s own
    # "wrapper publishes, Hessian callback reads" pattern -- explicit here since those callers
    # already own νvec directly as a function argument, no moments! closure round-trip needed).
    hzz_zc_op::Any
    hzz_zc_layout::Any
    hzz_zc_ws::Any
    nu_ref::Base.RefValue{Vector{Float64}}
    hzz_centered::Union{Nothing,ZCCenteredScratch}
    bin_zc_cross::Union{Nothing,BinZCrossScratch}
    # optimize/structured-cross-hessian-ZC-CM-2026-07-28: opt-in threading for the H_EC/H_EZ/H_CZ/
    # H_ZZ raw-table-fill primitives above (`threaded_cross_hessian.jl`). Mirrors `use_threaded_bins`'s
    # own "field on the ctx, defaulted from a global Ref, flippable per-context" discipline exactly.
    # Defaults to `false` (opt-in) until the performance/correctness gates in this task's own
    # deliverable docs justify flipping the default -- see `CROSS_HESSIAN_CROSS_THREADED_DEFAULT`.
    cross_hessian_threaded::Bool
    cross_hessian_workers::Int
    # optimize/structured-cross-hessian-ZC-CM-2026-07-28 ADDENDUM: H_ZZ backend selector + its own
    # persistent raw-Phi workspace (zc_gram_blas_candidates.jl). `raw_zc_ws` is `nothing` for plain
    # CM/common-Frechet (no ZC block, `hzz_zc_op===nothing`) and lazily built (once, campaign-
    # lifetime) on first use for CM+ZC.
    zc_gram_backend::Symbol
    zc_gram_workers::Int
    raw_zc_ws::Union{Nothing,ZCRawWeightedWorkspace}
    # True no-H operator bundle (2026-07-28 continuation): the base economic ctx (has `.γ`, `.D`,
    # etc.), needed by `prime_operator!` (cf_build/fill_K_directgp!/compressed_gravity_raw all take
    # `ctx`, not `cctx`). Every existing `moments!` closure (`wrap_moments_with_cm_archB` etc.)
    # already closes over this same `ctx` object directly -- this field just gives
    # `inner_loop_internal_cmlookup_production` (which only receives `cctx`, not `ctx`) the same
    # access without threading a new argument through every call site.
    econ_ctx::Any
    # Harmonization task (2026-07-28): lazily-built, persistent CMFrechetExtension (level_targets +
    # level-block Hessian scratch), reused across EVERY subsequent common-Fréchet Hessian callback
    # at this cctx -- same "typed Any purely to avoid a forward reference, nothing until first use"
    # pattern as `cmlookup_st` above (CMFrechetExtension is defined in cm_frechet_hessian.jl,
    # included after this file). `nothing` for every other family (plain CM/CM+ZC/origin-ZC/
    # unrestricted never populate this). Kept as a cache resolved inside
    # `archC_frechet_hess_cb_builder` rather than threaded through every one of that function's own
    # (many, historical) callers as a new required argument -- this field is the substitute for
    # widening that public call chain.
    frechet_ext_cache::Any
    # diagnose-optimize/HZZ-BLAS-and-HCZ-prep-2026-07-29 Part C: H_CZ prep backend selector +
    # persistent scratch, same "global-Ref default, opt-in per-cctx override" pattern as
    # `zc_gram_backend` above. `:origin_owned` (existing bin_zc_cross_hessian_fill!/_threaded!,
    # unchanged) | `:draw_chunk_thread_local` (hcz_drawchunk_candidate_2026-07-29.jl, 12-14x
    # faster at real D=20/W=100k per docs/PART_C_HCZ_CANDIDATE_RESULTS_2026-07-29.md, tolerance-
    # level correct -- not yet the default pending a complete-inner-solve gate).
    hcz_prep_backend::Symbol
    bin_zc_drawchunk::Any
    # Genuine-cold ZC Hessian K=3 optimization task (2026-08-01): H_ER (=H_EZ) backend selector +
    # persistent scratch, same "global-Ref default, opt-in per-cctx override" pattern as
    # `zc_gram_backend`/`hcz_prep_backend` above. `:winner_bin` (existing
    # winner_pair_cross_hessian_zc_block!/_threaded!, unchanged, dispatched on
    # `cross_hessian_threaded`) | `:drawmajor` (hez_drawmajor_candidate_2026-08-01.jl -- partitions
    # DRAWS instead of destination slots, so Z is read once per (w,x) instead of Ddest times; not
    # yet the default pending a genuine-cold complete-solve gate).
    zc_ez_backend::Symbol
    zc_drawmajor::Any
    # Profiled all-families completion task (2026-08-01): `profiled_layout` is `nothing` for every
    # existing caller (plain/OLD full economic layout, unchanged behavior) or a
    # `ProfiledEconomicMomentLayout` when this `cctx` was built for the REDUCED economic block
    # (`build_cm_augmented_obj_archB(...; base_obj=build_reduced_base_obj_for_family(...))`).
    # `profiled_theta_ref` is a shared `Ref{Any}` box (mirrors `core_cf_ref`'s own "moments! closure
    # publishes, Hessian callback reads" pattern exactly) holding the CURRENT outer point's full
    # theta vector -- needed by `build_reduced_homogeneous_winner_pair_ctx`, which `_fill_cm_HEE!`
    # cannot otherwise reach (unlike `cf`, theta is not derivable from anything already threaded to
    # the Hessian callback). `profiled_reduced_wctx`/`profiled_reduced_wctx_for` cache the built
    # `ReducedHomogeneousWinnerPairHessCtx`, rebuilt only when the `cf` identity changes -- same
    # "rebuild only on a new outer point" discipline as `core_ws`/`core_ws_for`.
    # `profiled_hee_packed` is persistent packed-Hessian scratch (row-major upper-triangle, length
    # `n(n+1)/2` for `n=1+layout.total_reduced_economic_moments`) for
    # `reduced_homogeneous_winner_pair_hessian!`'s own packed-output convention, unpacked into the
    # caller's dense `HEE` view -- sized once, resized only if `layout` itself changes (campaign-
    # lifetime constant in practice, exactly like `Hfull`/`Ews` above).
    profiled_layout::Any
    profiled_theta_ref::Base.RefValue{Any}
    profiled_reduced_wctx::Any
    profiled_reduced_wctx_for::Any
    profiled_hee_packed::Vector{Float64}
    # Profiled all-families completion task (2026-08-01), H_EC gather step: the "gather retained
    # rows at assembly" design (see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md) rebuilds a
    # FULL-width (unreduced) `WinnerPairHessCtx`/`WinnerBinCrossScratch` from the SAME `cf` every
    # outer point -- deliberately NOT the same object as `core_ws`/`cross_scratch` above (those, if
    # present at all for this cctx, are sized/keyed for the OLD non-profiled path and must not be
    # reused/aliased here). `profiled_full_wctx`/`profiled_full_wctx_for` cache the full `wctx`
    # (rebuild only on `cf` identity change, same discipline as `core_ws`); `profiled_full_ws` is its
    # `WinnerBinCrossScratch` (sized to the FULL `D*Ddest(+cf)` width, NOT the reduced `NCORE` --
    # genuinely larger than `cross_scratch` would be, since the cross-block computation itself is
    # NOT reduced by this design, only the packed output is -- see the master doc's own "not
    # FLOP-optimal for the cross-block" note). `profiled_hraw_ec_full` is the corresponding
    # full-width `(wctx.ncolI+1) x nO` raw-block scratch, gathered down into the existing (already
    # reduced-sized) `Hraw_EC` field above before the R-congruence/Hfull-write step.
    profiled_full_wctx::Any
    profiled_full_wctx_for::Any
    profiled_full_ws::Union{Nothing,WinnerBinCrossScratch}
    profiled_hraw_ec_full::Matrix{Float64}
    # Profiled restricted-inner-endtoend task (2026-08-01), H_EF profiled gather step: full-width
    # `(wctx_full.ncolI+1)`-length scratch for `winner_pair_cross_hessian_colsum!`/`_esum!` (common
    # Fréchet's H_E,level block), gathered down into `CMFrechetExtension.colsum`/`.Esum_wb`
    # (already reduced-sized, `cctx.NCORE`) before the `H_E,level` write -- same "gather at
    # assembly" design as `profiled_hraw_ec_full` above, reusing the SAME `profiled_full_wctx`/
    # `profiled_full_ws` (rebuilt/cached only on `cf` identity change, not duplicated here).
    profiled_colsum_full::Vector{Float64}
    profiled_esum_full::Vector{Float64}
    # ZC lane task (2026-08-02), CM+ZC widened-core H_EM gather: `profiled_full_ws_zc` is the ZC-
    # specific full-width scratch (`WinnerZCCrossScratch`, from `winner_pair_cross_hessian_zc_prep!`
    # -- a DIFFERENT type than flexible-CM's own `profiled_full_ws::WinnerBinCrossScratch` above,
    # hence a separate field rather than a reuse) for the mean/pair-Z cross block H_EM (this family's
    # own "H_EZ", filled via the SAME `winner_pair_cross_hessian_zc_block!(...; use_profiled_
    # correction=true)` kernel origin-ZC's H_EZ gather already uses -- reuses `profiled_full_wctx`/
    # `profiled_full_wctx_for` above unchanged, same `cf`-identity-keyed rebuild discipline).
    # `profiled_hraw_em_full` is the corresponding full-width `(wctx.ncolI+1) x n_restr` raw-block
    # scratch, gathered down into the reduced-sized `HEM` view before the H_EE-block write.
    profiled_full_ws_zc::Union{Nothing,WinnerZCCrossScratch}
    profiled_hraw_em_full::Matrix{Float64}
end

"Outer constructor: forwards to the full positional inner constructor, appending the new H_CZ prep backend fields with their defaults so neither existing CMBinHessCtx(...) call site (build_cm_bin_ctx/build_cm_meanzc_bin_ctx) needs to change. Also appends the genuine-cold H_EZ backend selector fields (zc_ez_backend/zc_drawmajor), the profiled all-families completion task's reduced-layout fields, and the ZC lane task's CM+ZC widened-core H_EM gather fields, all with their own defaults (nothing/fresh-Ref/empty-buffer), same backward-compatibility discipline throughout."
function CMBinHessCtx(args...; hcz_prep_backend::Symbol = HCZ_PREP_BACKEND_DEFAULT[], bin_zc_drawchunk = nothing,
        zc_ez_backend::Symbol = ZC_EZ_BACKEND_DEFAULT[], zc_drawmajor = nothing,
        profiled_layout = nothing, profiled_theta_ref::Base.RefValue{Any} = Ref{Any}(nothing),
        profiled_reduced_wctx = nothing, profiled_reduced_wctx_for = nothing,
        profiled_hee_packed::Vector{Float64} = Float64[],
        profiled_full_wctx = nothing, profiled_full_wctx_for = nothing,
        profiled_full_ws::Union{Nothing,WinnerBinCrossScratch} = nothing,
        profiled_hraw_ec_full::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
        profiled_colsum_full::Vector{Float64} = Float64[],
        profiled_esum_full::Vector{Float64} = Float64[],
        profiled_full_ws_zc::Union{Nothing,WinnerZCCrossScratch} = nothing,
        profiled_hraw_em_full::Matrix{Float64} = Matrix{Float64}(undef, 0, 0))
    return CMBinHessCtx(args..., hcz_prep_backend, bin_zc_drawchunk, zc_ez_backend, zc_drawmajor,
        profiled_layout, profiled_theta_ref, profiled_reduced_wctx, profiled_reduced_wctx_for, profiled_hee_packed,
        profiled_full_wctx, profiled_full_wctx_for, profiled_full_ws, profiled_hraw_ec_full,
        profiled_colsum_full, profiled_esum_full, profiled_full_ws_zc, profiled_hraw_em_full)
end

"""
    build_cm_bin_ctx(ctx, aug) -> CMBinHessCtx

Builds the Architecture-C precomputation (bin indices + scratch buffers) for
an existing `aug = build_cm_augmented_obj(...)` result. `z`/`origins` are
taken from `aug` itself so this is guaranteed to use IDENTICAL thresholds to
whatever CM matrix Architecture A is using.
"""
function build_cm_bin_ctx(ctx, aug; threaded_bins::Bool = true,
        core_hessian_backend::Symbol = CM_CORE_HESSIAN_BACKEND_DEFAULT[],
        core_hessian_workers::Int = CM_CORE_HESSIAN_WORKERS_DEFAULT[], core_hessian_storage::Symbol = CM_CORE_HESSIAN_STORAGE_DEFAULT[],
        inner_fg_backend::Symbol = CM_INNER_FG_BACKEND_DEFAULT[],
        cm_cross_hessian_backend::Symbol = CM_CROSS_HESSIAN_BACKEND_DEFAULT[],
        cross_hessian_threaded::Bool = CROSS_HESSIAN_THREADED_DEFAULT[],
        cross_hessian_workers::Int = CROSS_HESSIAN_WORKERS_DEFAULT[],
        zc_gram_backend::Symbol = ZC_GRAM_BACKEND_DEFAULT[],
        zc_gram_workers::Int = ZC_GRAM_THREADED_WORKERS_DEFAULT[],
        profiled_layout = nothing)
    L = aug.L; D = ctx.D; origins = aug.origins; nO = length(origins)
    refIndex1 = aug.refIndex1; z = aug.z
    NCORE = aug.ncore; ncm = aug.ncm
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    Bidx = compute_bin_indices(ctx.U, z)
    W = size(ctx.U, 1)
    L1 = L + 1
    # port/shared-winner-pair-core-hessian-production-2026-07-25: `aug.core_cf_ref` exists only
    # when `aug` came from `build_cm_augmented_obj_archB(...; use_compressed_core=true)` (the
    # production default); anything else (the plain `build_cm_augmented_obj`, or
    # `use_compressed_core=false`) falls back to dense BLAS for H_EE unconditionally.
    core_cf_ref = hasproperty(aug, :core_cf_ref) ? aug.core_cf_ref : Ref{Any}(nothing)
    # skip_cm_fill_ref removal (2026-07-27): `aug.moments_skip!`, if present, is the SECOND
    # `skip_fill=true`-baked-in closure `build_cm_production_context` built alongside the default
    # (`skip_fill=false`) one installed as `ctx_cm.obj.moments!` -- absent (e.g. non-archB aug,
    # diagnostic scripts, common-Fréchet, CM+ZC, none of which ever build a skip variant) defaults
    # to `nothing`, so `inner_loop_internal_cmlookup_production`'s `skip_fill=true`-argument branch
    # falls back to the always-fill `obj.moments!` (see that function's own dispatch).
    moments_skip_fn = hasproperty(aug, :moments_skip!) ? aug.moments_skip! : nothing
    # Profiled all-families completion task (2026-08-01): `aug.theta_ref`, if present (only when
    # `aug` came from `build_cm_augmented_obj_archB(...; use_compressed_core=true)`, the same
    # precondition as `core_cf_ref` above), is the shared box `wrap_moments_with_cm_archB`'s closure
    # publishes the current outer theta into -- needed by the reduced H_EE kernel
    # (`build_reduced_homogeneous_winner_pair_ctx`) when `profiled_layout !== nothing`. Absent for
    # every pre-existing `aug` (non-archB, or archB before this task), defaults to an unused fresh
    # `Ref{Any}(nothing)` -- harmless since `profiled_layout===nothing` means it's never read.
    profiled_theta_ref = hasproperty(aug, :theta_ref) ? aug.theta_ref : Ref{Any}(nothing)
    cctx = CMBinHessCtx(L, D, nO, origins, refIndex1, z, Bidx, NCORE, ncm, aug.contrasts, R,
        zeros(D, D, L1, L1), zeros(D, NCORE, L1), zeros(D, D, L, L), zeros(D, NCORE, L),
        Matrix{Float64}(undef, W, NCORE), Matrix{Float64}(undef, NCORE + ncm, NCORE + ncm),
        Matrix{Float64}(undef, NCORE, nO), R === nothing ? nothing : Matrix{Float64}(undef, NCORE, nO),
        Matrix{Float64}(undef, nO, nO), R === nothing ? nothing : Matrix{Float64}(undef, nO, nO), R === nothing ? nothing : Matrix{Float64}(undef, nO, nO),
        nothing, false,
        core_cf_ref, nothing, nothing, core_hessian_backend, core_hessian_workers, core_hessian_storage,
        NCORE, inner_fg_backend, nothing, moments_skip_fn, nothing, nothing,
        cm_cross_hessian_backend, nothing,
        :dense_reference, nothing,   # zc_cross_hessian_backend/zc_cross_scratch: plain CM never widens (no ZC block)
        nothing, nothing, nothing, Ref(Float64[]), nothing, nothing,   # hzz_zc_op/layout/ws/nu_ref/hzz_centered/bin_zc_cross: plain CM has no ZC block at all
        cross_hessian_threaded, cross_hessian_workers,
        zc_gram_backend, zc_gram_workers, nothing,   # raw_zc_ws: plain CM has no ZC block, lazily unused
        ctx,   # econ_ctx: true no-H operator bundle continuation
        nothing;   # frechet_ext_cache: harmonization task -- lazily built, nothing until first common-Fréchet Hessian call
        profiled_layout = profiled_layout, profiled_theta_ref = profiled_theta_ref)
    if threaded_bins
        cctx.tls = build_thread_local_scratch(cctx)
        cctx.use_threaded_bins = true
    end
    return cctx
end

"""
Build the D x D and D x NCORE x (L+1) weighted bin tables from CURRENT weights `w` (obj.arg2) and
economic block `E`. O(W*(D*NCORE + D^2)).

`fill_S=false` (winner-aware H_ER phase, 2026-07-27) skips the `S[x,j,bx] += ws*E[s,j]` inner loop
entirely -- the ONLY reader of `E` in this function -- so when the caller is about to fill `H_EC`
via the `:winner_bin` cross-Hessian backend instead (which never reads `E`/`obj.H`'s dense economic
columns, see winner_pair_cross_hessian.jl), this function performs NO dense economic-column read at
all, only the still-needed T-table accumulation for `H_CC` (untouched by this phase, per task
Section 2's own "do not alter the existing H_RR CM contingency-table block"). `Stab` is still
zeroed (defensive: stale values must never leak into a later `fill_S=true` call at a different
context) but never populated.
"""
function build_bin_tables!(cctx::CMBinHessCtx, H::Union{Nothing,AbstractMatrix{Float64}}, w::AbstractVector{Float64}; fill_S::Bool = true)
    D = cctx.D; NCORE = cctx.NCORE; Bidx = cctx.Bidx
    T = cctx.Ttab; S = cctx.Stab
    fill!(T, 0.0); fill!(S, 0.0)
    W = length(w)
    if fill_S
        # No-moments/no-composite-G task (2026-07-28): `E` is constructed HERE, lazily, only inside
        # the branch that actually reads it -- never at the caller's top level -- so that `H` need
        # not have `NCORE` economic columns at all when `fill_S=false` (the production default,
        # `:winner_bin`), the precondition this task's H-elimination relies on. Legacy-H removal
        # (2026-07-28): `H` is now `Union{Nothing,...}` -- `nothing` for OperatorCMBundle callers,
        # which always pass `fill_S=false` (production default `:winner_bin`), so this branch is
        # provably unreachable for them; the explicit error below is a fail-fast if that invariant
        # is ever violated, not expected to fire.
        H === nothing && error("build_bin_tables!: fill_S=true requested for an operator-mode bundle with no H field -- should be unreachable in production.")
        E = @view H[:, 2:1+NCORE]
        @inbounds for s in 1:W
            ws = w[s]
            for x in 1:D
                bx = Bidx[s, x]
                for j in 1:NCORE
                    S[x, j, bx] += ws * E[s, j]
                end
            end
            for x in 1:D
                bx = Bidx[s, x]
                for y in 1:D
                    by = Bidx[s, y]
                    T[x, y, bx, by] += ws
                end
            end
        end
    else
        @inbounds for s in 1:W
            ws = w[s]
            for x in 1:D
                bx = Bidx[s, x]
                for y in 1:D
                    by = Bidx[s, y]
                    T[x, y, bx, by] += ws
                end
            end
        end
    end
    return nothing
end

"2D-prefix-sum `Ttab` into `CT` (restricted to l,l' in 1:L) and 1D-prefix-sum `Stab` into `CScum`. O(D^2*L^2 + D*NCORE*L). `fill_S=false` (winner-aware H_ER phase) skips the CScum prefix-sum -- `Stab` was never populated by `build_bin_tables!(...; fill_S=false)`, so prefix-summing it would only waste O(D*NCORE*L) work on zeros."
function prefix_sum_tables!(cctx::CMBinHessCtx; fill_S::Bool = true)
    D = cctx.D; L = cctx.L; NCORE = cctx.NCORE
    T = cctx.Ttab; CT = cctx.CT
    @inbounds for x in 1:D, y in 1:D
        for l in 1:L
            for lp in 1:L
                v = T[x, y, l, lp]
                v += (l > 1 ? CT[x, y, l-1, lp] : 0.0)
                v += (lp > 1 ? CT[x, y, l, lp-1] : 0.0)
                v -= (l > 1 && lp > 1) ? CT[x, y, l-1, lp-1] : 0.0
                CT[x, y, l, lp] = v
            end
        end
    end
    if fill_S
        S = cctx.Stab; CS_ = cctx.CScum
        @inbounds for x in 1:D, j in 1:NCORE
            acc = 0.0
            for l in 1:L
                acc += S[x, j, l]
                CS_[x, j, l] = acc
            end
        end
    end
    return nothing
end

"""
    _fill_cm_HEE!(HEE, w, obj, cctx::CMBinHessCtx, E, M)

Shared H_EE fill for BOTH `hessian_cm_structured!` (serial Architecture C)
and `hessian_cm_structured_v2!` (threaded Architecture C, cm_hessian_threaded.jl)
-- task §4.2's "Do not implement a second winner-pair variant for this
family" -- one helper, called from both. Uses the shared exact winner-pair
backend when `cctx.core_cf_ref[]` holds a valid `CompressedFactual` for the
CURRENT theta (production default, `use_compressed_core=true` in
`wrap_moments_with_cm_archB`), rebuilding `cctx.core_ws` only when the `cf`
object identity has changed since the last call (a new outer point); falls
back to the original small dense BLAS gemm on `E` (task §5's named,
explicitly opt-in fallback) whenever no compressed core is available for
this point (`use_compressed_core=false`, or a `TiedWinnerError` this point),
or `cctx.core_hessian_backend === :dense_reference`.
"""
function _fill_cm_HEE!(HEE::AbstractMatrix, w::AbstractVector{Float64}, obj, cctx::CMBinHessCtx, H::Union{Nothing,AbstractMatrix{Float64}}, M)
    ncore = cctx.ncore_core
    NCORE = cctx.NCORE
    cf = cctx.core_cf_ref[]   # a CompressedFactual (success) OR a Symbol fallback reason (:tied_winner / :compressed_state_unavailable)
    # Profiled all-families completion task (2026-08-01): when this cctx was built for the reduced
    # economic layout (build_cm_bin_ctx(...; profiled_layout=layout)), H_EE is filled by the
    # ALREADY-VALIDATED unrestricted-family reduced kernel
    # (reduced_homogeneous_hessian_2026-08-01.jl), never the full winner-pair kernel below -- an
    # early, self-contained branch, so the pre-existing (profiled_layout===nothing) code path below
    # is completely untouched. Only reachable for `ncore == NCORE` (no CM+ZC mean/pair widening --
    # that combination is not ported yet, see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md).
    if cctx.profiled_layout !== nothing
        cf isa CompressedFactual || error("_fill_cm_HEE!: profiled_layout set but core_cf_ref[] is not a CompressedFactual (got $(typeof(cf))) -- the compressed-core fallback path is not supported for the reduced economic layout.")
        layout = cctx.profiled_layout
        θ_full = cctx.profiled_theta_ref[]
        θ_full === nothing && error("_fill_cm_HEE!: profiled_layout set but profiled_theta_ref[] is nothing -- theta was never published for this outer point (moments! closure not yet called before this Hessian callback?).")
        if cctx.profiled_reduced_wctx === nothing || cctx.profiled_reduced_wctx_for !== cf
            cctx.profiled_reduced_wctx = build_reduced_homogeneous_winner_pair_ctx(cf, cctx.econ_ctx, θ_full, layout)
            cctx.profiled_reduced_wctx_for = cf
        end
        wctx_r = cctx.profiled_reduced_wctx
        n = 1 + wctx_r.ncolI
        n == ncore || error("_fill_cm_HEE!: reduced width n=$n (1+layout.total_reduced_economic_moments) != cctx.ncore_core=$ncore -- CMBinHessCtx built inconsistently with this layout.")
        npacked = n * (n + 1) ÷ 2
        length(cctx.profiled_hee_packed) == npacked || (cctx.profiled_hee_packed = Vector{Float64}(undef, npacked))
        reduced_homogeneous_winner_pair_hessian!(cctx.profiled_hee_packed, obj, wctx_r)
        HEE_core = @view HEE[1:ncore, 1:ncore]
        packed = cctx.profiled_hee_packed
        k = 1
        @inbounds for i in 1:n
            for j in i:n
                v = packed[k]; k += 1
                HEE_core[i, j] = v
                HEE_core[j, i] = v
            end
        end
        # ZC lane task (2026-08-02): CM+ZC widened-core H_EM/H_MM. H_EE above is genuinely family-
        # independent (economic block only) -- reused verbatim. When ncore<NCORE (a real mean/pair-Z
        # widened block exists, i.e. this cctx is CM+ZC's, not plain flexible CM/common-Frechet),
        # H_EM is filled via the SAME "gather at assembly" design origin-ZC's own H_EZ gather uses
        # (archA_partitioned_hess_cb_builder): a FULL-width wctx/ws built once per callback via
        # winner_pair_cross_hessian_zc_prep!/_zc_block!(use_profiled_correction=true), gathered into
        # the reduced-sized HEM view. H_MM is copied VERBATIM from the non-profiled ncore<NCORE
        # branch below (depends only on cctx.hzz_zc_op/cctx.nu_ref, confirmed independent of economic
        # block width by direct read -- matches this task's own "unchanged H_ZZ" instruction, HMM
        # here being CM+ZC's own name for that same restriction-self-block).
        if ncore < NCORE
            HEM = @view HEE[1:ncore, ncore+1:NCORE]
            HMM = @view HEE[ncore+1:NCORE, ncore+1:NCORE]
            n_restr = NCORE - ncore
            cctx.hzz_zc_op !== nothing || error("_fill_cm_HEE!: profiled_layout set with ncore<NCORE (widened CM+ZC) but cctx.hzz_zc_op is nothing -- built unconditionally by build_cm_meanzc_bin_ctx whenever there IS a mean/pair block; this should be provably unreachable in production.")
            has_france = layout.france_ratio_reduced_j > 0
            bi_slot = has_france ? dest_slot(cctx.econ_ctx, cctx.econ_ctx.bi) : 0
            gpσ = has_france ? θ_full[3 + cctx.econ_ctx.D]^θ_full[2] : 0.0
            denom_cf = has_france ? gpσ * cctx.econ_ctx.γ.LPrime[cctx.econ_ctx.bi] : 0.0
            if cctx.profiled_full_wctx === nothing || cctx.profiled_full_wctx_for !== cf
                cctx.profiled_full_wctx = build_winner_pair_ctx(cf; bi_slot = bi_slot, gpσ = gpσ, denom_cf = denom_cf)
                cctx.profiled_full_wctx_for = cf
            end
            wctx_full = cctx.profiled_full_wctx
            ws_full = cctx.profiled_full_ws_zc
            if ws_full === nothing || ws_full.W != wctx_full.W || ws_full.max_nx < n_restr || ws_full.Ddest != wctx_full.Ddest
                ws_full = WinnerZCCrossScratch(wctx_full.W, n_restr, wctx_full.Ddest)
                cctx.profiled_full_ws_zc = ws_full
            end
            record_winner_cross_hessian_call!()
            winner_pair_cross_hessian_zc_prep!(ws_full, wctx_full, w)
            op = cctx.hzz_zc_op
            refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, cctx.nu_ref[])
            cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op, size(w, 1))
            # ROOT-CAUSE FIX precedent (diagnose-optimize/HZZ-BLAS-and-HCZ-prep-2026-07-29, Part B,
            # see the non-profiled ncore<NCORE branch's own identical comment below): ZcS is read
            # UNCONDITIONALLY by bin_zc_cross_hessian_fill! (H_ZC, hessian_cm_structured!'s own
            # profiled-branch widened-CM-grid gather, added alongside this H_EM/H_MM code) regardless
            # of zc_gram_backend -- always fill_S=true here, not gated.
            refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, w; fill_S = true)
            nx = n_restriction(op)
            Z = @view cctx.hzz_centered.Zc[:, 1:nx]
            if size(cctx.profiled_hraw_em_full) != (wctx_full.ncolI + 1, n_restr)
                cctx.profiled_hraw_em_full = Matrix{Float64}(undef, wctx_full.ncolI + 1, n_restr)
            end
            HEM_full = cctx.profiled_hraw_em_full
            # Phase 7 (dispatch-counters, 2026-08-02): CONFIRMED LIVE -- this profiled/reduced-layout
            # H_EM gather calls the base `winner_pair_cross_hessian_zc_block!` kernel UNCONDITIONALLY,
            # never consulting `cctx.zc_ez_backend` at all (unlike the non-profiled ncore<NCORE branch
            # above, which has a real :drawmajor/:drawmajor_v2/:threaded/serial if-chain). So on the
            # profiled/reduced path, H_EM/H_EZ never dispatches through :drawmajor_v2 regardless of
            # `zc_ez_backend`'s value -- recorded as a fallback so this is VISIBLE in the counters
            # rather than a silent gap; wiring the profiled gather to the drawmajor_v2 backend is a
            # separate, larger follow-up (not attempted here, out of this task's surgical scope).
            record_drawmajor_v2_fallback!()
            winner_pair_cross_hessian_zc_block!(HEM_full, wctx_full, ws_full, w, Z, M; use_profiled_correction = true)
            n_bilateral = length(layout.retained_full_factual_j)
            @views HEM[1, :] .= HEM_full[1, :]
            @inbounds for kk in 1:n_bilateral
                j_full = layout.retained_full_factual_j[kk]
                @views HEM[1 + kk, :] .= HEM_full[1 + j_full, :]
            end
            has_france && (@views HEM[ncore, :] .= HEM_full[1 + wctx_full.ncolI, :])
            # BUG FIX (found live 2026-08-02 via ForwardDiff Hessian cross-check on the reduced+
            # widened operator FG -- profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl's own
            # gate): SAME class of bug as the H_EC fix's own precedent comment above
            # ("c13_debug_archC.jl: uniform 2x discrepancy... the symmetrize-by-averaging step below
            # reads BOTH Hfull[i,j] and Hfull[j,i]") -- the packing step (line ~1326,
            # `0.5*(Hfull[i,j]+Hfull[j,i])`) applies to EVERY (i,j) pair with i,j<=NCORE, which
            # includes this H_EM cross-block (both economic and mean/pair-Z rows are <=NCORE). Only
            # `HEM` (upper: economic-row x mean/pair-col) was filled above; the transposed mirror
            # (mean/pair-row x economic-col) was left at its initial zero, so the average silently
            # halved every H_EM entry -- confirmed exactly (ratio 2.0 across the whole block, ForwardDiff
            # vs production) before this fix.
            @views HEE[ncore+1:NCORE, 1:ncore] .= transpose(HEM)
            # ---- H_MM -- UNCHANGED, copied verbatim from the non-profiled ncore<NCORE branch below ----
            if cctx.zc_gram_backend === :reference
                zc_restriction_gram!(HMM, cctx.hzz_centered, op, M)
            else
                cctx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(cctx.raw_zc_ws, op, size(w, 1))
                refresh_zc_raw_target_vector!(cctx.raw_zc_ws, cctx.hzz_zc_ws, op)
                zc_gram_dispatch!(HMM, cctx.zc_gram_backend, nothing, op, cctx.raw_zc_ws, w, M; workers = cctx.zc_gram_workers)
            end
        end
        return HEE
    end
    if cf isa CompressedFactual && cctx.core_hessian_backend !== :dense_reference
        if cctx.core_ws === nothing || cctx.core_ws_for !== cf
            cctx.core_ws = build_core_exact_hessian_workspace(cf)
            cctx.core_ws_for = cf
            record_compressed_core_rebuild!()
        end
        HEE_core = @view HEE[1:ncore, 1:ncore]
        @cmhess_prof "H_EE_core" fill_core_hessian_upper!(HEE_core, w, obj, cctx.core_ws;
            backend = cctx.core_hessian_backend, workers = cctx.core_hessian_workers, storage = cctx.core_hessian_storage)
        if ncore < NCORE
            # CM+mean/ZC only: mean/pair columns are folded into this SAME widened "economic"
            # block by cm_meanzc_moments.jl's own column layout (not a separate CM-grid-style
            # restriction block) -- task §4.3's "retain the existing method for the non-core
            # blocks" applies to exactly this cross (core x mean/pair) and (mean/pair x mean/pair)
            # corner. HMM (mean/pair x mean/pair, H_RR) stays dense BLAS unconditionally -- out of
            # scope (task Section 4). HEM (core x mean/pair, H_EM=H_ER) is winner-aware phase
            # Section 4's own target: dispatches to `winner_pair_cross_hessian_zc_block!` when
            # `cctx.zc_cross_hessian_backend === :winner_bin` and the SAME core_ws/cf guard
            # `_fill_cm_HEE!`'s own H_EE fill just satisfied above holds -- ONLY `Ews[:,
            # ncore+1:NCORE]` (EM, needed by HMM regardless of backend) is ever computed from `E`
            # in that path; `E[:, 1:ncore]` (the economic columns) is never read.
            HEM = @view HEE[1:ncore, ncore+1:NCORE]
            HMM = @view HEE[ncore+1:NCORE, ncore+1:NCORE]
            zc_direct_ready = cctx.hzz_zc_op !== nothing
            winner_bin_ok = _cm_zc_cross_hessian_wants_winner_bin(cctx, cf)
            # No-moments/no-composite-G task (2026-07-28): `Z` (the "already-centered restriction
            # columns") is now sourced from `cctx.hzz_centered.Zc` instead of `@view E[:,
            # ncore+1:NCORE]` -- `zc_restriction_operator.jl`'s own docstring confirms this is
            # BIT-IDENTICAL to `wrap_moments_with_cm_meanzc`'s dense `obj.H` Z columns ("the SAME
            # quantity... never read from obj.H"), and it is the SAME scratch H_ZZ/`zc_restriction_
            # gram!` below already uses (gated at D=4/D=20) -- refreshed once here, shared by both
            # HEM and HMM, so `obj.H`'s restriction columns are no longer read anywhere in this
            # callback. Falls back to the dense `E`-based path only when the direct ZC state isn't
            # available at all (plain CM/common-Frechet, which never widen `ncore < NCORE`).
            if winner_bin_ok && zc_direct_ready
              @cmhess_prof "H_ER_prep" begin
                op = cctx.hzz_zc_op
                refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, cctx.nu_ref[])
                cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op, size(w, 1))
                # ROOT-CAUSE FIX (diagnose-optimize/HZZ-BLAS-and-HCZ-prep-2026-07-29, Part B):
                # the ADDENDUM (2026-07-28) comment this replaces claimed "ZcS is only needed by
                # the :reference H_ZZ backend" -- FALSE. bin_zc_cross_hessian_fill! (H_CZ, below,
                # cm_meanzc-only) ALSO reads cctx.hzz_centered.ZcS UNCONDITIONALLY, regardless of
                # zc_gram_backend. Gating fill_S on backend==:reference left ZcS at its initial
                # all-zeros (or stale, from whichever point last had backend==:reference) value for
                # every non-:reference backend, silently corrupting H_CZ into a near-zero/stale
                # block -- confirmed live: this is the actual, full explanation for the
                # reproducible :blas_gemm/:centered_syrk/:blas_syrk/:threaded_packed inner-solve
                # failures (docs/HZZ_BACKEND_BAKEOFF_VERDICT_2026-07-29.md's "genuine_solver_
                # sensitivity" verdict was WRONG -- superseded by
                # docs/HZZ_HCZ_SHARED_ZCS_BUG_ROOT_CAUSE_2026-07-29.md). ZcS fill is a cheap
                # O(W*nx) elementwise multiply -- not worth a backend-conditional skip regardless.
                refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, w; fill_S = true)
                record_winner_cross_hessian_call!()
                wctx = serial_ctx(cctx.core_ws)
                n_restr = NCORE - ncore
                cctx.zc_cross_scratch = _ensure_zc_cross_scratch!(cctx, wctx.W, n_restr)
                winner_pair_cross_hessian_zc_prep!(cctx.zc_cross_scratch, wctx, w)
                nx = n_restriction(op)
                Z = @view cctx.hzz_centered.Zc[:, 1:nx]   # already-centered restriction columns, unweighted
              end
              @cmhess_prof "H_ER" if cctx.zc_ez_backend === :drawmajor
                    record_drawmajor_v2_fallback!()
                    nbilateral = wctx.has_cf ? wctx.ncolI - 1 : wctx.ncolI
                    cctx.zc_drawmajor = ensure_winner_zc_drawmajor_scratch!(cctx.zc_drawmajor, wctx.W, wctx.Ddest, nbilateral, nx, cctx.cross_hessian_workers)
                    winner_pair_cross_hessian_zc_block_drawmajor!(HEM, wctx, cctx.zc_cross_scratch, cctx.zc_drawmajor, w, Z, M; workers = cctx.cross_hessian_workers)
                elseif cctx.zc_ez_backend === :drawmajor_v2
                    # ZC Hessian backend production integration (2026-08-01): this branch was
                    # missing from BOTH the original genuine-cold-zc-hessian-k3 session's port AND
                    # this integration's own first pass -- cm_meanzc's H_ER dispatch only ever
                    # recognized :drawmajor (v1), never :drawmajor_v2, silently falling through to
                    # the :cross_hessian_threaded branch (byte-identical to :winner_bin) for any
                    # cctx with zc_ez_backend=:drawmajor_v2 set. origin-ZC's own octx dispatch
                    # (below) already had this branch; cm_meanzc's cctx dispatch did not. Found live
                    # via Gate 3's real-KNITRO-driver run (the isolated-kernel benchmark in Section 6
                    # called winner_pair_cross_hessian_zc_block_drawmajor_v2! directly, bypassing
                    # this dispatch entirely, so it never caught the gap). Fixed before merge.
                    record_drawmajor_v2_dispatch!()
                    nbilateral = wctx.has_cf ? wctx.ncolI - 1 : wctx.ncolI
                    cctx.zc_drawmajor = ensure_winner_zc_drawmajor_v2_scratch!(cctx.zc_drawmajor, wctx.W, wctx.Ddest, nbilateral, nx, cctx.cross_hessian_workers)
                    winner_pair_cross_hessian_zc_block_drawmajor_v2!(HEM, wctx, cctx.zc_cross_scratch, cctx.zc_drawmajor, w, Z, M; workers = cctx.cross_hessian_workers)
                elseif cctx.cross_hessian_threaded
                    record_drawmajor_v2_fallback!()
                    winner_pair_cross_hessian_zc_block_threaded!(HEM, wctx, cctx.zc_cross_scratch, w, Z, M; workers = cctx.cross_hessian_workers)
                else
                    record_drawmajor_v2_fallback!()
                    winner_pair_cross_hessian_zc_block!(HEM, wctx, cctx.zc_cross_scratch, w, Z, M)
                end
                # ADDENDUM (2026-07-28): H_ZZ backend dispatch -- :reference (existing, centered-Zc
                # BLAS gemm) | :blas_syrk | :blas_gemm | :threaded_packed (all three built directly
                # from the immutable raw Phi, zc_gram_blas_candidates.jl).
              @cmhess_prof "H_ZZ" if cctx.zc_gram_backend === :reference
                    record_blas_syrk_fallback!()
                    zc_restriction_gram!(HMM, cctx.hzz_centered, op, M)
                else
                    cctx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(cctx.raw_zc_ws, op, wctx.W)
                    refresh_zc_raw_target_vector!(cctx.raw_zc_ws, cctx.hzz_zc_ws, op)
                    zc_gram_dispatch!(HMM, cctx.zc_gram_backend, nothing, op, cctx.raw_zc_ws, w, M; workers = cctx.zc_gram_workers)
                end
            else
                # No-moments/no-composite-G task (2026-07-28): `E` constructed lazily, only here,
                # in the (now unreachable in production -- see `check_ties=false` above) dense
                # fallback -- never at this function's top level.
                record_dense_cross_hessian_call!()
                H === nothing && error("_fill_cm_HEE!: reached the dense H_EM/H_MM fallback for an operator-mode bundle with no H field -- this should be provably unreachable in production (winner_bin_ok && zc_direct_ready should always hold); indicates a real configuration bug, not expected behavior.")
                E = @view H[:, 2:1+NCORE]
                Ews = cctx.Ews
                @views Ews[:, ncore+1:NCORE] .= E[:, ncore+1:NCORE] .* sqrt.(w)
                EM = @view Ews[:, ncore+1:NCORE]
                @views Ews[:, 1:ncore] .= E[:, 1:ncore] .* sqrt.(w)
                EC = @view Ews[:, 1:ncore]
                BLAS.gemm!('T', 'N', 1 / M, EC, EM, 0.0, HEM)
                BLAS.gemm!('T', 'N', 1 / M, EM, EM, 0.0, HMM)
            end
            # production Hessian allocation audit (2026-08-02): was `@views HEE[ncore+1:NCORE,
            # 1:ncore] .= transpose(HEM)`. HEM is itself a view of HEE (`@view HEE[1:ncore,
            # ncore+1:NCORE]`, set above) -- broadcasting into a view of the SAME parent array as
            # the source, even at genuinely disjoint index ranges, hits Julia's broadcast
            # aliasing-defensive-copy path (`Base.mightalias`/`Broadcast.unalias`), which
            # materializes a full temporary the size of the destination before copying it in.
            # Confirmed live: isolated repro measured 1,683,472 bytes for a 700x300 case
            # (expected temp size 1,680,000 bytes, i.e. essentially the whole allocation), zero
            # bytes for the identical broadcast between two INDEPENDENT arrays, and zero bytes for
            # this same explicit loop against the same-parent-array views. This one line accounted
            # for 1,925,280 of cm_meanzc's 1,993,488 measured bytes/callback (96.6%,
            # Profile.Allocs, W=20,000) -- the only family/size combination where `ncore < NCORE`
            # (this branch) is ever reached; flexible_cm/common_frechet always have ncore==NCORE
            # and never execute this code at all. Mathematically identical assignment (a plain
            # elementwise copy of HEM's transpose into HEE's lower-left corner); see
            # PRODUCTION_HESSIAN_AUDIT_MASTER_2026-08-02.md's accepted-optimization list for the
            # correctness gate (D4 dense-truth + D20 fixed-state packed-Hessian bit-identity).
            ncore_wid = NCORE - ncore
            @inbounds for i in 1:ncore, j in 1:ncore_wid
                HEE[ncore + j, i] = HEM[i, j]
            end
        end
    else
        # No-moments/no-composite-G task (2026-07-28): `E` constructed lazily, only here.
        H === nothing && error("_fill_cm_HEE!: reached the dense H_EE fallback for an operator-mode bundle with no H field -- this should be provably unreachable in production (cf should always be a valid CompressedFactual and core_hessian_backend should never be :dense_reference on this construction path); indicates a real configuration bug, not expected behavior.")
        E = @view H[:, 2:1+NCORE]
        Ews = cctx.Ews
        @views Ews[:, 1:NCORE] .= E[:, 1:NCORE] .* sqrt.(w)
        BLAS.gemm!('T', 'N', 1 / M, @view(Ews[:, 1:NCORE]), @view(Ews[:, 1:NCORE]), 0.0, HEE)
        reason = cctx.core_hessian_backend === :dense_reference ? :debug_reference_requested :
                  (cf isa Symbol ? cf : :other)
        record_core_hessian_call!(:dense_inline_fallback; fallback_reason = reason)
    end
    return HEE
end

"""
Winner-aware H_ER phase (2026-07-27), Section 2: decide whether THIS Hessian callback may use the
`:winner_bin` cross-Hessian backend for `H_EC`. Requires (a) the backend is actually requested,
(b) EITHER `cctx.ncore_core == cctx.NCORE` (no CM+mean/pair-ZC widening -- plain flexible CM/
common-Frechet) OR `cctx.hzz_zc_op !== nothing` (CM+ZC's widened case, task's E/C/Z block-partition
release, 2026-07-27: `hessian_cm_structured!`/`_v2!` now ALSO fill the widened rows'
`H_CZ` cross block via `bin_zc_cross_hessian_block!` whenever this function and
`_cm_cross_hessian_wants_direct_hcz` both hold, so the full `NCORE`-row block is always fully
covered before this relaxation was made safe -- see that function's own docstring for the "why this
was previously a real correctness gap" history), and (c) `cctx.core_ws`/`core_ws_for` were ACTUALLY
refreshed for the CURRENT `cf` by `_fill_cm_HEE!` just now (i.e. H_EE itself used the winner-pair
backend this call, not a dense fallback) -- reusing `_fill_cm_HEE!`'s own decision instead of
re-deriving a second one prevents this backend from ever running against a stale/mismatched
`core_ws`. Not silent: callers that want `:winner_bin` but land here `false` fall back to the dense
CScum path and record `record_dense_cross_hessian_call!` (never a bare unrecorded fallback).
"""
function _cm_cross_hessian_wants_winner_bin(cctx::CMBinHessCtx, cf)
    return cctx.cm_cross_hessian_backend === :winner_bin &&
           (cctx.ncore_core == cctx.NCORE || cctx.hzz_zc_op !== nothing) &&
           cf isa CompressedFactual &&
           cctx.core_ws !== nothing &&
           cctx.core_ws_for === cf
end

"""
CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): decide whether THIS Hessian callback
must ALSO fill the widened rows (`ncore_core+1:NCORE`, the mean/pair-ZC restriction `Z` block) of
the CM-grid cross block via the NEW direct `bin_zc_cross_hessian_block!` primitive
(`winner_pair_cross_hessian.jl`), i.e. whether `_cm_cross_hessian_wants_winner_bin` is about to fill
ONLY the true-economic rows (`1:ncore_core`, via the existing `wctx`, which is always exactly
`ncore_core`-wide -- unaffected by CM+ZC's own widening) and therefore needs this companion call to
cover the rest. `true` exactly when there IS a widened Z block (`ncore_core < NCORE`) AND the raw ZC
feature state is available (`cctx.hzz_zc_op !== nothing`, CM+ZC only) AND
`_cm_cross_hessian_wants_winner_bin` itself holds for this callback.
"""
function _cm_cross_hessian_wants_direct_hcz(cctx::CMBinHessCtx, cf)
    return cctx.ncore_core < cctx.NCORE && cctx.hzz_zc_op !== nothing && _cm_cross_hessian_wants_winner_bin(cctx, cf)
end

"""
Winner-aware H_ER phase (2026-07-27), Section 4: decide whether THIS Hessian callback may use the
`:winner_bin` cross-Hessian backend for CM+ZC's H_EM (core x mean/pair cross) block, inside
`_fill_cm_HEE!`'s own `ncore < NCORE` branch. Requires (a) the backend is actually requested, (b)
`cctx.ncore_core < cctx.NCORE` (there IS a widened mean/pair block to fill -- plain flexible
CM/common-Frechet never reach this branch at all, `ncore_core == NCORE` there), and (c)
`cctx.core_ws`/`core_ws_for` were ACTUALLY refreshed for the CURRENT `cf` by `_fill_cm_HEE!`'s own
H_EE fill just above (H_EE itself used the winner-pair backend this call, not a dense fallback) --
same discipline `_cm_cross_hessian_wants_winner_bin` already established for the CM-grid H_EC
block, reused here rather than re-derived. Not silent: callers that want `:winner_bin` but land
here `false` fall back to the dense HEM path and record `record_dense_cross_hessian_call!`.
"""
function _cm_zc_cross_hessian_wants_winner_bin(cctx::CMBinHessCtx, cf)
    return cctx.zc_cross_hessian_backend === :winner_bin &&
           cctx.ncore_core < cctx.NCORE &&
           cf isa CompressedFactual &&
           cctx.core_ws !== nothing &&
           cctx.core_ws_for === cf
end

"""
CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): decide whether THIS Hessian callback
may use the NEW direct `zc_restriction_gram!` backend for H_ZZ (`_fill_cm_HEE!`'s widened HMM
block). SAME guard `_cm_zc_cross_hessian_wants_winner_bin` already establishes for HEM (reused, not
re-derived -- H_ZZ is only ever attempted alongside a winner-aware HEM, never on its own), PLUS
`cctx.hzz_zc_op !== nothing` (built only for CM+ZC, `build_cm_meanzc_bin_ctx` -- plain CM/common-
Frechet never widen, so this is always `nothing` there and this function always returns `false`).
Not silent: callers that want the direct backend but land here `false` fall back to the dense
`EM'*EM` path and record `record_dense_cross_hessian_call!`.
"""
function _cm_zc_wants_direct_hzz(cctx::CMBinHessCtx, cf)
    return cctx.hzz_zc_op !== nothing && _cm_zc_cross_hessian_wants_winner_bin(cctx, cf)
end

"Ensure `cctx.zc_cross_scratch` is sized for the current (W,n_restr); rebuild only on a genuine size change (campaign-lifetime constant in practice), never per-Hessian-callback."
function _ensure_zc_cross_scratch!(cctx::CMBinHessCtx, W::Int, n_restr::Int)
    cs = cctx.zc_cross_scratch
    if cs === nothing || cs.W != W || cs.max_nx < n_restr
        cctx.zc_cross_scratch = WinnerZCCrossScratch(W, n_restr)
    end
    return cctx.zc_cross_scratch
end

"Ensure `cctx.cross_scratch` is sized for the current (ncolI,D,L); rebuild only on a genuine size change (campaign-lifetime constant in practice), never per-Hessian-callback."
function _ensure_cm_cross_scratch!(cctx::CMBinHessCtx, ncolI::Int, D::Int, L::Int)
    cs = cctx.cross_scratch
    if cs === nothing || cs.ncolI != ncolI || cs.D != D || cs.L != L
        cctx.cross_scratch = WinnerBinCrossScratch(ncolI, D, L)
    end
    return cctx.cross_scratch
end

"""
    _dense_H_or_nothing(obj) -> Union{Nothing,Matrix{Float64}}

Legacy-H removal (2026-07-28): dispatched accessor letting `hessian_cm_structured!`/`_v2!`
(this file / `cm_hessian_threaded.jl`) work for BOTH `PsiObjectiveBundleImplicit`
(`:dense_reference`, returns `obj.H`) and `OperatorCMBundle` (no `H` field at all, returns
`nothing`) without an unconditional `@unpack H,...=obj` that would require every bundle type to
carry that field. `H` is only ever actually read inside the explicit dense-fallback branches of
`_fill_cm_HEE!`/`build_bin_tables!`, both of which fail fast (not silently) if handed `nothing`.
"""
_dense_H_or_nothing(obj::CS.PsiObjectiveBundleImplicit) = obj.H
# The OperatorPsiBundle method is defined in operator_psi_bundle.jl (included after this file --
# adding it here would be a forward reference to a not-yet-defined type).

"""
    _dense_H_copy_or_nothing(obj) -> Union{Nothing,Matrix{Float64}}

Same dispatch pattern as `_dense_H_or_nothing`, for `archA_partitioned_hess_cb_builder`'s
(origin-ZC) `H_copy` scratch -- only read inside that callback's own dense-fallback branches.
"""
_dense_H_copy_or_nothing(obj::CS.PsiObjectiveBundleImplicit) = obj.H_copy
# The OperatorPsiBundle method is defined in operator_psi_bundle.jl.

"""
    pack_upper_cm_hessian!(h, Hfull, NCORE, n)

Hessian upper-only cleanup (2026-07-28): the shared final packing step for `hessian_cm_structured!`
(this file) and `hessian_cm_structured_v2!` (`cm_hessian_threaded.jl`) -- factored out of both
(previously two independently-maintained, byte-identical copies of this loop, flagged in
`PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md` as a manual-sync risk the threaded file's own
header already warned about) so there is exactly one place this logic can drift.

The H_EC region (`i<=NCORE<j`) is filled by a MECHANICAL mirror (never independently re-derived --
see the H_EC block-fill loop above/in `cm_hessian_threaded.jl`), so `Hfull[i,j]` alone is exact;
averaging it against `Hfull[j,i]` is a provable no-op there, at real per-callback cost. H_EE
(`i,j<=NCORE`) and H_CC (`i,j>NCORE`) both keep full averaging: H_EE can be filled via a dense BLAS
`gemm!` fallback whose bit-exact cross-diagonal symmetry is a BLAS-implementation property, not a
language guarantee; H_CC's `(l,lp)` grid is genuinely independently accumulated from a fresh
prefix-sum evaluation per pair, not copied. `Hfull`'s own mirror writes are intentionally left in
place elsewhere (not removed) -- `cctx.Hfull` is read directly, both triangles, by several
diagnostic/test scripts (`diag_frechet_hardpoint_2026-07-27.jl`,
`test_frechet_winner_bin_her_wiring_d4.jl`), so breaking that invariant is out of scope here.
"""
function pack_upper_cm_hessian!(h::AbstractVector, Hfull::AbstractMatrix, NCORE::Int, n::Int)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            h[k] = i <= NCORE < j ? Hfull[i, j] : 0.5 * (Hfull[i, j] + Hfull[j, i])
            k += 1
        end
    end
    return h
end

"""
    fill_cm_HCC!(Hfull, cctx::CMBinHessCtx, M)

H_CC (CM-CM restriction self-block): raw per-threshold-block-pair computation, then optional R
congruence. Shared between flexible CM and common Fréchet (harmonization task, 2026-07-28) --
previously two verbatim-identical copies, one per family. `M` is `obj.M` (not a `CMBinHessCtx`
field), passed through exactly as both pre-existing call sites already had it in scope.
"""
function fill_cm_HCC!(Hfull::AbstractMatrix, cctx::CMBinHessCtx, M)
    CT = cctx.CT
    Hraw_CC = cctx.Hraw_CC
    L = cctx.L; nO = cctx.nO; NCORE = cctx.NCORE
    origins = cctx.origins; refIndex1 = cctx.refIndex1
    @inbounds for l in 1:L
        for lp in 1:L
            for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
                Hraw_CC[oi, pi] = (CT[o, p, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, p, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
            end
            rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
            block = if cctx.R === nothing
                Hraw_CC
            else
                mul!(cctx.RtHraw_CC, cctx.R', Hraw_CC)
                mul!(cctx.block_cc, cctx.RtHraw_CC, cctx.R)
            end
            @views Hfull[rows, cols] .= block
        end
    end
    return Hfull
end

"""
    hessian_cm_structured!(h, obj, cctx::CMBinHessCtx)

Architecture C Hessian callback. Requires `obj.arg0` to already reflect the
CURRENT (zeta,lambda) (same precondition as `chunked_hessian.jl`'s
`hessian_chunked!` -- caller must run `_prep_for_hessian!(obj,x)` first, see
below). Writes the packed upper-triangular Hessian into `h`, matching
`cc_algo/PsiObjectiveBundle.jl::hessian!`'s own packing exactly.
"""
function hessian_cm_structured!(h, obj, cctx::CMBinHessCtx, extension::Any = nothing)
    # Harmonization task (2026-07-28): `extension` is `nothing` for flexible CM/CM+ZC (every
    # existing call site, unchanged) or a `CMFrechetExtension` (cm_frechet_hessian.jl, included
    # after this file -- `Any`-typed here purely to avoid a forward reference, same reason
    # `cmlookup_st`/`frechet_ext_cache` are `Any` on `CMBinHessCtx` itself) for common Fréchet,
    # which now calls this SAME function instead of its own separate
    # `hessian_cm_frechet_structured!` copy. Julia specializes/compiles this method separately per
    # concrete runtime type of `extension` (`Nothing` vs `CMFrechetExtension`), so this costs
    # flexible CM's hot path nothing -- it is JIT-compiled exactly as if `extension` were typed
    # `Nothing` for every one of its own calls.
    @unpack M, arg0, arg2, ddPsi! = obj
    H = _dense_H_or_nothing(obj)
    ddPsi!(arg2, arg0)
    w = arg2
    NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
    refIndex1 = cctx.refIndex1; origins = cctx.origins

    # No-moments/no-composite-G task (2026-07-28): `E` is no longer constructed eagerly here --
    # `_fill_cm_HEE!`/`build_bin_tables!` now construct it lazily, only inside their own
    # (unreachable-in-production, `:dense_reference`-only) fallback branches, so `H` need not have
    # `NCORE` economic columns at all on the production `:winner_bin` path.
    Hfull = cctx.Hfull
    fill!(Hfull, 0.0)
    HEE = @view Hfull[1:NCORE, 1:NCORE]
    cf = cctx.core_cf_ref[]
    _fill_cm_HEE!(HEE, w, obj, cctx, H, M)   # may rebuild cctx.core_ws/core_ws_for for this cf

    if cctx.profiled_layout !== nothing
        # Profiled all-families completion task (2026-08-01): H_EC "gather retained rows at
        # assembly" step -- see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md for the full
        # design rationale. Rebuilds (cf-identity-cached) the FULL, unreduced
        # `WinnerPairHessCtx`/`WinnerBinCrossScratch` from the SAME `cf` _fill_cm_HEE! already used
        # for the reduced H_EE, calls the already-validated, UNTOUCHED cross-block primitives
        # (`winner_pair_cross_hessian_fill!`/`_cm_block!`) with `use_profiled_correction=true`
        # exactly as the unrestricted family does, then gathers only `layout`-retained rows into
        # the (smaller) `Hfull` -- zero lines inside either of those two functions change.
        # Restricted-inner-endtoend task (2026-08-01): the H_EF gather branch below (§8) makes
        # `extension isa CMFrechetExtension` a supported case here too -- previously this line
        # unconditionally errored for any non-nothing `extension` (H_EC only). Any OTHER non-nothing,
        # non-CMFrechetExtension `extension` is still not a recognized case for this branch.
        # ZC lane task (2026-08-02): this branch's own gather loop below only fills
        # `Hraw_EC[1:cctx.ncore_core, :]` (the TRUE-economic rows) before copying `Hraw_EC[1:NCORE,
        # :]` wholesale into `Hfull` -- correct only when `ncore_core==NCORE` (plain flexible
        # CM/common-Frechet, this branch's ONLY callers until this task). CM+ZC's own widened
        # `ncore_core<NCORE` case would silently copy STALE `Hraw_EC[ncore_core+1:NCORE,:]` (the
        # mean/pair-Z rows, never written by this loop) into the CM-grid cross block -- a genuine
        # gap this task did not close (H_ZC/H_CC gather for CM+ZC's own CM-grid block under the
        # reduced economic layout is separately-scoped follow-up work, not attempted here; H_EE/
        # H_EM/H_MM above ARE closed and gated). Loud, not silent: refuse rather than corrupt.
        # ZC lane task (2026-08-02): CM+ZC widened-core H_ZC (mean/pair-Z x CM-grid cross). When
        # ncore<NCORE, `Hraw_EC[ncore+1:NCORE,:]` (the mean/pair-Z rows) is filled from the SAME
        # `bin_zc_cross_hessian_fill!`/`_block!` primitive the non-profiled `use_direct_hcz` branch
        # below already uses UNCHANGED -- it depends only on `cctx.Bidx`/`cctx.hzz_centered.ZcS`
        # (theta/bin-table-based), never on the economic block width or `wctx`, so it is genuinely
        # independent of the profiled/reduced layout, same rationale as H_MM above. NOTE: this
        # dispatches through the plain serial kernel, not `hcz_prep_dispatch!`'s optimized
        # `:draw_chunk_reordered` candidate -- that dispatch is only wired into the THREADED twin
        # (cm_hessian_threaded.jl), which the profiled/reduced path does not use for ANY family
        # (flexible CM's own profiled H_EC gather predates this task and has the identical
        # limitation -- `threaded_bins=false` is required for every profiled/reduced cctx in this
        # codebase, not a new restriction this task introduces).
        local bin_zc_ws_ec
        if cctx.ncore_core < NCORE
            nz_ec = n_restriction(cctx.hzz_zc_op)
            bin_zc_ws_ec = ensure_bin_zc_cross_scratch!(cctx.bin_zc_cross, D, L, nz_ec)
            cctx.bin_zc_cross = bin_zc_ws_ec
            # Phase 7 (dispatch-counters, 2026-08-02): this profiled/reduced H_CZ gather calls the
            # plain serial kernel directly, bypassing `hcz_prep_dispatch!`/`cctx.hcz_prep_backend`
            # entirely (see the file-header comment a few lines above this branch) -- :draw_chunk_
            # reordered never dispatches here regardless of that Ref's value. Recorded so it is
            # visible, not fixed here (out of this task's surgical scope).
            record_draw_chunk_reordered_fallback!()
            bin_zc_cross_hessian_fill!(bin_zc_ws_ec, cctx.Bidx, cctx.hzz_centered.ZcS)
        end
        # BUGFIX (found live, 2026-08-01, via a direct user challenge to re-verify H_CC rather than
        # trust "unchanged code must be correct"): this branch skipped build_bin_tables!/
        # prefix_sum_tables! entirely, so cctx.CT (the theta/weight-dependent bin table
        # fill_cm_HCC! below reads) was STALE -- whatever it held from a previous, unrelated call
        # (or all-zeros if never built). H_CC is CM-grid-only (does not depend on the economic
        # block at all, `fill_S=false` since this path never reads Stab/CScum), so this fix is
        # identical in spirit to the non-profiled branch's own `build_bin_tables!(cctx, H, w;
        # fill_S=!use_winner_bin)` call -- just always fill_S=false here, since the profiled path
        # never uses the dense S-table for ANY block. Confirmed the actual bug (not a conditioning
        # issue) via a direct H_CC-only numeric comparison against the full model: max|Δ| was
        # ~0.5 (real, reproducible, non-noise) before this fix.
        build_bin_tables!(cctx, H, w; fill_S = false)
        prefix_sum_tables!(cctx; fill_S = false)
        layout = cctx.profiled_layout
        has_france = layout.france_ratio_reduced_j > 0
        bi_slot = has_france ? dest_slot(cctx.econ_ctx, cctx.econ_ctx.bi) : 0
        # gpσ = gp^σ, needed for build_winner_pair_ctx's own Lam_homog[cf.cf_col] -- SAME formula as
        # build_reduced_homogeneous_winner_pair_ctx's own gpσ computation (θ_full[2]=σ, θ_full[3+D]=gp).
        # denom_cf = gpσ*wPrime_bi*LPrime[bi] (wPrime_bi≡1) -- SAME formula, needed for
        # WinnerPairHessCtx's own denom_cf_scaled field (H_EC France-row second bugfix).
        gpσ = has_france ? cctx.profiled_theta_ref[][3 + D]^cctx.profiled_theta_ref[][2] : 0.0
        denom_cf = has_france ? gpσ * cctx.econ_ctx.γ.LPrime[cctx.econ_ctx.bi] : 0.0
        if cctx.profiled_full_wctx === nothing || cctx.profiled_full_wctx_for !== cf
            cctx.profiled_full_wctx = build_winner_pair_ctx(cf; bi_slot = bi_slot, gpσ = gpσ, denom_cf = denom_cf)
            cctx.profiled_full_wctx_for = cf
        end
        wctx_full = cctx.profiled_full_wctx
        ws_full = cctx.profiled_full_ws
        if ws_full === nothing || ws_full.ncolI != wctx_full.ncolI || ws_full.D != D || ws_full.L != L || ws_full.Ddest != wctx_full.Ddest
            ws_full = WinnerBinCrossScratch(wctx_full.ncolI, D, L, wctx_full.Ddest)
            cctx.profiled_full_ws = ws_full
        end
        record_winner_cross_hessian_call!()
        winner_pair_cross_hessian_fill!(wctx_full, ws_full, obj, cctx.Bidx)
        if size(cctx.profiled_hraw_ec_full) != (wctx_full.ncolI + 1, nO)
            cctx.profiled_hraw_ec_full = Matrix{Float64}(undef, wctx_full.ncolI + 1, nO)
        end
        Hraw_EC_full = cctx.profiled_hraw_ec_full
        Hraw_EC = cctx.Hraw_EC   # reduced-sized (NCORE x nO) -- reused as the gather TARGET here
        n_bilateral = length(layout.retained_full_factual_j)
        @inbounds for l in 1:L
            winner_pair_cross_hessian_cm_block!(Hraw_EC_full, wctx_full, ws_full, l, origins, refIndex1, M; use_profiled_correction = true)
            @views Hraw_EC[1, :] .= Hraw_EC_full[1, :]
            @inbounds for k in 1:n_bilateral
                j_full = layout.retained_full_factual_j[k]
                @views Hraw_EC[1 + k, :] .= Hraw_EC_full[1 + j_full, :]
            end
            if has_france
                @views Hraw_EC[1 + layout.total_reduced_economic_moments, :] .= Hraw_EC_full[1 + wctx_full.ncolI, :]
            end
            if cctx.ncore_core < NCORE
                Hraw_EC_z = @view Hraw_EC[cctx.ncore_core+1:NCORE, :]
                bin_zc_cross_hessian_block!(Hraw_EC_z, bin_zc_ws_ec, l, origins, refIndex1, M)
            end
            cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
            block_ec = if cctx.R === nothing
                Hraw_EC
            else
                mul!(cctx.block_ec, Hraw_EC, cctx.R)
            end
            @views Hfull[1:NCORE, cols] .= block_ec
            @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
        end
        fill_cm_HCC!(Hfull, cctx, M)

        # Restricted-inner-endtoend task (2026-08-01), §8: common-Fréchet's H_E,level (H_EF) gather
        # branch. `extension !== nothing` (not `isa CMFrechetExtension`) for the same reason line
        # ~1430 below uses it -- this file must not reference the CMFrechetExtension type name
        # directly (flexible CM's own scripts never load cm_frechet_hessian.jl). H_CM,level/
        # H_level,level (the OTHER two blocks `_fill_frechet_level_blocks!` computes) are genuinely
        # UNCHANGED by the profiled economic layout -- they read only `cctx.CT`/`Wtab`/`T1`
        # (restriction-bin tables, already freshened by `build_bin_tables!`/`prefix_sum_tables!`
        # above), never `wctx`/the economic block width -- so `_fill_frechet_level_blocks_profiled!`
        # below only replaces the H_E,level piece, reusing the verbatim H_CM,level/H_level,level code.
        if extension !== nothing
            _fill_frechet_level_blocks_profiled!(Hfull, cctx, w, wctx_full, ws_full, layout, extension, M)
        end
    else
    use_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cf)
    use_direct_hcz = _cm_cross_hessian_wants_direct_hcz(cctx, cf)
    build_bin_tables!(cctx, H, w; fill_S = !use_winner_bin)
    prefix_sum_tables!(cctx; fill_S = !use_winner_bin)

    # harmonization task (2026-07-28): initialized to `nothing` (not left possibly-undefined) --
    # `wctx`/`cross_ws` are now also passed as plain function arguments to
    # `_fill_frechet_level_blocks!` below, which eagerly evaluates its arguments; an unassigned
    # local would throw UndefVarError at that call site the instant use_winner_bin is ever false,
    # even for calls that don't end up using them.
    local wctx, cross_ws, bin_zc_ws
    wctx = nothing; cross_ws = nothing; bin_zc_ws = nothing
    if use_winner_bin
        record_winner_cross_hessian_call!()
        wctx = serial_ctx(cctx.core_ws)
        cross_ws = _ensure_cm_cross_scratch!(cctx, wctx.ncolI, D, L)
        winner_pair_cross_hessian_fill!(wctx, cross_ws, obj, cctx.Bidx)
        if use_direct_hcz
            # CM+ZC E/C/Z block-partition + H_CZ release (2026-07-27): fills the widened rows
            # (`ncore_core+1:NCORE`) the plain wctx-based `winner_pair_cross_hessian_cm_block!` call
            # below never touches -- see `_cm_cross_hessian_wants_direct_hcz`'s own docstring for why
            # this pairing is what makes relaxing `_cm_cross_hessian_wants_winner_bin`'s guard safe.
            # `cctx.hzz_centered` was ALREADY refreshed for this callback's (θ, ν) by `_fill_cm_HEE!`
            # (H_ZZ, above) -- reused here as-is, not rebuilt a second time.
            record_winner_cross_hessian_call!()
            nz = n_restriction(cctx.hzz_zc_op)
            bin_zc_ws = ensure_bin_zc_cross_scratch!(cctx.bin_zc_cross, D, L, nz)
            cctx.bin_zc_cross = bin_zc_ws
            # Phase 7 (dispatch-counters, 2026-08-02): the SERIAL Architecture-C callback
            # (hessian_cm_structured!, this function) never calls `hcz_prep_dispatch!` at all --
            # that dispatcher is wired ONLY into the threaded twin (hessian_cm_structured_v2!,
            # cm_hessian_threaded.jl). So :draw_chunk_reordered never fires here regardless of
            # `cctx.hcz_prep_backend`'s value; recorded so it is visible, not fixed here (out of
            # this task's surgical scope -- see the file-header comment on the profiled H_CZ branch
            # above for the identical rationale).
            record_draw_chunk_reordered_fallback!()
            bin_zc_cross_hessian_fill!(bin_zc_ws, cctx.Bidx, cctx.hzz_centered.ZcS)
        end
    else
        record_dense_cross_hessian_call!()
    end

    # ---- H_EC raw, then optional R congruence (right-multiply by R per threshold block) ----
    # Allocation/Hessian port task §4.2: Hraw_EC/block_ec now live in cctx (persistent,
    # campaign-lifetime, sized once in build_cm_bin_ctx/build_cm_meanzc_bin_ctx) instead of
    # Hraw_EC being reallocated per hessian_cm_structured! call and block_ec = Hraw_EC * cctx.R
    # reallocating fresh on EVERY one of the L threshold-block iterations within that call.
    CS_ = cctx.CScum
    Hraw_EC = cctx.Hraw_EC   # reused per threshold block
    ncore_core = cctx.ncore_core
    @inbounds for l in 1:L
        if use_winner_bin
            Hraw_EC_core = use_direct_hcz ? (@view Hraw_EC[1:ncore_core, :]) : Hraw_EC
            winner_pair_cross_hessian_cm_block!(Hraw_EC_core, wctx, cross_ws, l, origins, refIndex1, M)
            if use_direct_hcz
                Hraw_EC_z = @view Hraw_EC[ncore_core+1:NCORE, :]
                bin_zc_cross_hessian_block!(Hraw_EC_z, bin_zc_ws, l, origins, refIndex1, M)
            end
        else
            for (oi, o) in enumerate(origins)
                for j in 1:NCORE
                    Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
                end
            end
        end
        cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
        block_ec = if cctx.R === nothing
            Hraw_EC
        else
            mul!(cctx.block_ec, Hraw_EC, cctx.R)
        end
        @views Hfull[1:NCORE, cols] .= block_ec
        # BUG FIX (found via c13_debug_archC.jl: uniform 2x discrepancy in H_EC vs
        # Architecture A): the symmetrize-by-averaging step below reads BOTH
        # Hfull[i,j] and Hfull[j,i] -- must mirror this block into the transposed
        # (CM-row, E-col) position too, or the average silently halves every H_EC
        # entry (H_CC was unaffected because that loop already visits both (l,l')
        # orderings explicitly; H_EE unaffected because BLAS gemm! fills both
        # triangles of a symmetric product).
        @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
    end

    # ---- H_CC raw, then optional R congruence (per threshold-block pair) ----
    # harmonization task (2026-07-28): extracted to the shared fill_cm_HCC! (also used by common
    # Fréchet) -- previously two verbatim-identical copies, one per family.
    fill_cm_HCC!(Hfull, cctx, M)

    # harmonization task (2026-07-28): common Fréchet's ONLY genuinely family-specific piece (the
    # "CM-F" common-level anchor blocks H_E,level / H_CM,level / H_level,level) -- was
    # hessian_cm_frechet_structured!'s own separate copy of everything above THIS point too;
    # everything above is now the one shared implementation both families call.
    # `extension !== nothing` (not `extension isa CMFrechetExtension`): this file must not reference
    # the CMFrechetExtension type name directly -- flexible CM's own scripts (which always pass
    # extension=nothing) do not load cm_frechet_hessian.jl at all, so a name reference here would
    # throw UndefVarError for them even though they never take this branch.
    if extension !== nothing
        _fill_frechet_level_blocks!(Hfull, cctx, w, H, M, use_winner_bin, wctx, cross_ws, extension)
    end
    end

    # symmetrize defensively (analytically symmetric; absorbs FP-order noise, same
    # defensive pattern as compressed_inner_alt_solvers.jl's denseaccum callback) -- see
    # pack_upper_cm_hessian!'s own docstring (above) for the per-block rationale.
    n = NCORE + ncm
    pack_upper_cm_hessian!(h, Hfull, NCORE, n)
    return h
end

"""
Same prep step chunked_hessian.jl uses (`_prep_for_hessian!`), duplicated here so this file has no
load-order dependency on chunked_hessian.jl.

No-moments/no-composite-G task (2026-07-28): this dense `H[:,2:1+outer_constr_index]` gemv
reconstruction of `r` (plus its downstream-unused `Psi!(arg1,arg0)` call) is now the EXPLICIT
`:dense_reference` fallback only -- every production Hessian callback (flexible-CM, common-Fréchet,
CM+ZC, ZC-only) calls `operator_hessian_weights.jl::operator_prep_for_hessian!` instead, which
recomputes `r` via the same dense-G-free operator forward kernels the FG callback already uses (or
reuses the FG callback's own already-published `obj.arg0` under a strict same-point cache). This
function is retained, byte-identical, purely for `moment_representation=:dense_reference` reference
gates -- see docs/SHARED_OPERATOR_DUAL_INDEX_AND_HESSIAN_WEIGHTS_2026-07-28.md.
"""
function _archC_prep_for_hessian!(obj, x)
    @unpack H, arg0, arg1, outer_constr_index, Psi! = obj
    BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
    Psi!(arg1, arg0)
    record_hessian_weight_dense_recompute!()
    return nothing
end

"""
    _prep_dual_index_for_archC!(cctx::CMBinHessCtx, obj, x)
    _prep_dual_index_for_archA!(octx::OriginZCCoreHessCtx, obj, x)

No-moments/no-composite-G task (2026-07-28): the ONE call every `archC_hess_cb_builder`/
`archA_partitioned_hess_cb_builder` closure makes to obtain `obj.arg0` -- dispatches to the shared
`operator_prep_for_hessian!(st, x)` (dense-G-free, cached) whenever this context's operator-mode FG
state is available, else falls back to the retained `_archC_prep_for_hessian!` (explicit
`:dense_reference` reference path). `cctx.cmlookup_st`/`octx.fg_lookup_st` are the SAME `Any`-typed
lazily-built state fields the FG-callback side already populates (`CMLookupState`/
`CMFrechetLookupState`/`CMMeanZCOperatorState` for `cctx`, `OriginZCOperatorState` for `octx`) --
these closures never construct or own a separate state, purely read what the FG side already built.
"""
function _prep_dual_index_for_archC!(cctx::CMBinHessCtx, obj, x)
    _record_cm_hessian_capture_x!(x)   # D=20 profiling task (2026-07-28): opt-in, see cm_hessian_subblock_profiling.jl
    @cmhess_prof "hessw_operator_prep" begin
        st = cctx.cmlookup_st
        if st !== nothing && cctx.inner_fg_backend !== :dense_reference
            operator_prep_for_hessian!(st, x)
        else
            _archC_prep_for_hessian!(obj, x)
        end
    end
    return nothing
end


# ============================================================================
# ARCHITECTURE D: matrix-free Hessian-vector-product diagnostic.
# Hv = (1/M) Z' (w .* (Z*v)), Z = H[:,2:1+outer_constr_index]. O(W*n) per
# call, O(n) extra memory (vs O(n^2) for the dense architectures). Diagnostic/
# validation candidate only, per task brief -- not expected to be a production
# winner unless it unexpectedly is (measured, not assumed).
# ============================================================================
function hvp_dense!(Hv::AbstractVector, obj, v::AbstractVector, zbuf::AbstractVector)
    @unpack H, M, arg0, arg2, ddPsi!, outer_constr_index = obj
    ddPsi!(arg2, arg0)
    Z = @view H[:, 2:1+outer_constr_index]
    mul!(zbuf, Z, v)                 # zbuf = Z*v            (W)
    zbuf .*= arg2                    # zbuf = w .* (Z*v)     (W)
    mul!(Hv, transpose(Z), zbuf)     # Hv   = Z' * zbuf      (n)
    Hv ./= M
    return Hv
end

function _callbackEvalHV_inner_dense!(kc, cb, evalRequest, evalResult, userParams)
    obj = userParams
    x = evalRequest.x
    v = evalRequest.vec
    @prof "inner_dual_hvp_callback_dense" begin
        _archC_prep_for_hessian!(obj, x)
        n = length(v)
        zbuf = Vector{Float64}(undef, size(obj.H, 1))
        Hv = Vector{Float64}(undef, n)
        hvp_dense!(Hv, obj, v, zbuf)
        @views evalResult.hessVec[1:n] .= Hv
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

# ============================================================================
# Generic KNITRO wiring, mirroring chunked_hessian.jl's
# inner_loop_KNITRO_chunked / inner_loop_internal_chunked pattern EXACTLY,
# parameterized by a `hess_cb_builder(obj) -> Function` (dense hessopt=1
# variants) or by `hvp=true` (hessopt=5 product variant). FG callback,
# variable/bound/init-value setup, complementarity wiring are all reused
# UNCHANGED from oracle_fast.jl (included by callers before this file).
# ============================================================================
function inner_loop_KNITRO_archgeneric(obj; hess_cb_builder = nothing, hvp::Bool = false)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)

    # Ported from diag/fullA-inner-blas-threading (parallelism_guards.jl); see the same note in
    # oracle_fast.jl::inner_loop_KNITRO_profiled -- this is the CM (Architecture B/C) production
    # inner solve, reached via cm_production_value_v2 -> cm_base_state_v2 ->
    # inner_loop_internal_archgeneric.
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_profiled!)
        KNITRO.KN_set_cb_user_params(kc, cb, obj)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        hessopt = KNITRO.KN_get_int_param(kc, "hessopt")
        if hvp
            hessopt == 5 || error("inner_loop_KNITRO_archgeneric(hvp=true): expected hessopt=product(5), got $hessopt -- obj.inner_loop_opt must point at ek_inner_hvp.opt")
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalHV_inner_dense!)
        elseif hessopt == 1
            hess_cb = hess_cb_builder(obj)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        @prof "inner_knitro_dual_solve_arch" begin
            KNITRO.KN_solve(kc)
        end
        nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        # end-to-end profiling task (2026-07-29): stash the real per-solve KNITRO status/iteration/
        # FG/Hessian-eval record BEFORE KN_free destroys kc, so a caller (e2e_record_inner_call!)
        # can read it immediately after this function returns. No-op (single Ref check) unless the
        # caller opted into e2e profiling -- see e2e_outer_profiling.jl.
        isdefined(Main, :e2e_knitro_log_stash!) && E2E_PROFILE_ENABLED[] && e2e_knitro_log_stash!(nSTatus, kc)
        KNITRO.KN_free(kc)

        return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
    finally
        CS.guard_exit_inner_solve!()
    end
end

function inner_loop_internal_archgeneric(obj, θ; hess_cb_builder = nothing, hvp::Bool = false)
    # Task C (2026-07-27): inner_loop_KNITRO_archgeneric (below) ALWAYS registers the generic dense
    # FG callback (callbackEvalFG_inner!, cc_algo/inner_loop_functions.jl) -- every operator-FG family
    # dispatches through a DIFFERENT top-level function instead (inner_loop_internal_cmlookup_production/
    # _cmfrechetlookup_production/etc., each with its own dedicated operator FG callback), so this
    # function firing at all is unconditionally equivalent to "the dense-reference generic FG path was
    # used for this inner solve" -- safe to record unconditionally, no branch needed (this was the
    # counter's own documented gap, see docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_2026-07-27.md B.2:
    # "NOT wired at the one site that would make it meaningful ... inner_loop_internal_archgeneric").
    record_generic_dense_fg!()
    @prof "inner_moment_build" begin
        obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    end
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_archgeneric(obj; hess_cb_builder = hess_cb_builder, hvp = hvp)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end
    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess
    end
end

"Architecture A's hess_cb_builder: the unchanged generic `hessian!`, wired via the profiled callback exactly as oracle_fast.jl does. Retained as the :dense_reference / emergency-revert path (task §5) -- origin-ZC's production default is `archA_partitioned_hess_cb_builder` below."
archA_hess_cb_builder(obj) = _callbackEvalH_inner_profiled!

"""
    OriginZCCoreHessCtx

port/shared-winner-pair-core-hessian-production-2026-07-25 (task §4.4):
origin-ZC's per-context state for the H_EE/H_ER/H_RR partition --
`H = [[H_EE H_ER];[H_ER' H_RR]]`, H_EE = the shared exact winner-pair
backend, H_ER/H_RR = the ORIGINAL dense BLAS contraction Architecture A
already did, now restricted to just those two blocks instead of one
monolithic gemm over core+restriction columns combined. H_ER computed once,
H_RE never computed independently (mirrored via `transpose`).
"""
mutable struct OriginZCCoreHessCtx
    NCORE::Int      # = ncore_econ (winner-pair core width, INCLUDING the zeta/intercept column)
    n_eta::Int      # restriction (origin-specific mean/pairwise-ZC) column count
    core_cf_ref::Ref{Any}
    core_ws::Union{Nothing,CoreExactHessianWorkspace}
    core_ws_for::Any
    core_hessian_backend::Symbol
    core_hessian_workers::Int
    core_hessian_storage::Symbol
    # port/shared-inner-fg-operator-and-verification-2026-07-26: origin-ZC's operator FG state.
    # `Any`-typed (mirrors CMBinHessCtx's own `cmlookup_st` field, cm_lookup_kernels.jl) so this
    # file has no load-order dependency on zc_restriction_operator.jl/cm_originzc_lookup_kernels.jl
    # -- those files self-include-guard their own dependencies instead.
    fg_backend::Symbol                 # :dense_reference (default) | :operator
    fg_zc_op::Any                      # ZCRestrictionOperator, built once per campaign, or `nothing`
    fg_layout::Any                     # MeanZCTargetLayout, or `nothing`
    fg_lookup_st::Any                  # OriginZCOperatorState, cached once per octx, or `nothing`
    # Winner-aware H_ER phase (2026-07-27, task Section 5): which backend fills H_ER (core x
    # mean/pairwise-ZC cross block) in `archA_partitioned_hess_cb_builder`. :dense_reference
    # (default until this section's own gates pass) | :winner_bin
    # (winner_pair_cross_hessian_zc_block!, winner_pair_cross_hessian.jl -- reuses the SAME
    # core_ws/wctx H_EE already builds). HRR stays dense BLAS unconditionally, out of scope.
    zc_cross_hessian_backend::Symbol
    zc_cross_scratch::Union{Nothing,WinnerZCCrossScratch}
    # CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): DEDICATED raw-ZC-feature state
    # for the NEW shared direct H_ZZ primitive (`zc_restriction_operator.jl::zc_restriction_gram!`)
    # -- intentionally SEPARATE from `fg_zc_op`/`fg_layout` above (those are gated on
    # `fg_backend===:operator` and `archOZ_verified_state`'s own `:operator` verification path uses
    # `fg_zc_op !== nothing` as an explicit "was this built with :operator" prerequisite check;
    # repurposing them here, needed regardless of fg_backend, would silently defeat that check).
    # Built ALWAYS (unlike `fg_zc_op`) whenever `octx.n_eta > 0`, `nothing` otherwise. `hzz_zc_ws`
    # holds the CURRENT outer point's refreshed targets; `nu_ref` is the shared box
    # `archOZ_base_state`/`_verified_state` (cm_originzc_production.jl) publish the current νfull
    # into (mirrors `core_cf_ref`'s own "wrapper publishes, Hessian callback reads" pattern).
    hzz_zc_op::Any
    hzz_zc_layout::Any
    hzz_zc_ws::Any
    nu_ref::Base.RefValue{Vector{Float64}}
    hzz_centered::Union{Nothing,ZCCenteredScratch}
    # Legacy-H cleanup (2026-07-28): the skip-fill variant of `wrap_moments_with_originzc`'s
    # moments! closure (economic-block-only skip, sharing the same core_cf_ref), mirroring
    # CMBinHessCtx's own `moments_skip!` field exactly. `nothing` until built.
    moments_skip!::Union{Nothing,Function}
    # optimize/structured-cross-hessian-ZC-CM-2026-07-28 (+ ADDENDUM): mirrors CMBinHessCtx's own
    # cross_hessian_threaded/cross_hessian_workers/zc_gram_backend/zc_gram_workers/raw_zc_ws fields
    # exactly -- see that struct's own docstrings.
    cross_hessian_threaded::Bool
    cross_hessian_workers::Int
    zc_gram_backend::Symbol
    zc_gram_workers::Int
    raw_zc_ws::Union{Nothing,ZCRawWeightedWorkspace}
    # True no-H operator bundle (2026-07-28 continuation): mirrors CMBinHessCtx's own econ_ctx field.
    econ_ctx::Any
    # genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Section 7/8: mirrors CMBinHessCtx's own
    # zc_ez_backend/zc_drawmajor fields exactly, so origin-ZC's H_ER (this task's H_EZ block, filled
    # at line ~1660 below via winner_pair_cross_hessian_zc_block_threaded!) can ALSO select the
    # :drawmajor / :drawmajor_v2 candidates -- previously this dispatch had no backend Symbol at all
    # (always :winner_bin's own threaded kernel). New fields, appended via the existing outer
    # kwarg-constructor pattern; default :winner_bin is byte-identical to pre-existing behavior.
    zc_ez_backend::Symbol
    zc_drawmajor::Any
    # Restricted-inner-endtoend task (2026-08-01), §8: ZC-only's own profiled/reduced-economic-layout
    # scratch, mirroring CMBinHessCtx's own identically-named fields field-for-field (same rationale,
    # same "gather at assembly" design -- see that struct's own docstrings for the full derivation).
    # `profiled_reduced_wctx`/`_for`/`profiled_hee_packed`: H_EE, via the SAME family-independent
    # `build_reduced_homogeneous_winner_pair_ctx`/`reduced_homogeneous_winner_pair_hessian!` kernels
    # `_fill_cm_HEE!` already uses -- H_EE has nothing to do with the restriction type, only the
    # economic block, so this is a genuine reuse, not a re-derivation. `profiled_full_wctx`/`_for`/
    # `profiled_full_ws`: the FULL (unreduced) WinnerPairHessCtx/WinnerBinCrossScratch for the H_EZ
    # (HER) gather. `profiled_her_full`: the corresponding full-width `(wctx.ncolI+1) x n_eta_total`
    # raw HER scratch, gathered down into the existing (already reduced-sized) HER view before the
    # Hessian-write step.
    profiled_layout::Any
    profiled_theta_ref::Base.RefValue{Any}
    profiled_reduced_wctx::Any
    profiled_reduced_wctx_for::Any
    profiled_hee_packed::Vector{Float64}
    profiled_full_wctx::Any
    profiled_full_wctx_for::Any
    profiled_full_ws::Union{Nothing,WinnerZCCrossScratch}
    profiled_her_full::Matrix{Float64}
end

"""
    build_originzc_core_hess_ctx(aug, ctx; core_hessian_backend=:exact_winner_pair_parallel, core_hessian_workers=10, core_hessian_storage=:full_stride, fg_backend=:dense_reference) -> OriginZCCoreHessCtx

`aug` is `build_originzc_augmented_obj(...)`'s return value -- needs
`aug.ncore_econ` and `aug.core_cf_ref` (the shared box
`wrap_moments_with_originzc`'s moments! closure publishes a fresh
`CompressedFactual` into every outer point, mirroring flexible CM's
`core_cf_ref`). `fg_backend=:operator` opts into the new shared-economic-operator +
ZC-restriction-operator FG (port/shared-inner-fg-operator-and-verification-2026-07-26); default
`:dense_reference` preserves the pre-existing `inner_loop_internal_archgeneric` dense path exactly.
"""
function build_originzc_core_hess_ctx(aug, ctx = nothing; core_hessian_backend::Symbol = ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT[],
        core_hessian_workers::Int = ORIGINZC_CORE_HESSIAN_WORKERS_DEFAULT[], core_hessian_storage::Symbol = ORIGINZC_CORE_HESSIAN_STORAGE_DEFAULT[],
        fg_backend::Symbol = :dense_reference,
        zc_cross_hessian_backend::Symbol = ORIGINZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT[],
        cross_hessian_threaded::Bool = CROSS_HESSIAN_THREADED_DEFAULT[],
        cross_hessian_workers::Int = CROSS_HESSIAN_WORKERS_DEFAULT[],
        zc_gram_backend::Symbol = ZC_GRAM_BACKEND_DEFAULT[],
        zc_gram_workers::Int = ZC_GRAM_THREADED_WORKERS_DEFAULT[],
        zc_ez_backend::Symbol = ZC_EZ_BACKEND_DEFAULT[],
        profiled_layout = nothing)   # restricted-inner-endtoend task (2026-08-01): mirrors
        # build_cm_bin_ctx's own kwarg -- `nothing` (every existing caller) preserves this function
        # byte-for-byte.
    n_eta_total = aug.obj_cm.outer_constr_index - aug.ncore_econ
    core_cf_ref = hasproperty(aug, :core_cf_ref) ? aug.core_cf_ref : Ref{Any}(nothing)
    # port/shared-inner-fg-operator-and-verification-2026-07-26: build the ZC restriction operator
    # EAGERLY here (aug.Zraw_all/Zpairraw_all/layout are already available at this call site) rather
    # than lazily at solve time, so inner_loop_internal_originzc_operator doesn't need `aug` threaded
    # through to solve time at all -- octx alone is a complete, self-sufficient campaign context.
    # Self-guarded include (this file's own established convention, see lines 43-45 above) so this
    # file has no unconditional load-order dependency on zc_restriction_operator.jl for callers that
    # never request fg_backend=:operator.
    local fg_zc_op, fg_layout
    if fg_backend === :operator
        isdefined(Main, :ZCRestrictionOperator) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))
        fg_zc_op = ZCRestrictionOperator(aug.Zraw_all, aug.Zpairraw_all, size(aug.Zraw_all[1], 2))
        fg_layout = aug.layout
    else
        fg_zc_op = nothing
        fg_layout = nothing
    end
    # CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): DEDICATED raw-ZC-feature state
    # for the shared direct H_ZZ primitive, built ALWAYS (independent of fg_backend) whenever this
    # arm actually has a restriction block (`n_eta_total > 0`) -- see the struct field's own
    # docstring for why this is a separate object from `fg_zc_op`/`fg_layout`.
    local hzz_zc_op, hzz_zc_layout, hzz_zc_ws
    if n_eta_total > 0
        isdefined(Main, :ZCRestrictionOperator) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))
        hzz_zc_op = ZCRestrictionOperator(aug.Zraw_all, aug.Zpairraw_all, size(aug.Zraw_all[1], 2))
        hzz_zc_layout = aug.layout
        hzz_zc_ws = ZCRestrictionWorkspace(hzz_zc_op)
    else
        hzz_zc_op = nothing
        hzz_zc_layout = nothing
        hzz_zc_ws = nothing
    end
    profiled_theta_ref = hasproperty(aug, :theta_ref) ? aug.theta_ref : Ref{Any}(nothing)
    return OriginZCCoreHessCtx(aug.ncore_econ, n_eta_total, core_cf_ref, nothing, nothing,
        core_hessian_backend, core_hessian_workers, core_hessian_storage,
        fg_backend, fg_zc_op, fg_layout, nothing,
        zc_cross_hessian_backend, nothing,
        hzz_zc_op, hzz_zc_layout, hzz_zc_ws, Ref(Float64[]), nothing,
        hasproperty(aug, Symbol("moments_skip!")) ? aug.moments_skip! : nothing,
        cross_hessian_threaded, cross_hessian_workers,
        zc_gram_backend, zc_gram_workers, nothing,   # raw_zc_ws: lazily built on first H_ZZ call
        ctx,   # econ_ctx: true no-H operator bundle continuation
        zc_ez_backend, nothing,   # zc_drawmajor: lazily built on first :drawmajor/:drawmajor_v2 call
        profiled_layout, profiled_theta_ref, nothing, nothing, Float64[],
        nothing, nothing, nothing, Matrix{Float64}(undef, 0, 0))
end

"""
Winner-aware H_ER phase (2026-07-27), Section 5: decide whether THIS Hessian callback may use the
`:winner_bin` cross-Hessian backend for origin-ZC's H_ER (core x mean/pair cross) block. Requires
(a) the backend is actually requested, (b) `octx.n_eta > 0` (there IS a restriction block to fill),
and (c) `octx.core_ws`/`core_ws_for` were ACTUALLY refreshed for the CURRENT `cf` by this SAME
callback's own H_EE fill just above (not stale, not a dense fallback) -- same discipline
`_cm_cross_hessian_wants_winner_bin`/`_cm_zc_cross_hessian_wants_winner_bin` already establish.
Not silent: callers that want `:winner_bin` but land here `false` fall back to the dense HER path
and record `record_dense_cross_hessian_call!`.
"""
function _originzc_zc_cross_hessian_wants_winner_bin(octx::OriginZCCoreHessCtx, cf)
    return octx.zc_cross_hessian_backend === :winner_bin &&
           octx.n_eta > 0 &&
           cf isa CompressedFactual &&
           octx.core_ws !== nothing &&
           octx.core_ws_for === cf
end

"Ensure `octx.zc_cross_scratch` is sized for the current (W,n_restr); rebuild only on a genuine size change, never per-Hessian-callback."
function _ensure_originzc_zc_cross_scratch!(octx::OriginZCCoreHessCtx, W::Int, n_restr::Int)
    cs = octx.zc_cross_scratch
    if cs === nothing || cs.W != W || cs.max_nx < n_restr
        octx.zc_cross_scratch = WinnerZCCrossScratch(W, n_restr)
    end
    return octx.zc_cross_scratch
end

"""
    _prep_dual_index_for_archA!(octx::OriginZCCoreHessCtx, obj, x)

No-moments/no-composite-G task (2026-07-28): ZC-only's analogue of
`_prep_dual_index_for_archC!` -- see that function's docstring above for the full contract.
"""
function _prep_dual_index_for_archA!(octx::OriginZCCoreHessCtx, obj, x)
    st = octx.fg_lookup_st
    if st !== nothing && octx.fg_backend !== :dense_reference
        operator_prep_for_hessian!(st, x)
    else
        _archC_prep_for_hessian!(obj, x)
    end
    return nothing
end

"""
    archA_partitioned_hess_cb_builder(octx::OriginZCCoreHessCtx)

Origin-ZC's production hess_cb_builder: partitions the Hessian into
H_EE (shared exact winner-pair backend) and H_ER/H_RR (existing dense BLAS,
restricted to those two blocks -- never a combined core+restriction gemm).
Falls back to the fully dense combined gemm (byte-identical to pre-port
Architecture A) whenever `octx.core_cf_ref[]` is unavailable for this point
or `octx.core_hessian_backend === :dense_reference`.
"""
function archA_partitioned_hess_cb_builder(octx::OriginZCCoreHessCtx)
    # True no-H operator bundle (2026-07-28 continuation): `∂∂f_∂∂x` (used below purely as a
    # per-call n x n scratch buffer for assembling the block Hessian before packing -- every entry
    # used is fully OVERWRITTEN by the HEE/HER/HRR fills before being read, never relies on a prior
    # value) is ALSO a `PsiObjectiveBundleImplicit`-only field, absent from `OperatorPsiBundle`.
    # Rather than adding an obj-type-dispatched accessor for this one purely-local scratch use, this
    # closure now owns its OWN persistent scratch matrix (built once per KNITRO solve, when this
    # builder is constructed -- not once per Hessian callback invocation), used identically for
    # BOTH bundle types. This removes the obj.∂∂f_∂∂x dependency entirely rather than working around it.
    n_scratch = octx.NCORE + octx.n_eta
    scratch_full = Matrix{Float64}(undef, n_scratch, n_scratch)
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        obj = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archA_partitioned" begin
            # Genuine-cold ZC Hessian K=3 optimization task (2026-08-01), Section 6: block-level
            # `@cmhess_prof` instrumentation for origin-ZC, at the same resolution CM+ZC's
            # `_fill_cm_HEE!`/`hessian_cm_structured_v2!` already have. Purely additive timing
            # wraps around the EXISTING calls below -- no control flow, argument, or numerical
            # change. `originZC_misc` bundles `_prep_dual_index_for_archA!`, the `@unpack`/dense-
            # handle resolution, and `ddPsi!` (CM+ZC keeps `ddpsi` as its own top-level label,
            # cm_hessian_threaded.jl:183; not split out separately here since none of this callback's
            # OTHER labels depend on further decomposing it, and ddPsi! is a cheap O(M) elementwise
            # call, not expected to be a material share -- see ZC_HESSIAN_PROFILER_LABEL_AUDIT).
            @cmhess_prof "originZC_misc" begin
                _prep_dual_index_for_archA!(octx, obj, xloc)   # refreshes obj.arg0 from the current dual point (operator-cached or dense-fallback), precondition for ddPsi!(arg2,arg0) below
                # True no-H operator bundle (2026-07-28 continuation): was `@unpack H, H_copy, M, arg0,
                # arg2, ddPsi!, ∂∂f_∂∂x = obj` -- an unconditional H/H_copy unpack that would throw
                # immediately on OperatorPsiBundle (no H/H_copy field). H/H_copy are only actually read
                # inside this callback's own dense-fallback branches below (all already guarded by
                # `winner_bin_ok`/`cf isa CompressedFactual` checks); `_dense_H_or_nothing`/
                # `_dense_H_copy_or_nothing` mirror the identical dispatch flexible-CM's
                # hessian_cm_structured! already uses.
                @unpack M, arg0, arg2, ddPsi! = obj
                H = _dense_H_or_nothing(obj)
                H_copy = _dense_H_copy_or_nothing(obj)
                ∂∂f_∂∂x = scratch_full
                ddPsi!(arg2, arg0)
                NCORE = octx.NCORE; n_eta_total = octx.n_eta; n = NCORE + n_eta_total
                cf = octx.core_cf_ref[]   # a CompressedFactual (success) OR a Symbol fallback reason
            end
            if octx.profiled_layout !== nothing
                # Restricted-inner-endtoend task (2026-08-01), §8: ZC-only's profiled/reduced-
                # economic-layout branch -- mirrors hessian_cm_structured!'s own profiled branch
                # (cm_hessian_architectures.jl) exactly: H_EE via the family-independent reduced
                # homogeneous kernel (SAME one _fill_cm_HEE! uses -- H_EE has nothing to do with the
                # restriction type), H_EZ (HER) via the SAME "gather at assembly" design H_EC/H_EF
                # already use (full-width wctx/ws, use_profiled_correction=true, gather retained
                # rows). H_ZZ (HRR) is UNCHANGED, copied verbatim from the branch below -- it reads
                # only octx.hzz_zc_op/octx.nu_ref (restriction-only state, independent of the
                # economic block width), confirmed by direct read, matching this task's own
                # "unchanged H_ZZ" instruction for the ZC-only family.
                cf isa CompressedFactual || error("archA_partitioned_hess_cb_builder: profiled_layout set but core_cf_ref[] is not a CompressedFactual (got $(typeof(cf))) -- the compressed-core fallback path is not supported for the reduced economic layout.")
                layout = octx.profiled_layout
                θ_full = octx.profiled_theta_ref[]
                θ_full === nothing && error("archA_partitioned_hess_cb_builder: profiled_layout set but profiled_theta_ref[] is nothing -- theta was never published for this outer point (moments! closure not yet called before this Hessian callback?).")
                # ---- H_EE ----
                if octx.profiled_reduced_wctx === nothing || octx.profiled_reduced_wctx_for !== cf
                    octx.profiled_reduced_wctx = build_reduced_homogeneous_winner_pair_ctx(cf, octx.econ_ctx, θ_full, layout)
                    octx.profiled_reduced_wctx_for = cf
                end
                wctx_r = octx.profiled_reduced_wctx
                n_ee = 1 + wctx_r.ncolI
                n_ee == NCORE || error("archA_partitioned_hess_cb_builder: reduced width n_ee=$n_ee (1+layout.total_reduced_economic_moments) != octx.NCORE=$NCORE -- OriginZCCoreHessCtx built inconsistently with this layout.")
                npacked = n_ee * (n_ee + 1) ÷ 2
                length(octx.profiled_hee_packed) == npacked || (octx.profiled_hee_packed = Vector{Float64}(undef, npacked))
                reduced_homogeneous_winner_pair_hessian!(octx.profiled_hee_packed, obj, wctx_r)
                HEE = @view ∂∂f_∂∂x[1:NCORE, 1:NCORE]
                packed = octx.profiled_hee_packed
                k = 1
                @inbounds for i in 1:n_ee
                    for j in i:n_ee
                        v = packed[k]; k += 1
                        HEE[i, j] = v
                        HEE[j, i] = v
                    end
                end
                # ---- H_EZ (HER), profiled: gather from full-width wctx_full/ws_full ----
                if n_eta_total > 0
                    has_france = layout.france_ratio_reduced_j > 0
                    bi_slot = has_france ? dest_slot(octx.econ_ctx, octx.econ_ctx.bi) : 0
                    gpσ = has_france ? θ_full[3 + octx.econ_ctx.D]^θ_full[2] : 0.0
                    denom_cf = has_france ? gpσ * octx.econ_ctx.γ.LPrime[octx.econ_ctx.bi] : 0.0
                    if octx.profiled_full_wctx === nothing || octx.profiled_full_wctx_for !== cf
                        octx.profiled_full_wctx = build_winner_pair_ctx(cf; bi_slot = bi_slot, gpσ = gpσ, denom_cf = denom_cf)
                        octx.profiled_full_wctx_for = cf
                    end
                    wctx_full = octx.profiled_full_wctx
                    ws_full = octx.profiled_full_ws
                    if ws_full === nothing || ws_full.W != wctx_full.W || ws_full.max_nx < n_eta_total || ws_full.Ddest != wctx_full.Ddest
                        ws_full = WinnerZCCrossScratch(wctx_full.W, n_eta_total, wctx_full.Ddest)
                        octx.profiled_full_ws = ws_full
                    end
                    record_winner_cross_hessian_call!()
                    winner_pair_cross_hessian_zc_prep!(ws_full, wctx_full, arg2)
                    op = octx.hzz_zc_op
                    refresh_zc_targets!(octx.hzz_zc_ws, op, octx.hzz_zc_layout, octx.nu_ref[])
                    octx.hzz_centered = ensure_zc_centered_scratch!(octx.hzz_centered, op, size(arg2, 1))
                    refresh_zc_centered!(octx.hzz_centered, op, octx.hzz_zc_ws, arg2; fill_S = false)
                    nx = n_restriction(op)
                    Z = @view octx.hzz_centered.Zc[:, 1:nx]
                    if size(octx.profiled_her_full) != (wctx_full.ncolI + 1, n_eta_total)
                        octx.profiled_her_full = Matrix{Float64}(undef, wctx_full.ncolI + 1, n_eta_total)
                    end
                    HER_full = octx.profiled_her_full
                    # Phase 7 (dispatch-counters, 2026-08-02): CONFIRMED LIVE -- same gap as CM+ZC's
                    # own profiled H_EM branch above (cm_hessian_architectures.jl's _fill_cm_HEE!):
                    # this profiled/reduced-layout H_EZ gather calls the base kernel unconditionally,
                    # never consulting `octx.zc_ez_backend` -- :drawmajor_v2 never dispatches here
                    # regardless of that Ref's value. Recorded so it is visible, not silently absent
                    # from the counters; not fixed here (out of this task's surgical scope).
                    record_drawmajor_v2_fallback!()
                    winner_pair_cross_hessian_zc_block!(HER_full, wctx_full, ws_full, arg2, Z, M; use_profiled_correction = true)
                    n_bilateral = length(layout.retained_full_factual_j)
                    HER = @view ∂∂f_∂∂x[1:NCORE, NCORE+1:n]
                    @views HER[1, :] .= HER_full[1, :]
                    @inbounds for kk in 1:n_bilateral
                        j_full = layout.retained_full_factual_j[kk]
                        @views HER[1 + kk, :] .= HER_full[1 + j_full, :]
                    end
                    has_france && (@views HER[NCORE, :] .= HER_full[1 + wctx_full.ncolI, :])
                    # ---- H_ZZ (HRR) -- UNCHANGED, copied verbatim from the non-profiled branch below ----
                    HRR = @view ∂∂f_∂∂x[NCORE+1:n, NCORE+1:n]
                    if octx.zc_cross_hessian_backend === :winner_bin && octx.hzz_zc_op !== nothing
                        record_winner_cross_hessian_call!()
                        refresh_zc_targets!(octx.hzz_zc_ws, op, octx.hzz_zc_layout, octx.nu_ref[])
                        octx.hzz_centered = ensure_zc_centered_scratch!(octx.hzz_centered, op, M)
                        refresh_zc_centered!(octx.hzz_centered, op, octx.hzz_zc_ws, arg2; fill_S = octx.zc_gram_backend === :reference)
                        if octx.zc_gram_backend === :reference
                            zc_restriction_gram!(HRR, octx.hzz_centered, op, M)
                        else
                            octx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(octx.raw_zc_ws, op, M)
                            refresh_zc_raw_target_vector!(octx.raw_zc_ws, octx.hzz_zc_ws, op)
                            zc_gram_dispatch!(HRR, octx.zc_gram_backend, nothing, op, octx.raw_zc_ws, arg2, M; workers = octx.zc_gram_workers)
                        end
                    else
                        record_dense_cross_hessian_call!()
                        H_copy === nothing && error("archA_partitioned_hess_cb_builder: profiled branch reached the dense H_ZZ (HRR) fallback for an operator-mode bundle with no H_copy field -- provably unreachable in production; indicates a real configuration bug.")
                        HC_eta = @view H_copy[:, 2+NCORE:1+n]
                        BLAS.gemm!('T', 'N', 1 / M, HC_eta, HC_eta, 0.0, HRR)
                    end
                end
            elseif cf isa CompressedFactual && octx.core_hessian_backend !== :dense_reference
                if octx.core_ws === nothing || octx.core_ws_for !== cf
                    octx.core_ws = build_core_exact_hessian_workspace(cf)
                    octx.core_ws_for = cf
                    record_compressed_core_rebuild!()
                end
                HEE = @view ∂∂f_∂∂x[1:NCORE, 1:NCORE]
                @cmhess_prof "originZC_H_EE_core" fill_core_hessian_upper!(HEE, arg2, obj, octx.core_ws;
                    backend = octx.core_hessian_backend, workers = octx.core_hessian_workers, storage = octx.core_hessian_storage)
                if n_eta_total > 0
                    HER = @view ∂∂f_∂∂x[1:NCORE, NCORE+1:n]
                    zc_ready = octx.zc_cross_hessian_backend === :winner_bin && octx.hzz_zc_op !== nothing
                    winner_bin_ok = _originzc_zc_cross_hessian_wants_winner_bin(octx, cf)
                    if winner_bin_ok && zc_ready
                        # No-moments/no-composite-G task (2026-07-28): `Z` (the "already-centered
                        # restriction columns") is now sourced from `octx.hzz_centered.Zc` --
                        # `zc_restriction_operator.jl`'s own docstring confirms this is BIT-IDENTICAL
                        # to `wrap_moments_with_originzc`'s dense `obj.H` Z columns ("the SAME
                        # quantity... never read from obj.H"), and it is already independently
                        # validated (this IS the same scratch H_RR/`zc_restriction_gram!` below
                        # already uses, gated at D=4/D=20). Refreshed HERE (before H_ER, not after)
                        # so both H_ER and H_RR share the one computation -- `obj.H`'s restriction
                        # columns are no longer read anywhere in this callback.
                        @cmhess_prof "originZC_H_EZ_prep" begin
                            op = octx.hzz_zc_op
                            refresh_zc_targets!(octx.hzz_zc_ws, op, octx.hzz_zc_layout, octx.nu_ref[])
                            octx.hzz_centered = ensure_zc_centered_scratch!(octx.hzz_centered, op, size(arg2, 1))
                            # ADDENDUM (2026-07-28): this call's OWN ZcS is never read (HRR's block below
                            # redoes refresh_zc_centered! independently right before it needs ZcS) -- skip
                            # it here unconditionally, mirroring CM+ZC's own fill_S gating.
                            refresh_zc_centered!(octx.hzz_centered, op, octx.hzz_zc_ws, arg2; fill_S = false)
                            record_winner_cross_hessian_call!()
                            wctx = serial_ctx(octx.core_ws)
                            octx.zc_cross_scratch = _ensure_originzc_zc_cross_scratch!(octx, wctx.W, n_eta_total)
                            winner_pair_cross_hessian_zc_prep!(octx.zc_cross_scratch, wctx, arg2)
                            nx = n_restriction(op)
                            Z = @view octx.hzz_centered.Zc[:, 1:nx]   # already-centered restriction columns, UNweighted
                        end
                        @cmhess_prof "originZC_H_EZ_fill" if octx.zc_ez_backend === :drawmajor
                            record_drawmajor_v2_fallback!()
                            nbilateral_o = wctx.has_cf ? wctx.ncolI - 1 : wctx.ncolI
                            octx.zc_drawmajor = ensure_winner_zc_drawmajor_scratch!(octx.zc_drawmajor, wctx.W, wctx.Ddest, nbilateral_o, nx, octx.cross_hessian_workers)
                            winner_pair_cross_hessian_zc_block_drawmajor!(HER, wctx, octx.zc_cross_scratch, octx.zc_drawmajor, arg2, Z, M; workers = octx.cross_hessian_workers)
                        elseif octx.zc_ez_backend === :drawmajor_v2
                            record_drawmajor_v2_dispatch!()
                            nbilateral_o = wctx.has_cf ? wctx.ncolI - 1 : wctx.ncolI
                            octx.zc_drawmajor = ensure_winner_zc_drawmajor_v2_scratch!(octx.zc_drawmajor, wctx.W, wctx.Ddest, nbilateral_o, nx, octx.cross_hessian_workers)
                            winner_pair_cross_hessian_zc_block_drawmajor_v2!(HER, wctx, octx.zc_cross_scratch, octx.zc_drawmajor, arg2, Z, M; workers = octx.cross_hessian_workers)
                        elseif octx.cross_hessian_threaded
                            record_drawmajor_v2_fallback!()
                            winner_pair_cross_hessian_zc_block_threaded!(HER, wctx, octx.zc_cross_scratch, arg2, Z, M; workers = octx.cross_hessian_workers)
                        else
                            record_drawmajor_v2_fallback!()
                            winner_pair_cross_hessian_zc_block!(HER, wctx, octx.zc_cross_scratch, arg2, Z, M)
                        end
                    else
                        record_dense_cross_hessian_call!()
                        # moment_representation threading task (2026-07-29): matches _fill_cm_HEE!'s
                        # identical guards (lines ~873/887 above) -- this dense H_ER fallback is
                        # provably unreachable in production (winner_bin_ok && zc_ready should always
                        # hold at defaults), but was previously an unguarded `H_copy[...] .= H[...]`
                        # that would throw a confusing `MethodError: view(::Nothing, ...)` rather than
                        # a clear error for an operator-mode bundle (H/H_copy both nothing).
                        H === nothing && error("archA_partitioned_hess_cb_builder: reached the dense H_ER cross-Hessian fallback for an operator-mode bundle with no H field -- this should be provably unreachable in production (winner_bin_ok && zc_ready should always hold); indicates a real configuration bug, not expected behavior.")
                        @views H_copy[:, 2+NCORE:1+n] .= H[:, 2+NCORE:1+n]
                        @views H_copy[:, 2+NCORE:1+n] .*= .√arg2
                        HC_eta = @view H_copy[:, 2+NCORE:1+n]
                        @views H_copy[:, 2:1+NCORE] .= H[:, 2:1+NCORE]
                        @views H_copy[:, 2:1+NCORE] .*= .√arg2
                        HC_core = @view H_copy[:, 2:1+NCORE]
                        BLAS.gemm!('T', 'N', 1 / M, HC_core, HC_eta, 0.0, HER)
                    end
                    # Hessian upper-only cleanup (2026-07-28): this family's final packing loop
                    # (below) is a plain copy that only ever reads i<=j -- there is no averaging
                    # step downstream (unlike CM/common-Frechet's `0.5*(Hfull[i,j]+Hfull[j,i])`).
                    # `∂∂f_∂∂x[NCORE+1:n, 1:NCORE]` (row>NCORE>=col, strictly lower-triangular) is
                    # therefore never read by anything -- the mirror write that used to populate it
                    # was provably dead computation. Removed; HER's upper-triangle position
                    # (`∂∂f_∂∂x[1:NCORE, NCORE+1:n]`, filled by whichever branch above ran) is the
                    # only copy the packer needs. See PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md.
                    HRR = @view ∂∂f_∂∂x[NCORE+1:n, NCORE+1:n]
                    # CM+ZC E/C/Z block-partition + H_CZ/H_ZZ release (2026-07-27): origin-ZC's HRR
                    # (task's H_ZZ) now dispatches to the SAME shared `zc_restriction_gram!`
                    # (zc_restriction_operator.jl) CM+ZC's own HMM uses -- computed DIRECTLY from
                    # `octx.hzz_zc_op`'s raw `Zraw_all`/`Zpairraw_all` feature matrices plus the
                    # current outer point's targets (`octx.nu_ref[]`, published by
                    # `archOZ_base_state`/`_verified_state`, cm_originzc_production.jl), NEVER from
                    # `HC_eta`/`obj.H`. Falls back to the ORIGINAL dense `HC_eta'*HC_eta` gemm
                    # (byte-identical to pre-refactor production) whenever `octx.zc_cross_hessian_
                    # backend !== :winner_bin` or `octx.hzz_zc_op` is unavailable.
                    if octx.zc_cross_hessian_backend === :winner_bin && octx.hzz_zc_op !== nothing
                        record_winner_cross_hessian_call!()
                        local op
                        @cmhess_prof "originZC_H_ZZ_weight" begin
                            op = octx.hzz_zc_op
                            refresh_zc_targets!(octx.hzz_zc_ws, op, octx.hzz_zc_layout, octx.nu_ref[])
                            # True no-H operator bundle (2026-07-28 continuation): was `size(H, 1)` --
                            # would throw on OperatorPsiBundle (H===nothing). `M` (already unpacked
                            # above, = obj.M = the draw count) is the identical value: H always has
                            # exactly M rows by construction, this was only ever using H for its size,
                            # never its contents.
                            octx.hzz_centered = ensure_zc_centered_scratch!(octx.hzz_centered, op, M)
                            # ADDENDUM (2026-07-28): H_ZZ (=HRR) backend dispatch, same options/dispatcher
                            # as CM+ZC's HMM -- shared, not a separate implementation.
                            refresh_zc_centered!(octx.hzz_centered, op, octx.hzz_zc_ws, arg2; fill_S = octx.zc_gram_backend === :reference)
                        end
                        @cmhess_prof "originZC_H_ZZ_gram" if octx.zc_gram_backend === :reference
                            record_blas_syrk_fallback!()
                            zc_restriction_gram!(HRR, octx.hzz_centered, op, M)
                        else
                            octx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(octx.raw_zc_ws, op, M)
                            refresh_zc_raw_target_vector!(octx.raw_zc_ws, octx.hzz_zc_ws, op)
                            zc_gram_dispatch!(HRR, octx.zc_gram_backend, nothing, op, octx.raw_zc_ws, arg2, M; workers = octx.zc_gram_workers)
                        end
                    else
                        record_dense_cross_hessian_call!()
                        # moment_representation threading task (2026-07-29): same guard rationale as
                        # the H_ER fallback above -- provably unreachable in production, but was
                        # unguarded (would throw a confusing UndefVarError on HC_eta, or MethodError
                        # on H_copy, for an operator-mode bundle).
                        H_copy === nothing && error("archA_partitioned_hess_cb_builder: reached the dense H_ZZ (HRR) cross-Hessian fallback for an operator-mode bundle with no H_copy field -- this should be provably unreachable in production; indicates a real configuration bug, not expected behavior.")
                        BLAS.gemm!('T', 'N', 1 / M, HC_eta, HC_eta, 0.0, HRR)
                    end
                end
            else
                H === nothing && error("archA_partitioned_hess_cb_builder: reached the fully-dense combined Hessian fallback for an operator-mode bundle with no H field -- this should be provably unreachable in production (cf should always be a valid CompressedFactual and core_hessian_backend should never be :dense_reference on this construction path); indicates a real configuration bug, not expected behavior.")
                @views H_copy[:, 2:1+n] .= H[:, 2:1+n]
                @views H_copy[:, 2:1+n] .*= .√arg2
                BLAS.gemm!('T', 'N', 1 / M, @view(H_copy[:, 2:1+n]), @view(H_copy[:, 2:1+n]), 0.0, ∂∂f_∂∂x)
                reason = octx.core_hessian_backend === :dense_reference ? :debug_reference_requested :
                          (cf isa Symbol ? cf : :other)
                record_core_hessian_call!(:dense_inline_fallback; fallback_reason = reason)
            end
            @cmhess_prof "originZC_pack" begin
                k = 1
                @inbounds for i in 1:n
                    for j in i:n
                        evalResult.hess[k] = ∂∂f_∂∂x[i, j]
                        k += 1
                    end
                end
            end
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end

"""
Architecture C's hess_cb_builder: closes over a CMBinHessCtx.

Allocation/Hessian port task §6.1/6.2: dispatches to the threaded Hessian
(hessian_cm_structured_v2!, cm_hessian_threaded.jl) when `cctx.use_threaded_bins` is true (the
production default as of this port, set by build_cm_bin_ctx/build_cm_meanzc_bin_ctx) -- validated
to agree with the serial path to ~1e-13 and measured 3.52x faster at a real, hard D=20/W=80,000/
L=50 point (test_cm_threaded_hessian.jl). Falls back to the original serial hessian_cm_structured!
unchanged when `cctx.use_threaded_bins` is false (an explicit opt-out, e.g.
build_cm_bin_ctx(ctx, aug; threaded_bins=false)). Every existing caller of archC_hess_cb_builder
(archC_base_state/archC_verified_state, cm_production_bundle.jl) needs no changes -- the dispatch
is entirely internal to this function.
"""
function archC_hess_cb_builder(cctx::CMBinHessCtx)
    if cctx.use_threaded_bins
        return (kc, cb, evalRequest, evalResult, userParams) -> begin
            o = userParams
            xloc = evalRequest.x
            @prof "inner_dual_hessian_callback_archC" begin
                _prep_dual_index_for_archC!(cctx, o, xloc)
                hessian_cm_structured_v2!(evalResult.hess, o, cctx; threaded_bins = true, tls = cctx.tls, use_syrk = true)
            end
            _INNER_CALL_COUNTERS[].n_hess_calls += 1
            return 0
        end
    end
    return (kc, cb, evalRequest, evalResult, userParams) -> begin
        o = userParams
        xloc = evalRequest.x
        @prof "inner_dual_hessian_callback_archC" begin
            _prep_dual_index_for_archC!(cctx, o, xloc)
            hessian_cm_structured!(evalResult.hess, o, cctx)
        end
        _INNER_CALL_COUNTERS[].n_hess_calls += 1
        return 0
    end
end
