# ============================================================================
# Production integration of the QMC-vs-pseudorandom draw-design comparison
# validated on `diag/fullA-d20-qmc-delta1` (worktree
# gravity-fullA-d20-qmc-delta1, tip 5882c16). That worktree is SUBSUMED by
# this file: its three generator functions (qmc_draws.jl::pseudorandom_U /
# halton_U / sobol_U) and its parallel U-injection context builder
# (qmc_context_real_d20.jl::d20_real_setup_qmc) are ADDITIVE, already present
# in this branch unmodified, and are reused here as-is -- nothing in either
# file is changed by this commit.
#
# This file adds exactly ONE new production entry point,
# `d20_real_setup_design`, that is a thin selector over three already-existing,
# already-validated code paths:
#
#   :pseudorandom      -> EXACTLY today's production call sequence
#                         (`Random.seed!(draw_seed); d20_real_setup(...)`,
#                         context_real_d20.jl, unmodified). Bit-for-bit
#                         identical behavior to before this file existed.
#   :sobol_randomized  -> qmc_draws.jl::sobol_U (Sobol.jl SobolSeq base
#                         sequence + a plain Cranley-Patterson random shift,
#                         mod 1 -- NOT Owen/digital scrambling; named
#                         "_randomized", not "_scrambled", precisely because
#                         of that) through qmc_context_real_d20.jl's
#                         d20_real_setup_qmc.
#   :halton_scrambled  -> qmc_draws.jl::halton_U (cc_algo/rhalton.jl,
#                         genuine per-digit Owen-style scrambling) through the
#                         same d20_real_setup_qmc path.
#
# All three route through the SAME inverse-CDF transform
# (qmc_context_real_d20.jl::exp_from_uniform01, a literal copy of
# prepare_cc/genRands.jl::genExpRands!'s `-log(1-u)`), the same country /
# dimension ordering (U is W x D, origin-indexed, D=D20_REAL=20), the same
# Frechet/moment/CC machinery (master_prepare_cc vs master_prepare_cc_qmc
# differ ONLY at the U-injection point, per qmc_context_real_d20.jl's own
# header), and the same 1/W quadrature weights (SamplingWeight = ones(W) in
# both). Nothing about the CC objective, divergence, moments, winner rules,
# or tie convention is touched here.
#
# The QMC context builder (`d20_real_setup_qmc`) does not build the
# pairwise/witness infeasibility-screen structures the production driver
# needs (unlike `d20_real_setup`, see context_real_d20.jl's `build_screen`
# block) -- this file replicates that block for the QMC branches so every
# draw_design gets identical screening behavior downstream (§ "screen parity"
# below), without modifying qmc_context_real_d20.jl itself.
# ============================================================================
include(joinpath(@__DIR__, "qmc_context_real_d20.jl"))   # -> d20_real_setup, d20_real_setup_qmc, context_real_d20.jl, exp_from_uniform01
include(joinpath(@__DIR__, "qmc_draws.jl"))               # -> pseudorandom_U, halton_U, sobol_U
using Sobol
using SHA   # AUD-11 fix: stable cross-process/cross-version checksums (stdlib, no Project.toml entry needed)
# sha256_of_matrix now lives in oracle.jl (context_fingerprint, the AUD-08 fix, is its primary
# unconditional consumer and oracle.jl is the more universally-included file); guard include so
# draw_design_meta works even if this file is used in isolation from oracle.jl (found live: a
# pre-existing test that includes oracle.jl but not draw_design.jl needs the reverse guard too --
# see oracle.jl's own top-of-file include of this pattern is NOT added there to avoid a real
# circular include, since production always includes draw_design.jl before oracle.jl anyway;
# this guard only helps the draw_design.jl-in-isolation direction).
isdefined(Main, :sha256_of_matrix) || include(joinpath(@__DIR__, "oracle.jl"))

const VALID_DRAW_DESIGNS = (:pseudorandom, :sobol_randomized, :halton_scrambled)

const DRAW_DESIGN_DESCRIPTIONS = Dict(
    :pseudorandom     => "Julia default RNG (MersenneTwister via Random.rand!) -- production baseline, byte-for-byte unmodified from the pre-existing d20_real_setup/Random.seed!(draw_seed) path",
    :sobol_randomized => "Sobol.jl SobolSeq deterministic base sequence + an independent Cranley-Patterson random shift (mod 1) per seed -- a plain randomized-QMC shift, NOT Owen/digital scrambling (Sobol.jl does not implement digital scrambling)",
    :halton_scrambled => "cc_algo/rhalton.jl scrambled Halton sequence -- genuine Owen-style per-digit scrambling (independent random digit permutation per radix digit, per dimension), ported from Art B. Owen's R code",
)

"""
    draw_design_meta(design, seed, D, W, U; timing=NamedTuple()) -> NamedTuple

Per-context metadata block logged for every draw design: design name,
randomization method (named honestly, see DRAW_DESIGN_DESCRIPTIONS), seed, D,
W, package versions, the transform convention, and checksums of both the
recovered raw-uniform draws and the transformed (Exp(1)) productivity draws.

The raw uniform draws are recovered from the final Exp(1) matrix via the
transform's own inverse (`U01 = 1 - exp(-U)`) rather than threaded through
separately -- `exp_from_uniform01` is a bijection on [0,1) x [0,Inf), so this
is exact (not an approximation), and it means the :pseudorandom path (whose
raw U01 is overwritten in place by genExpRands! and never separately
returned) gets the same checksum treatment as the QMC paths with no change to
genExpRands!/drawU.jl.
"""

function draw_design_meta(design::Symbol, seed::Int, D::Int, W::Int, Uexp::AbstractMatrix{Float64};
        timing::NamedTuple = NamedTuple())
    U01_recovered = 1.0 .- exp.(-Uexp)
    return (
        draw_design = design,
        randomization_method = DRAW_DESIGN_DESCRIPTIONS[design],
        draw_seed = seed,
        D = D,
        W = W,
        transform_convention = "Exp(1) via U[i] = -log(1 - U01[i]) (prepare_cc/genRands.jl::genExpRands!'s elementwise transform, applied identically to all three designs)",
        sobol_jl_version = design == :sobol_randomized ? string(pkgversion(Sobol)) : missing,
        julia_version = string(VERSION),
        checksum_uniform = sha256_of_matrix(U01_recovered),
        checksum_transformed = sha256_of_matrix(Uexp),
        n_at_boundary = count(x -> x <= 0.0 || x >= 1.0, U01_recovered),
        n_inf_transformed = count(!isfinite, Uexp),
        timing = timing,
    )
end

"""
    d20_real_setup_design(; W, δ=1.0, find_smallest=true, draw_design=:pseudorandom,
                           draw_seed=20260719, outer_loop_opt=..., inner_loop_opt=...,
                           needs_outer_moment_jacobian=false, build_screen=true,
                           log_draw_meta=true) -> ctx

Production entry point selecting among the three validated draw designs. Returns
the SAME NamedTuple shape `d20_real_setup` already returns (so every existing
downstream function -- evaluate_fullA, compute_winners, build_pivot_elimination,
composite_gradient_at_fast_buffered, build_ranged_screen_context, the whole
c10_d20_production_driver.jl machinery -- works unchanged), plus three extra
fields: `draw_design`, `draw_seed`, `draw_meta` (see `draw_design_meta`).

`draw_design = :pseudorandom` (the default) takes the EXACT same code path
production already takes -- `Random.seed!(draw_seed); d20_real_setup(...)` --
with no change to timing-relevant control flow. `log_draw_meta` defaults to
`true` (matching this task's "log per-context metadata" requirement) but can
be set `false` for a caller that wants a literal zero-added-instruction
:pseudorandom path (skips the checksum/hash computation entirely); see
docs/fullA_D20_draw_design_overhead_report.md (or the commit message) for the
measured overhead of leaving it on, which is negligible next to context
construction at D=20/W=80000.
"""
function d20_real_setup_design(; W::Int, δ::Float64 = 1.0, find_smallest::Bool = true,
        draw_design::Symbol = :pseudorandom, draw_seed::Int = 20260719,
        outer_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "csw_outer_25.opt"),
        inner_loop_opt::AbstractString = joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"),
        needs_outer_moment_jacobian::Bool = false, build_screen::Bool = true,
        log_draw_meta::Bool = true,
        # Part A (2026-07-23): passthrough to d20_real_setup's own destination_sample kwarg
        # (default :exclude_row, matching that function's new default). CM/originZC checkpoint
        # callers (cm_checkpoint.jl, cm_originzc_checkpoint.jl) explicitly pass :all_legacy here
        # since their moment/pivot-elimination layers are not rectangularized in this release.
        destination_sample::Symbol = :exclude_row)
    draw_design in VALID_DRAW_DESIGNS ||
        error("d20_real_setup_design: draw_design must be one of $(VALID_DRAW_DESIGNS), got :$(draw_design)")

    if draw_design == :pseudorandom
        # ---- EXACT existing production call sequence. Do not add anything
        # ahead of this that could perturb the global RNG state consumed by
        # d20_real_setup's internal draw (see c10_d20_production_driver.jl's
        # file-header note on why this Random.seed! call has to happen here,
        # immediately before d20_real_setup). ----
        Random.seed!(draw_seed)
        t_ctx = @elapsed ctx0 = d20_real_setup(W = W, δ = δ, find_smallest = find_smallest,
            outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
            needs_outer_moment_jacobian = needs_outer_moment_jacobian, build_screen = build_screen,
            destination_sample = destination_sample)
        timing = (uniform_and_transform = NaN, ctx_build = t_ctx,
                  pairwise = ctx0.screen_setup_wall.pairwise, witness = ctx0.screen_setup_wall.witness)
    else
        # Part A (2026-07-23) / QMC destination_sample port (2026-07-26): d20_real_setup_qmc now
        # mirrors d20_real_setup's destination_sample/row_idx/D_dest wiring exactly (see
        # qmc_context_real_d20.jl), so this branch simply passes destination_sample through --
        # no more hard-error opt-out needed for :exclude_row here.
        D = D20_REAL
        gen = draw_design == :sobol_randomized ? sobol_U : halton_U
        t_gen = @elapsed Uexp = gen(W, D; seed = draw_seed)
        t_ctx = @elapsed ctx_qmc = d20_real_setup_qmc(W = W, U_injected = Uexp, δ = δ, find_smallest = find_smallest,
            outer_loop_opt = outer_loop_opt, inner_loop_opt = inner_loop_opt,
            needs_outer_moment_jacobian = needs_outer_moment_jacobian, destination_sample = destination_sample)

        # ---- screen parity: d20_real_setup_qmc does not build these (it mirrors
        # d20_real_setup exactly except at the U-injection point, and predates the
        # infeasibility-screen wiring) -- replicate context_real_d20.jl's own
        # build_screen block verbatim so the QMC branches are drop-in compatible
        # with the driver's screened_eval (which reads ctx.pairwise/ctx.witness). ----
        screen_pairwise = nothing; screen_witness = nothing
        t_pairwise = NaN; t_witness = NaN
        if build_screen
            ctx_min = (U = ctx_qmc.U, D = ctx_qmc.D, D_dest = ctx_qmc.D_dest)
            t_pairwise = @elapsed screen_pairwise = precompute_pairwise_M(ctx_min)
            t_witness = @elapsed screen_witness = build_extreme_draw_witness(ctx_min)
        end
        ctx0 = merge(ctx_qmc, (pairwise = screen_pairwise, witness = screen_witness,
                                screen_setup_wall = (pairwise = t_pairwise, witness = t_witness)))
        timing = (uniform_and_transform = t_gen, ctx_build = t_ctx, pairwise = t_pairwise, witness = t_witness)
    end

    meta = log_draw_meta ? draw_design_meta(draw_design, draw_seed, ctx0.D, W, ctx0.U; timing = timing) : nothing
    return merge(ctx0, (draw_design = draw_design, draw_seed = draw_seed, draw_meta = meta))
end
