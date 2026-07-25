# ============================================================================
# Benchmark F* Fréchet target construction (port-prep 2026-07-24). See
# docs/FIXED_FRECHET_FULL_SPEC_MATH_AND_TARGETS_2026-07-24.md §2/§3 for the
# full derivation: because ctx.U is i.i.d. Exp(1) BY CONSTRUCTION (genExpRands!,
# prepare_cc/genRands.jl) and every CM architecture operates directly on U,
# the CDF-feature benchmark target is the EXACT (not estimated, not
# numerically approximated) Exp(1) quantile/CDF -- no θ*/σ dependence beyond
# the grid itself. θ*/σ ARE live (`1/ctx.μHat`, `ctx.σ`), NEVER hard-coded
# (task brief §3 -- do not repeat the draft's historical θ*=6.8), and are
# used by the truncated-power target formula, which -- unlike the pre-
# omit-ROW reconciliation branch this file descends from -- is now ACTIVE by
# default (`frechet_feature_set=:cdf_power`), not dormant.
#
# Dimension note (task brief §2): `ctx.D` here is the ORIGIN count (always
# 20, invariant across destination_sample) -- see
# docs/FIXED_FRECHET_FULL_SPEC_MATH_AND_TARGETS_2026-07-24.md §4. This file
# never reads `ctx.D_dest`.
# ============================================================================

using SpecialFunctions: gamma_inc, loggamma
using SHA: sha256, bytes2hex

"""
    FrechetReferenceTargets

Immutable benchmark-target bundle (task brief §5's/§10's required provenance
fields). `thresholds`/`targets` are the CDF-feature family (`u_l* =
-log(1-p_l)`, `t_l* = p_l`); `power_targets` is the truncated-power (eq.38)
companion -- ALWAYS computed (cheap, closed form) and, under
`frechet_feature_set=:cdf_power`, consumed by the moment/Hessian
construction in `cm_frechet_bases.jl`.
"""
struct FrechetReferenceTargets
    family::Symbol                  # :frechet
    theta_star::Float64              # 1/ctx.μHat -- LIVE, never hard-coded
    scale::Float64                   # 1.0 (production normalization)
    sigma::Float64                   # ctx.σ
    D::Int                           # origin count (ctx.D), for provenance/fingerprint only
    probs::Vector{Float64}           # p_l, L entries
    thresholds::Vector{Float64}      # u_l* = -log(1-p_l), U-space (Exp(1)) analytic quantiles
    targets::Vector{Float64}         # t_l* = p_l, CDF-feature targets
    power_targets::Vector{Float64}   # t_power_l* = γ(pw+1, u_l*), pw = (1-σ)/θ*
    target_sha256::String            # sha256 of `targets` (CDF block)
    power_target_sha256::String      # sha256 of `power_targets` (power block, port-prep addition)
    feature_layout_version::Int
end

const FRECHET_FEATURE_LAYOUT_VERSION = 2   # bumped from the pre-omit-ROW archive's v1: power block is
                                            # now consumed by default, and the layout carries a D field
                                            # for the origin/destination dimension-fingerprint distinction.

"sha256 hex digest of a Float64 vector's raw bits -- deterministic, used for checkpoint/context fingerprinting."
function _float_vector_sha256(v::AbstractVector{Float64})
    return bytes2hex(sha256(reinterpret(UInt8, collect(v))))
end

"""
    frechet_cdf_quantile(p) -> Float64

Analytic Exp(1) quantile: `F_Exp^{-1}(p) = -log(1-p)`. This IS the
`(F*)^{-1}(p)` the task brief asks for, expressed in the codebase's native
`U`-space.
"""
frechet_cdf_quantile(p::Float64) = -log1p(-p)

"""
    frechet_power_target(pw, u) -> Float64

`E_{Exp(1)}[U^pw * 1{U<=u}] = ∫_0^u t^pw e^{-t} dt = γ(pw+1, u)` (lower
incomplete gamma, unnormalized), via `SpecialFunctions.gamma_inc`'s
regularized lower `P(a,x)` times `Γ(a)`. Requires `pw > -1` (else the
integral diverges at 0); production `pw = (1-σ)/θ*` with `σ>1,θ*>0` gives
`pw<0` in general, so this is asserted rather than silently producing
NaN/Inf.
"""
function frechet_power_target(pw::Float64, u::Float64)
    a = pw + 1.0
    a > 0 || error("frechet_power_target: pw+1=$a must be > 0 (integral diverges at 0 for a<=0), pw=$pw")
    u >= 0 || error("frechet_power_target: u=$u must be >= 0")
    P, _ = gamma_inc(a, u, 0)   # (P, Q) regularized lower/upper incomplete gamma
    return P * exp(loggamma(a))
end

"""
    build_frechet_reference_targets(ctx, cfg::CMFrechetConfig; L=cfg.cm.cm_grid_size) -> FrechetReferenceTargets

Builds the benchmark target bundle from the canonical `ctx` (`theta_star =
1/ctx.μHat`, `sigma = ctx.σ`, NOT hardcoded, live under whatever
`destination_sample` `ctx` was built with) and `cfg`'s resolved grid
probabilities. Deterministic and cheap (`O(L)`); no draws, no optimization.
Asserts `size(ctx.U, 2) == ctx.D` (task brief §2's dimension-safety
requirement) so a future accidental `ctx.D_dest` substitution upstream fails
loudly here rather than silently building a wrong-sized target bundle.
"""
function build_frechet_reference_targets(ctx, cfg::CMFrechetConfig; L::Int = cfg.cm.cm_grid_size)
    @assert size(ctx.U, 2) == ctx.D "build_frechet_reference_targets: ctx.U has $(size(ctx.U,2)) columns, " *
        "expected ctx.D=$(ctx.D) (origin count) -- the fixed-Fréchet marginal restriction is an " *
        "origin-indexed restriction and must never be sized off ctx.D_dest (destination count, " *
        "$(hasproperty(ctx, :D_dest) ? ctx.D_dest : "n/a")); see docs/FIXED_FRECHET_FULL_SPEC_MATH_AND_TARGETS_2026-07-24.md §4."
    probs = cm_resolve_probs_for_L(cfg.cm, L)
    theta_star = 1.0 / ctx.μHat
    sigma = ctx.σ
    thresholds = frechet_cdf_quantile.(probs)
    targets = copy(probs)   # t_l* = p_l exactly
    pw = (1.0 - sigma) / theta_star
    power_targets = [frechet_power_target(pw, u) for u in thresholds]
    return FrechetReferenceTargets(:frechet, theta_star, 1.0, sigma, ctx.D, collect(probs), thresholds, targets,
                                    power_targets, _float_vector_sha256(targets), _float_vector_sha256(power_targets),
                                    FRECHET_FEATURE_LAYOUT_VERSION)
end

"""
    report_frechet_targets(ctx, cfg::CMFrechetConfig, targets::FrechetReferenceTargets)

Task brief §3's required startup report: active destination sample, live
θ*/μ̂/σ, D/D_dest/L, feature set, basis, analytic target fingerprints.
Explicitly states this run does NOT claim to reproduce any historical
figure (e.g. the paper draft's own reported θ*=6.8) -- only the same
restriction family at the ACTIVE calibration.
"""
function report_frechet_targets(ctx, cfg::CMFrechetConfig, targets::FrechetReferenceTargets)
    ds = hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :unknown
    dd = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    println("=== Fixed-Fréchet reference targets ===")
    println("  destination_sample   = $ds")
    println("  D (origins)          = $(ctx.D)")
    println("  D_dest (destinations)= $dd")
    println("  L (grid size)        = $(length(targets.probs))")
    println("  theta_star (=1/muHat)= $(targets.theta_star)   [LIVE, not the draft's historical 6.8]")
    println("  muHat                = $(ctx.μHat)")
    println("  sigma                = $(targets.sigma)")
    println("  feature_set          = $(cfg.frechet_feature_set)")
    println("  basis                = $(cfg.frechet_basis)")
    println("  contrasts            = $(cfg.cm.contrasts)")
    println("  target_sha256 (CDF)  = $(targets.target_sha256)")
    println("  target_sha256 (POW)  = $(targets.power_target_sha256)")
    println("  feature_layout_ver   = $(targets.feature_layout_version)")
    println("  NOTE: this run implements the same restriction FAMILY as the paper draft, evaluated")
    println("        at this codebase's own live calibration -- it does not reproduce any historical")
    println("        figure/theta* value.")
    flush(stdout)
    return nothing
end
