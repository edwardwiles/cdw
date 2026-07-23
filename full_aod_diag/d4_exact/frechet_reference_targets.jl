# ============================================================================
# Benchmark F* Fréchet target construction (task brief §5). See
# docs/FIXED_FRECHET_MARGINALS_MATH_NOTE_2026-07-23.md §2/§5 for the full
# derivation: because ctx.U is i.i.d. Exp(1) BY CONSTRUCTION (genExpRands!,
# prepare_cc/genRands.jl) and every CM architecture operates directly on U,
# the CDF-feature benchmark target is the EXACT (not estimated, not
# numerically approximated) Exp(1) quantile/CDF -- no θ*/σ dependence. θ*/σ
# are still retrieved from the canonical context (not hardcoded, per task
# brief §5) and stored for provenance/fingerprinting, and are used by the
# (currently dormant, see math note §4) cumulative-power target formula.
# ============================================================================

using SpecialFunctions: gamma_inc, loggamma
using SHA: sha256, bytes2hex

"""
    FrechetReferenceTargets

Immutable benchmark-target bundle (task brief §5's required provenance
fields). `thresholds`/`targets` are the ACTIVE CDF-feature family
(`u_l* = -log(1-p_l)`, `t_l* = p_l`); `power_targets` is the (dormant, see
math note §4) cumulative-power companion, always computed (cheap, closed
form) for architectural completeness but not consumed by any production
moment path in this release.
"""
struct FrechetReferenceTargets
    family::Symbol                  # :frechet
    theta_star::Float64              # 1/ctx.μHat
    scale::Float64                   # 1.0 (production normalization)
    sigma::Float64                   # ctx.σ
    probs::Vector{Float64}           # p_l, L entries
    thresholds::Vector{Float64}      # u_l* = -log(1-p_l), U-space (Exp(1)) analytic quantiles
    targets::Vector{Float64}         # t_l* = p_l, CDF-feature targets
    power_targets::Vector{Float64}   # t_power_l* = γ(pw+1, u_l*), pw = (1-σ)/θ*  (dormant block)
    target_sha256::String            # sha256 of `targets` (CDF block only -- the active block)
    feature_layout_version::Int
end

const FRECHET_FEATURE_LAYOUT_VERSION = 1   # bump if the column layout (§7 of the math note) ever changes

"sha256 hex digest of a Float64 vector's raw bits -- deterministic, used for checkpoint/context fingerprinting (task brief §5/§9)."
function _float_vector_sha256(v::AbstractVector{Float64})
    return bytes2hex(sha256(reinterpret(UInt8, collect(v))))
end

"""
    frechet_cdf_quantile(p) -> Float64

Analytic Exp(1) quantile: `F_Exp^{-1}(p) = -log(1-p)`. This IS the
`(F*)^{-1}(p)` the task brief asks for, expressed in the codebase's native
`U`-space (see math note §2 for why this is exact, not an approximation).
"""
frechet_cdf_quantile(p::Float64) = -log1p(-p)

"""
    frechet_power_target(pw, u) -> Float64

`E_{Exp(1)}[U^pw * 1{U<=u}] = ∫_0^u t^pw e^{-t} dt = γ(pw+1, u)` (lower
incomplete gamma, unnormalized), via `SpecialFunctions.gamma_inc`'s
regularized lower `P(a,x)` times `Γ(a)`. Requires `pw > -1` (else the
integral diverges at 0); the production `pw = (1-σ)/θ*` with `σ>1,θ*>0`
gives `pw<0` in general, so this is asserted rather than silently producing
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
1/ctx.μHat`, `sigma = ctx.σ`, NOT hardcoded) and `cfg`'s resolved grid
probabilities. Deterministic and cheap (`O(L)`); no draws, no optimization.
"""
function build_frechet_reference_targets(ctx, cfg::CMFrechetConfig; L::Int = cfg.cm.cm_grid_size)
    probs = cm_resolve_probs_for_L(cfg.cm, L)
    theta_star = 1.0 / ctx.μHat
    sigma = ctx.σ
    thresholds = frechet_cdf_quantile.(probs)
    targets = copy(probs)   # t_l* = p_l exactly (math note §2/§5)
    pw = (1.0 - sigma) / theta_star
    power_targets = [frechet_power_target(pw, u) for u in thresholds]
    return FrechetReferenceTargets(:frechet, theta_star, 1.0, sigma, collect(probs), thresholds, targets,
                                    power_targets, _float_vector_sha256(targets), FRECHET_FEATURE_LAYOUT_VERSION)
end
