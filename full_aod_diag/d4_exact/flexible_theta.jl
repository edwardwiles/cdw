# ============================================================================
# Flexible-theta (trade elasticity as an outer parameter) production port, 2026-07-25.
#
# Ported from experiment/fullA-theta-aspace-reparam-2026-07-25 (flexible_theta.jl +
# flexible_theta_production.jl there) onto CURRENT production's rectangular
# (D origins x D_dest active destinations, post-omit-ROW) context shape -- NOT copied
# verbatim: the experimental source assumed a square D x D layout throughout (it predates
# the omit-ROW-destination production release). Every D^2/D^2-1 length here is generalized
# to D*Ddest/D*Ddest-1, and every `reshape(...,(D,D))` to the active-destination-aware
# convention `cc_algo/active_layout.jl` and `context_real_d20.jl::d20_real_setup` already
# use (Ddest = ctx.D_dest under :exclude_row, Ddest = ctx.D under :all_legacy).
#
# See docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md for the full
# derivation this file implements.
#
# DESIGN (unchanged from the experimental prototype, cross-checked live rather than
# assumed): mu_code is decoded from eta_theta ENTIRELY UPSTREAM of every validated core
# function -- evaluate_fullA/evaluate_fullA_screened_ranged (oracle.jl/fast_range_screen.jl),
# EK_moments_gammanorm_directgp! (moments_gammanorm.jl), build_compressed_factual
# (compressed_moments.jl) -- NONE of those are modified by this file. The only production
# file with a (small, additive, backward-compatible: new `μ` keyword, default preserves
# byte-identical fixed-mode behavior) edit is gravity_elimination.jl.
#
# OUTER VECTOR LAYOUT: w_ext = [eta_theta; gp; zfree] in z-space (matches current
# production's own [gp; zfree] convention with exactly one new leading coordinate), or
# w_ext_a = [eta_theta; gp; a_nonpivot] in a-space (flexible_theta_aspace_production.jl).
# `eta_theta = log(theta_trade)`, `theta_trade = exp(eta_theta)`, `mu_code = 1/theta_trade`.
# ============================================================================

isdefined(Main, :d20_real_setup) || isdefined(Main, :d4_exact_setup) ||
    error("flexible_theta.jl requires context_real_d20.jl or context.jl to already be included first.")
isdefined(Main, :evaluate_fullA) ||
    error("flexible_theta.jl requires oracle.jl to already be included first (needs evaluate_fullA, FullAEvalKey, context_fingerprint, reject_point).")

"D_dest for ctx -- Ddest==ctx.D unless row_idx excludes ROW (matches gravity_elimination.jl's _ctx_ddest)."
_flex_ddest(ctx) = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

"""
    make_flexible_theta(ctx; theta_lo, theta_hi) -> ctx_flex

Generic converter: works on ANY ctx with the shape `d4_exact_setup()` / `context_scaled.jl::
d_exact_setup_scaled` / `context_real_d20.jl::d20_real_setup` all share (θ0_up, θ_lo, θ_hi,
free_idx, fixed_idx, fixed_vals, m, μHat, l_full) -- moves the mu slot (absolute theta_full
index 1) from `fixed_idx` to `free_idx`, replacing its degenerate `[theta_star,theta_star]`
box (in eta_theta units: a single point) with `[log(theta_lo), log(theta_hi)]`. `theta_star`
(the gravity-calibrated value, `1/ctx.μHat`) is preserved as `ctx_flex.theta_star`;
`θ0_up[1]` is rewritten to `log(theta_star)` so a caller pinning `theta_lo=theta_hi=theta_star`
reproduces the exact fixed-mode calibration point (S3's pinned-equivalence assertion).

Every OTHER field of `ctx` (γ, U, obj, τ, q_tilde, N_obs, Aod_offset, D_dest, row_idx,
destination_sample, active_origins/active_destinations, ...) is carried through unchanged --
this is additive, not a re-derivation. Used for BOTH D=4 gates (via
`d4_exact_setup_flexible_theta`/`context_scaled.jl`-built ctx) and D=20 gates
(`make_flexible_theta(d20_real_setup(...); ...)`), so there is exactly one implementation of
the free/fixed-index surgery to keep correct.

Sets `ctx_flex.trade_elasticity_mode = :flexible` and `ctx_flex.A_coordinate_mode` (caller-
supplied, default `:z_space` -- `flexible_theta_aspace_production.jl`'s `make_flexible_theta_aspace`
wrapper sets it to `:theta_decoupled_aspace`) per task §2's explicit configuration surface.
"""
function make_flexible_theta(ctx; theta_lo::Float64, theta_hi::Float64, A_coordinate_mode::Symbol = :z_space)
    theta_star = 1.0 / ctx.μHat
    (theta_lo <= theta_star <= theta_hi) || error("make_flexible_theta: theta_star=$theta_star (from gravity calibration) is not inside the configured box [$theta_lo, $theta_hi] -- widen the box or check theta_star.")

    θ0_up = copy(ctx.θ0_up)
    θ0_up[1] = log(theta_star)
    θ_lo = copy(ctx.θ_lo); θ_hi = copy(ctx.θ_hi)
    θ_lo[1] = log(theta_lo); θ_hi[1] = log(theta_hi)

    free_idx = vcat(1, ctx.free_idx)
    fixed_idx = setdiff(ctx.fixed_idx, 1)
    fixed_vals = θ0_up[fixed_idx]
    m = CS.FreeParamMap(ctx.l_full, free_idx, fixed_idx, fixed_vals)
    @assert CS.n_free(m) == CS.n_free(ctx.m) + 1 "make_flexible_theta: expected exactly one more free coordinate than the fixed-theta ctx"

    return merge(ctx, (θ0_up = θ0_up, θ_lo = θ_lo, θ_hi = θ_hi,
        free_idx = free_idx, fixed_idx = fixed_idx, fixed_vals = fixed_vals, m = m,
        trade_elasticity_mode = :flexible, A_coordinate_mode = A_coordinate_mode, theta_star = theta_star,
        theta_lo = theta_lo, theta_hi = theta_hi,
        eta_theta_star = log(theta_star), theta_param_version = 1))
end

"""
    decode_theta_full(w_ext, ctx) -> x_free_decoded

`w_ext[1]` (eta_theta, for a ctx built via `make_flexible_theta`) decoded to `mu_code`, returned
as position 1 of a NEW vector (rest copied unchanged). In `:fixed` mode (ctx lacks
`trade_elasticity_mode`, or it is `:fixed`, i.e. every existing production ctx today) this is
the identity map -- callers built on top of this see BYTE-IDENTICAL behavior to calling the
production functions directly (task §12's fixed-mode invariance requirement).
"""
function decode_theta_full(w_ext::AbstractVector{Float64}, ctx)
    if !hasproperty(ctx, :trade_elasticity_mode) || ctx.trade_elasticity_mode == :fixed
        return collect(w_ext)
    end
    ctx.trade_elasticity_mode == :flexible || error("decode_theta_full: unknown trade_elasticity_mode=$(ctx.trade_elasticity_mode)")
    η_θ = w_ext[1]
    # Box check in eta_theta (log) space against ctx.θ_lo[1]/ctx.θ_hi[1] -- the SAME log-space
    # bounds KNITRO's own box constraint enforces -- with a tiny relative tolerance for
    # floating-point roundoff (a pinned box theta_lo=theta_hi=theta_star can fail an exact
    # linear-space comparison by 1 ULP after the log/exp round-trip; checked live on the
    # original z-space port).
    tol = 1e-9 * max(1.0, abs(ctx.θ_lo[1]), abs(ctx.θ_hi[1]))
    (ctx.θ_lo[1] - tol <= η_θ <= ctx.θ_hi[1] + tol) || error("decode_theta_full: eta_theta=$η_θ (theta_trade=$(exp(η_θ))) outside configured box [$(ctx.θ_lo[1]), $(ctx.θ_hi[1])] (log space; linear box [$(ctx.theta_lo), $(ctx.theta_hi)])")
    xd = collect(w_ext)
    xd[1] = exp(-η_θ)   # = 1/theta_trade = mu_code
    return xd
end

"""
    freeze_theta_ctx(ctx, mu_frozen) -> ctx_frozen

Builds a ctx whose `FreeParamMap` (`m`/`free_idx`/`fixed_idx`/`fixed_vals`) is shaped EXACTLY
like the original fixed-theta ctx (mu fixed, `[gp;Aod(D*Ddest)]` free, `n_free=D*Ddest+1`) but
with `fixed_vals[1]` set to the CURRENT `mu_frozen` (which may differ from this ctx's own
calibration mu) -- everything else (γ, U, obj, τ, q_tilde, ...) carried through unchanged.

This is task §8's "the existing production C+/buffered A-gradient kernel for the gp/A block AT
FIXED THETA": several production gradient kernels (`composite_gradient_at_fast_buffered`/
`composite_gradient_at_Cplus`/`build_lfix_base_cache`/`lfix_incremental.jl`) call
`CS.reconstruct_full(x_free0, ctx.m)` INTERNALLY and hard-assume the fixed-mode `x_free` shape
(`[gp;Aod(D*Ddest)]`, length `D*Ddest+1`) -- they are NOT `ctx.m`-shape-generic. Feeding them a
flexible `ctx` (whose `ctx.m` expects `D*Ddest+2` free entries, mu included) together with a
`D*Ddest+1`-length reduced vector silently MISALIGNS the reconstruction (this exact bug class
was caught live during the original z-space port: a `DomainError` from `(-Inf)^y` traced to
`aod_pow_cell` reading a garbage `μ=θ_full[1]`, because `ctx.m` there expected one more free
coordinate than the vector actually supplied). Freezing theta via a fresh `ctx.m` (not merely
freezing the scalar) is what makes this reuse correct. Cheap: one `CS.FreeParamMap`
construction, no re-solve, no re-derivation of the pivot/nullspace map (unrelated to this).
"""
function freeze_theta_ctx(ctx, mu_frozen::Float64)
    isempty(ctx.free_idx) && throw(DimensionMismatch("freeze_theta_ctx: ctx.free_idx is empty, nothing to freeze"))
    ctx.free_idx[1] == 1 || throw(DimensionMismatch(
        "freeze_theta_ctx: expected mu at ctx.free_idx[1]==1 (flexible-theta ctx convention), got " *
        "ctx.free_idx[1]=$(ctx.free_idx[1]) -- this ctx is not shaped as make_flexible_theta produces"))
    base_free_idx = ctx.free_idx[2:end]            # drop position-1 (mu) -> [gp;Aod(D*Ddest)] positions
    base_fixed_idx = vcat(1, ctx.fixed_idx)         # mu index 1 restored to fixed
    base_fixed_vals = vcat(mu_frozen, ctx.θ0_up[ctx.fixed_idx])
    m_frozen = CS.FreeParamMap(ctx.l_full, base_free_idx, base_fixed_idx, base_fixed_vals)
    return merge(ctx, (m = m_frozen, free_idx = base_free_idx, fixed_idx = base_fixed_idx, fixed_vals = base_fixed_vals,
                        trade_elasticity_mode = :fixed))
end

"""
    context_fingerprint_flexible_theta(ctx) -> String

Extends the base `context_fingerprint(ctx)` (oracle.jl, unmodified) with a block hashing
`(trade_elasticity_mode, A_coordinate_mode, theta_lo, theta_hi, theta_param_version, sigma,
destination_sample)` -- the flexible-theta BOUNDS/mode/layout config, never the current theta
value (that belongs in the exact-point cache key, e.g. `FullAEvalKey.x_free`, which already
contains mu via `xf[1]` once decoded -- see cache_fingerprint_extensions.jl / S10). Two contexts
differing only in theta bounds, coordinate mode, or destination sample must fingerprint
differently; two evaluations at the SAME context but different theta must fingerprint
IDENTICALLY (only x_free differs).
"""
const FLEXIBLE_THETA_FINGERPRINT_SCHEMA = 1
function context_fingerprint_flexible_theta(ctx)::String
    base = context_fingerprint(ctx)   # oracle.jl, unmodified, unmoved schema
    buf = IOBuffer()
    write(buf, base)
    write(buf, htol(Int64(FLEXIBLE_THETA_FINGERPRINT_SCHEMA)))
    mode = hasproperty(ctx, :trade_elasticity_mode) ? ctx.trade_elasticity_mode : :fixed
    write(buf, string(mode))
    if mode == :flexible
        write(buf, string(hasproperty(ctx, :A_coordinate_mode) ? ctx.A_coordinate_mode : :z_space))
        write(buf, htol(reinterpret(UInt64, Float64(ctx.theta_lo))))
        write(buf, htol(reinterpret(UInt64, Float64(ctx.theta_hi))))
        write(buf, htol(Int64(ctx.theta_param_version)))
    end
    write(buf, htol(reinterpret(UInt64, Float64(ctx.σ))))
    write(buf, string(hasproperty(ctx, :destination_sample) ? ctx.destination_sample : :all_legacy))
    write(buf, "moment_layout_v1")   # bump this literal if the compressed-moment layout ever changes shape
    return bytes2hex(sha256(take!(buf)))
end

# ============================================================================
# z-space (OLD parametrization) decode/eval/theta-secant -- rectangularized port of the
# original experimental flexible_theta_production.jl. NOT wired into any production driver
# (fixed-theta production stays z-space via the untouched build_pivot_elimination/pivot_expand;
# flexible-theta production uses ONLY the a-space path, flexible_theta_aspace_production.jl).
# Kept here, ported faithfully, SOLELY to serve as the "old parametrization" comparison arm
# required by task §13's D=4 correctness gates (economic equivalence at theta-star, off-theta
# feasibility contrast) and task §15's practical-value comparison baseline -- i.e. this is test/
# comparison scaffolding, not a second production code path.
# ============================================================================

"""
    decode_and_expand_flexible(w_ext, ctx) -> NamedTuple

`w_ext = [eta_theta; gp; z_nonpivot]` (length `D*Ddest+1`, matching the real driver's `w =
[gp; zfree_reduced]` plus exactly one new leading coordinate).
"""
function decode_and_expand_flexible(w_ext::AbstractVector{Float64}, ctx)
    (hasproperty(ctx, :trade_elasticity_mode) && ctx.trade_elasticity_mode == :flexible) ||
        error("decode_and_expand_flexible: ctx is not in flexible-theta mode -- call make_flexible_theta first")
    eta_theta = w_ext[1]
    tol = 1e-9 * max(1.0, abs(ctx.θ_lo[1]), abs(ctx.θ_hi[1]))
    if !(ctx.θ_lo[1] - tol <= eta_theta <= ctx.θ_hi[1] + tol)
        reject_point(eta_theta, "decode_and_expand_flexible: eta_theta=$eta_theta (theta=$(exp(eta_theta))) outside " *
            "configured box [$(ctx.θ_lo[1]), $(ctx.θ_hi[1])] (log space; linear box [$(ctx.theta_lo), $(ctx.theta_hi)])")
    end
    theta = exp(eta_theta)
    mu = 1.0 / theta
    gp = w_ext[2]
    z_nonpivot = @view w_ext[3:end]
    D = ctx.D; Ddest = _flex_ddest(ctx)
    length(z_nonpivot) == D * Ddest - 1 ||
        throw(DimensionMismatch("decode_and_expand_flexible: w_ext has length $(length(w_ext)), expected D*Ddest+1=$(D*Ddest+1)"))
    pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / ctx.theta_lo, mu_probe2 = 1.0 / ctx.theta_hi)
    Aod_levels = vec(exp.(pivot_expand_cheap(z_nonpivot, pgc, mu)))
    xf = vcat(mu, gp, Aod_levels)
    return (xf = xf, mu = mu, theta = theta, eta_theta = eta_theta, gp = gp, pgc = pgc,
            z_nonpivot = collect(z_nonpivot))
end

"Inverse direction: full (eta_theta, gp, log-A matrix D x Ddest) -> the reduced z-space outer vector."
function reduce_to_w_ext(eta_theta::Float64, gp::Float64, logA_full::AbstractMatrix{Float64}, pgc::PivotGravityElimCache)
    return vcat(eta_theta, gp, pivot_reduce_cheap(logA_full, pgc))
end

"z-space screened_eval_flexible: delegates to the existing production screened_eval, matching screened_eval_flexible_A's shape."
function screened_eval_flexible(w_ext::AbstractVector{Float64}, ctx, rsc::RangedScreenContext, sc::ScreenCounters,
        n_eval_ref::Ref{Int}, pgc_unused = nothing; warm::Bool = true, bank::Union{Nothing,DualBank} = nothing,
        exact_cache::Union{Nothing,SafeExactCache,CrossDeltaExactCache} = nothing,
        neg_cache::Union{Nothing,SafeNegativeCache} = nothing)
    d = decode_and_expand_flexible(w_ext, ctx)
    result, screen_meta = screened_eval(d.xf, ctx, rsc, sc, n_eval_ref; warm = warm, bank = bank,
        zfree = d.z_nonpivot, exact_cache = exact_cache, neg_cache = neg_cache)
    return result, screen_meta, d
end

"z-space fixed-dual theta secant kernel (OLD parametrization; holds z_nonpivot fixed, not a_nonpivot)."
function theta_fixed_dual_delta_pivot(w_ext::AbstractVector{Float64}, inner_x_fixed::AbstractVector{Float64}, ctx)
    d = decode_and_expand_flexible(w_ext, ctx)
    obj = ctx.obj
    θ_full = CS.reconstruct_full(d.xf, ctx.m)
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_full, obj.U, obj)
    obj.H[:, 2] .= 1.0
    fval = obj(inner_x_fixed)
    return -fval
end
