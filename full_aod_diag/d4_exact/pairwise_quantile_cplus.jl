# ================================================================================================
# Pairwise-quantile-independence family onto Backend C+ (the FACTORIZED price representation),
# 2026-08-11.
#
# WHY THIS FILE EXISTS. This family's outer gradient was built on the DENSE economic path
# (`build_lfix_base_cache`, lfix_incremental.jl), which materializes `price0` and `pTsigma0` as
# `W x D x Ddest` tensors -- 304 MB EACH at W=100,000/D=20/Ddest=19 -- and refills them on every
# gradient call. Backend C+ (`lfix_factorized.jl`) never materializes either tensor at all: it
# stores the log-space factorization `S_{sod} = logCC_{od} + mulU_{so}` in O(W*D) + O(D^2) and
# reconstructs any cell in O(1) on demand. Four of the five production families already have an
# adapter onto it (`cm_originzc_cplus.jl`, `cm_meanzc_cplus.jl`, `cm_frechet_cplus.jl`,
# `lfix_cm_cplus.jl`); this family did not, purely because it was wired to the dense path without
# checking whether a faster shared one existed. This file closes that gap.
#
# It is also where the mu-invariance shows up. `mu = theta_full[1]` is NOT a free outer coordinate
# (`context_real_d20.jl`: `free_idx = vcat(3 + Dact, Aod_offset+1:...)`, i.e. gp and the A_od block
# only), so `U^(-mu)` is a CAMPAIGN CONSTANT. The dense path recomputes it 380 times per gradient
# call; the factorized path caches `log(U)` for the whole campaign in
# `canonical_price_precompute_workspace.jl` and only rescales it. (`UPow`/`UsigmaPow` are still
# recomputed per call there -- a further mu-guard would remove those two W x D power arrays too,
# but that is SHARED code and is flagged, not changed here.)
#
# DECOMPOSITION ARGUMENT -- identical to the one `cm_originzc_cplus.jl` / `lfix_cm_cplus.jl` prove
# for their own arms, and it carries over unchanged. The augmented base-point dual scalar
#     q_s(theta) = -zeta* - lambda_E*'E_s(theta) - lambda_R*'G^R_s
# has a term `lambda_R*'G^R_s` that is CONSTANT across every outer (g, A_od) probe: this family's
# restriction rows are indicator functions of `bin[w,o]`, and under version B the bins are fixed for
# the whole campaign, so they do not depend on theta at all. Backend C+'s
# `build_lfix_base_cache_C!` / `composite_gradient_at_Cplus_from_cache` require only that, so
# `composite_gradient_at_Cplus_from_cache` is reused here COMPLETELY UNMODIFIED.
#
# The constancy of that term is about the DERIVATIVE and does NOT excuse omitting it from `q0`.
# `q0` is the per-draw LEVEL the economic block linearizes around, and dropping the restriction's
# contribution linearizes about the wrong point -- a real bug this family already made once (memory
# `feedback-q0-restriction-fold-is-a-level-not-a-derivative`). The fold and its EXACT cross-check
# against the independently recomputed `r` are carried over here verbatim; `LFixBaseCacheC` carries
# its own `q0` field, so nothing about the check changes.
#
# PURELY ADDITIVE: does not modify lfix_factorized.jl, lfix_factorized_workspace.jl,
# lfix_cm_cplus.jl, pairwise_quantile_outer_production.jl, or any shared economic file. Every
# function here is new, and the dense path remains available as the reference the gate compares
# against (test_pairwise_quantile_cplus_gate.jl).
# ================================================================================================

isdefined(Main, :with_q0_C) || include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
isdefined(Main, :build_lfix_base_cache_C!) || include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))

"""
    build_lfix_base_cache_pairwise_quantile_C!(ws, x_free0, ctx_cm, base; verify=nothing,
                                               q0_check_tol=1e-8, validate_dense=false) -> LFixBaseCacheC

Backend C+ twin of `build_lfix_base_cache_pairwise_quantile`
(pairwise_quantile_outer_production.jl). Calls `build_lfix_base_cache_C!` UNCHANGED, then folds in
THIS restriction's own `G_R*lambda_R` contribution via `with_q0_C` -- structurally identical to
`build_lfix_base_cache_originzc_C!`'s two lines, with this family's own fold in place of
`originzc_fixed_contribution`.

SIGN: `pairwise_quantile_forward!` ACCUMULATES `-G_R*lambda_R` into its accumulator (its own
`arg0[w] -= Rw`), so starting from zeros it yields exactly the term `q0` is missing and it is
ADDED. (origin-ZC subtracts, because `originzc_fixed_contribution` returns `+Z*nu`.) This is the
same convention and the same operator call the dense path uses, deliberately not re-derived, so the
two cannot drift.

`verify`: when supplied, the corrected `q0` is cross-checked against the verifier's independently
recomputed `r`, to floating point. Both are `r = -zeta - E*lambda_E - G_R*lambda_R` by two
different routes, so a mismatch is a hard error, not a warning.
"""
function build_lfix_base_cache_pairwise_quantile_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector,
        ctx_cm, base::BaseDualState; verify = nothing, q0_check_tol::Float64 = 1e-8,
        validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    op = ctx_cm.pq_op
    ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
    lam_M, lam_P = reshape_pq_duals(base.λstar, op, ncore1)
    contrib = zeros(op.W)
    pairwise_quantile_forward!(contrib, lam_M, lam_P, op, ctx_cm.pq_mass_state)   # = -G_R*lambda_R
    q0_new = cache0.q0 .+ contrib
    if verify !== nothing && hasproperty(verify, :r_current)
        err = maximum(abs, q0_new .- verify.r_current)
        err <= q0_check_tol ||
            error("build_lfix_base_cache_pairwise_quantile_C!: corrected q0 disagrees with the " *
                  "independently recomputed r by max|diff|=" * string(err) * " (tol=" *
                  string(q0_check_tol) * "). Same quantity by two routes -- a mismatch means the " *
                  "restriction fold or the factorized cache is wrong, and any economic gradient " *
                  "built on it would be silently wrong.")
    end
    return with_q0_C(cache0, q0_new)
end

"""
    pairwise_quantile_production_gradient_cplus(x_free0, raw_masses, pcx, ctx, pe, pool, ws;
                                                base=nothing, verify=nothing, timers=nothing,
                                                kwargs...) -> (g_ext, meta)

`:cplus`-backend analog of `pairwise_quantile_production_gradient`. Returns the SAME
`vcat(g_econ, g_mass)` vector of length `D*Ddest + n_raw(layout)`, in the same coordinate order --
only the economic half is computed through the factorized representation instead of the dense one.

The restriction half is untouched and already free: `pairwise_quantile_mass_gradient_vec` is the
exact closed-form envelope derivative and measured 0.0001 s at W=100,000.

`pool`/`ws` are the caller-owned persistent `GradWorkspacePool`/`LFixFactorizedWorkspace` -- built
ONCE per run (`pairwise_quantile_cplus_workspaces`), never per call, which is the entire point of
the backend.
"""
function pairwise_quantile_production_gradient_cplus(x_free0::AbstractVector,
        raw_masses::AbstractVector{Float64}, pcx, ctx, pe, pool::GradWorkspacePool,
        ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing, verify = nothing,
        timers::Union{Nothing,PQGradTimers} = nothing, kwargs...)
    t_enter = time()
    if base === nothing || verify === nothing
        t0 = time()
        base, verify = archPQ_verified_state(x_free0, raw_masses, pcx.ctx_cm)
        timers === nothing || (timers.t_solve += time() - t0)
    end
    # Same mutable-ambient-state hazard as the dense path: pq_mass_state is per-outer-point and some
    # intervening solve may have moved it. See ensure_pq_masses!.
    ensure_pq_masses!(pcx.ctx_cm, raw_masses)
    t0 = time()
    cache = build_lfix_base_cache_pairwise_quantile_C!(ws, x_free0, pcx.ctx_cm, base; verify = verify)
    timers === nothing || (timers.t_cache += time() - t0)
    t0 = time()
    g_econ, meta = composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache;
        base = base, kwargs...)
    timers === nothing || (timers.t_econ += time() - t0)
    t0 = time()
    g_mass = pairwise_quantile_mass_gradient_vec(base, verify, pcx.ctx_cm, raw_masses)
    timers === nothing || (timers.t_mass += time() - t0)
    timers === nothing || (timers.t_total += time() - t_enter)
    return vcat(g_econ, g_mass), meta
end

"""
    pairwise_quantile_cplus_workspaces(ctx) -> (pool, ws)

The two persistent Backend C+ workspaces this family needs, sized off `ctx`. Build ONCE per run and
hold them for its lifetime -- rebuilding per gradient call would reintroduce exactly the allocation
this backend exists to remove.
"""
function pairwise_quantile_cplus_workspaces(ctx)
    D = ctx.D
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    W = size(ctx.obj.U, 1)
    return (build_grad_workspace_pool(W), build_lfix_factorized_workspace(D, Ddest, W))
end
