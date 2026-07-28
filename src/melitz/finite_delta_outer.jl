# Section 3 (2026-07-23 governing-correction session): the DIRECT finite-delta bound
# problem,
#
#     minimize/maximize  g(theta) = theta_free[1]   (= log gamma_prime[target])
#     subject to         Delta(theta) <= delta
#
# via KNITRO with an EXPLICIT nonlinear constraint, REUSING `PsiObjectiveBundleImplicit`
# and `outer_loop`/`outer_loop_functions.jl` EXACTLY AS-IS (per explicit instruction: do
# not invent a new bundle type -- the big-picture methodology is identical to the
# Ricardian model; only the `moments!`/`moments_jacobian!` functions fed to the SAME
# generic struct differ, exactly the way `ccOuter.jl` feeds it `EK_moments!`).
#
# What is supplied here (the "callbacks" the user asked to adapt):
#
#   - `melitz_moments_adapter_outer!` (`moments!`): reuses `melitz_moments_adapter!`
#     verbatim for G (the D^2+1 active moment system, unchanged), then overwrites K with
#     `theta_free[1]` (= log gamma_prime_target) -- the actual "Implicit" objective for
#     this program (a deterministic function of theta alone, exactly like the Ricardian
#     GT counterfactual, `docs/melitz_delta_star.md` Section 16/§5 of the CC-outer-loop
#     research note). `PsiObjectiveBundleImplicit`'s own `inner_loop_internal` reads
#     `H[1,1]` as kappa (`H_save = H[1,1]*(-1)^find_smallest`) -- setting K to a CONSTANT
#     column (same value at every draw) is what makes this "Implicit" rather than
#     "Explicit" in this codebase's own vocabulary.
#
#   - `melitz_moments_jacobian_b!`/`melitz_moments_jacobian_d!` (`moments_jacobian!`):
#     Melitz-specific analogues of `EK_moments_Jacobian!`, using Methods B/D (Section 1)
#     instead of naive ForwardDiff-through-the-hard-participation-gate for the MOMENT
#     Jacobian `G_jac` (the objective's own Jacobian, `K_jac`, is trivial: `[1,0,...,0]`
#     at every draw, since K=theta_free[1] exactly). This is the field
#     `PsiObjectiveBundleImplicit` already exposes for exactly this purpose
#     (`calculate_jac_θ!`/`calculate_grad_k!`, `cc_algo/outer_loop_functions.jl:232-303`,
#     default to ForwardDiff when `moments_jacobian! == error`, else call the supplied
#     function directly) -- no cc_algo file is modified.
#
# Everything else -- `PsiObjectiveBundleImplicit`'s struct/functor, the KNITRO F/G
# callback dispatch pattern, the nested real KNITRO inner solve -- follows the Ricardian
# model's own `ccOuter.jl` usage pattern (`counterType==1` branch) line for line, BUT (see
# the 2026-07-23 correctness-repair session below) this file's OWN combined callback no
# longer delegates the outer objective/constraint SEMANTICS to `outer_loop`/
# `outer_loop_constraints!`/`callbackEval_and_ConsF/G_outer!` (still reused for the
# Ricardian model, untouched) -- it builds its own KNITRO problem directly, because this
# milestone's objective/constraint conventions (Section 3.1/4.1 below) are Melitz-specific
# and must not silently inherit `cc_algo`'s shared 1e10-scaled, inner-solve-coupled
# convention (see the 2026-07-23 bug writeup below for why that coupling was unsafe).
#
# ============================================================================
# 2026-07-23 correctness-repair session: three real bugs fixed, on top of the (still
# valid) Section 18 sign fix. See docs/melitz_delta_star.md Section 20 for the full report.
# ============================================================================
#
# 1. OBJECTIVE CONTAMINATION (main prompt Section 3). The prior combined callback set
#    `evalResult.obj[1] = -objSol` where `objSol` came from `inner_loop_internal`'s own
#    return value -- `obj.H_save = theta[1]*(-1)^find_smallest` on a SUCCESSFUL inner
#    solve, but the FIXED FAILURE SENTINEL `-1e10` (regardless of theta) whenever the
#    inner solve failed (`inner_loop_internal`'s own `nStatus` branch,
#    `cc_algo/inner_loop_functions.jl:217-241`). This is EXACTLY the "lower run's outer
#    objective becomes 1.0e10" bug: a failed inner solve silently overwrote a perfectly
#    well-defined outer objective (`theta[1]` is always finite) with a value that has
#    nothing to do with the counterfactual gamma coordinate. Fixed (Section 3.1 below):
#    the objective is now computed DIRECTLY from `theta` -- `evalResult.obj[1] =
#    find_smallest ? theta[1] : -theta[1]` -- with NO dependence on the inner solve's
#    success/failure at all. The inner solve is still needed (for the divergence
#    constraint), but its own return value never touches `evalResult.obj`.
#
# 2. OPAQUE 1e10 CONSTRAINT SCALING (main prompt Section 4). The prior constraint row was
#    `constr[1] = +1e10*Delta(theta) <= 1e10*delta` -- correct in SIGN (Section 18) but
#    badly scaled: at `delta=1e-3`, a raw divergence violation of `4.26e-6` produced a
#    KNITRO feasibility error around `42,600`, next to the cutoff rows' own O(1) scale.
#    Fixed: `c_delta(theta) = Delta(theta)/delta <= 1` (Section 4.1's preferred
#    dimensionless representation) -- the SAME underlying `1e10*Delta(theta)` raw value
#    the shared `PsiObjectiveBundleImplicit` functor already computes (unchanged, no
#    cc_algo edits), just divided by `1e10*delta` for BOTH the constraint value and its
#    Jacobian (Section 4.2: same scaling factor applied consistently everywhere) before
#    handing either to KNITRO.
#
# 3. INNER FAILURE HANDLING (main prompt Section 5). The prior callback wrote `local_c[1]
#    = 1e9` as a hand-invented placeholder whenever `abs(objSol)==1e10`, telling KNITRO
#    the point WAS successfully evaluated with a huge (but finite, differentiable-looking)
#    constraint value -- exactly the failure mode `full_aod_diag/d4_exact/
#    c9_phase8_d20_pilot.jl` (the mature Ricardian-model pilot, found live in a PRIOR
#    session on that file) already diagnosed and fixed for the Ricardian model: KNITRO.jl's
#    own callback wrapper (`_try_catch_handler` in `C_wrapper.jl`) already catches any
#    exception thrown inside an eval callback and converts it to a proper KNITRO
#    evaluation-error return code, telling KNITRO "this point could not be evaluated,
#    reject it and backtrack" -- the correct, robust way to signal infeasibility. Fixed:
#    on a genuine inner-solve failure (bad `nStatus` even after one cold retry), this
#    file's own callback now `throw`s a `DomainError`, matching that same convention,
#    rather than inventing a constraint value.
#
# Also implemented (main prompt Section 2): initial-incumbent installation, live
# feasible-candidate tracking during the trajectory, and end-of-run cold reverification --
# see `solve_melitz_finite_delta_bound`'s own docstring.

using KNITRO
using LinearAlgebra: dot

# ============================================================================
# moments!: the Implicit objective K = theta_free[1], G = the D^2+1 active moments
# (unchanged from melitz_moments_adapter!).
# ============================================================================

"""
    melitz_moments_adapter_outer!(K, G, theta, U, obj)

`moments!` for the finite-delta OUTER Implicit bundle. Fills `G` EXACTLY as
`melitz_moments_adapter!` does (the same D^2+1 active moment system used by the inner
Delta(theta) solve -- no duplicate G-construction logic), then overwrites `K` with the
constant `theta[1]` (`= log gamma_prime_target`, the real outer objective for the
upper/lower gains-from-trade program). A constant K (same value at every draw) is what
makes this bundle "Implicit" in this codebase's vocabulary -- `PsiObjectiveBundleImplicit`
reads only `H[1,1]` as kappa. NOTE (2026-07-23 correctness-repair session): `K`/`H_save`
is used ONLY by the (now-unused-for-the-objective) legacy `calculate_grad_k!` bookkeeping
path; the outer objective itself is computed directly from `theta` in the combined
callback (Section 3.1), never read off `K`/`H_save`.
"""
function melitz_moments_adapter_outer!(K, G, theta, U, obj)
    melitz_moments_adapter!(K, G, theta, U, obj)
    K .= theta[1]
    return nothing
end

# ============================================================================
# moments_jacobian!: Melitz-specific replacements for naive ForwardDiff-through-the-
# hard-participation-gate (Section 1's whole point). K_jac is trivial; G_jac uses
# Method B (finite-difference secant) or Method D (hand-derived, chained through the
# ALREADY-validated gravity-pivot ForwardDiff Jacobian) -- see gradient_lab.jl.
# ============================================================================

"""
    make_melitz_moments_jacobian_b(h) -> Function

Returns a `moments_jacobian!`-compatible closure (Backend B, Section 4.1): `K_jac` is the
exact `[1,0,...,0]` row (repeated for every draw); `G_jac[:,:,k]` is the central finite
difference `[fixed_active_set_moments(theta+h*e_k) - fixed_active_set_moments(theta-h*e_k)]
/ (2h)` for every outer coordinate `k` -- exactly Method B's own construction
(`method_b_fixed_dual_secant`), generalized from one aggregated scalar to the FULL
per-draw moment matrix (so `PsiObjectiveBundleImplicit`'s own existing `calculate_jac_θ!`/
`∂c_∂θ` machinery, UNCHANGED, can assemble the divergence-constraint Jacobian from it via
the same BLAS contraction production uses).

`calculate_grad_k!`'s own small-`N` (2-draw) probe call is detected by `size(U,1) <
size(obj.U,1)` and skipped (that call only reads `K_jac`, discarding `G_jac` entirely --
`cc_algo/outer_loop_functions.jl:279-291`).

Section 12.4/7.5 optimization: the `2*n` displaced moment builds per gradient callback
(`n=30` at D=4, i.e. 60 full `W x (D^2+1)` matrix constructions) now reuse TWO persistent
`Float64` buffers (`fixed_active_set_moments!`, gradient_lab.jl) closed over by this
function, rebuilt only if a first call or a `(W, num_moments)` shape change is detected
(defensive, e.g. a caller reusing this closure across bundles of different `D`/`W`) --
rather than allocating a fresh `(W x num_moments)` matrix on every one of the 60 probes.
Bit-identical numerics to the prior always-allocating version (same shared fill body).
"""
function make_melitz_moments_jacobian_b(h::Real)
    Gp_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Gm_buf = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    profit_buf = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function melitz_moments_jacobian_b!(K_jac, G_jac, theta, U, obj)
        n = length(theta)
        K_jac .= 0.0
        K_jac[:, 1] .= 1.0
        if size(U, 1) < size(obj.U, 1)
            G_jac .= 0.0
            return nothing
        end
        ctx = obj.γ
        W = size(obj.U, 1)
        d = ctx.moment_layout.num_moments
        if Gp_buf[] === nothing || size(Gp_buf[]) != (W, d)
            Gp_buf[] = zeros(Float64, W, d)
            Gm_buf[] = zeros(Float64, W, d)
            profit_buf[] = zeros(Float64, W)
        end
        Gp, Gm, profit_j = Gp_buf[], Gm_buf[], profit_buf[]
        ei = zeros(n)
        @inbounds for k in 1:n
            ei[k] = 1.0
            fixed_active_set_moments!(Gp, profit_j, theta .+ h .* ei, ctx, obj)
            fixed_active_set_moments!(Gm, profit_j, theta .- h .* ei, ctx, obj)
            @views G_jac[:, :, k] .= (Gp .- Gm) ./ (2h)
            ei[k] = 0.0
        end
        return nothing
    end
    return melitz_moments_jacobian_b!
end

"""
    melitz_moment_directional_derivative(theta, v, ctx, obj) -> dG (W x num_moments)

Section 1.5's Method D per-column formulas (the SAME closed-form active-set-conditional
derivatives as `method_d_hand_derived`), but returning the FULL directional-derivative
MATRIX (every moment column) rather than the single dual-scalar-weighted aggregate --
factored out so both `method_d_hand_derived` (Section 1) and
`make_melitz_moments_jacobian_d` (Section 3) share ONE implementation.
"""
function melitz_moment_directional_derivative(theta::AbstractVector, v::AbstractVector, ctx, obj)
    D, j = ctx.D, ctx.target_country
    W = size(obj.U, 1)
    sigma = ctx.sigma
    layout = ctx.moment_layout

    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    dvec = ForwardDiff.derivative(a -> expand_theta_econ_vector(theta .+ a .* v, ctx), 0.0)
    D2 = D * D
    dA = reshape(dvec[1:D2], D, D)
    df = reshape(dvec[D2+1:2*D2], D, D)
    dgamma = dvec[2*D2+1]
    dfjj = dvec[2*D2+2]

    expenditure_prime = ctx.w_prime * ctx.L[j]
    price_power_autarky = gamma_prime_j

    dG = zeros(Float64, W, layout.num_moments)
    dprofit_j = zeros(Float64, W)

    @inbounds for o in 1:D, d in 1:D
        trade_col = layout.trade_index[o, d]
        for w in 1:W
            z = obj.U[w, o]
            firm = melitz_firm(ctx.w[o], ctx.tau[o, d], A[o, d], f[o, d], sigma, ctx.expenditure[d], 1.0, z)
            if firm.active
                dG[w, trade_col] = (sigma - 1) * (firm.unconstrained_revenue / ctx.expenditure[d]) / A[o, d] * dA[o, d]
            end
            if o == j && firm.active
                dprofit_j[w] += (sigma - 1) * firm.unconstrained_revenue / sigma / A[o, d] * dA[o, d] -
                                 ctx.w[o] * df[o, d]
            end
        end
    end

    link_col = layout.focal_link_index
    @inbounds for w in 1:W
        z_j = obj.U[w, j]
        firm_auk = melitz_firm(ctx.w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, price_power_autarky, z_j)
        dpi_auk = 0.0
        if firm_auk.active
            dpi_auk = (sigma - 1) * firm_auk.unconstrained_revenue / sigma / A[j, j] * dA[j, j] -
                      firm_auk.unconstrained_revenue / sigma / price_power_autarky * dgamma -
                      ctx.w_prime * dfjj
        end
        dG[w, link_col] = dprofit_j[w] / ctx.w[j] - dpi_auk / ctx.w_prime
    end
    return dG
end

"""
    make_melitz_moments_jacobian_d() -> Function

Backend D (Section 1.5/4.1) analogue of `make_melitz_moments_jacobian_b`: `G_jac[:,:,k]`
is `melitz_moment_directional_derivative(theta, e_k, ctx, obj)`, the hand-derived
closed-form branch derivative (exact at a fixed active set, no finite-difference
bandwidth). Cross-validated against Method B/C at zero-switch points (Section 1/2).
"""
function make_melitz_moments_jacobian_d()
    function melitz_moments_jacobian_d!(K_jac, G_jac, theta, U, obj)
        n = length(theta)
        K_jac .= 0.0
        K_jac[:, 1] .= 1.0
        if size(U, 1) < size(obj.U, 1)
            G_jac .= 0.0
            return nothing
        end
        ctx = obj.γ
        ei = zeros(n)
        @inbounds for k in 1:n
            ei[k] = 1.0
            @views G_jac[:, :, k] .= melitz_moment_directional_derivative(theta, ei, ctx, obj)
            ei[k] = 0.0
        end
        return nothing
    end
    return melitz_moments_jacobian_d!
end

# ============================================================================
# Bundle construction, reusing PsiObjectiveBundleImplicit AS-IS.
# ============================================================================

"""
    build_melitz_implicit_bundle(ctx, theta_free_init; delta, find_smallest,
        gradient_backend=:B, h=1e-4, inner_loop_opt=..., outer_loop_opt=...,
        delta_evaluation_cap=nothing) -> PsiObjectiveBundleImplicit

Constructs the SAME `PsiObjectiveBundleImplicit` struct the Ricardian model uses
(`cc_algo/PsiObjectiveBundle.jl`, unmodified), wired to Melitz's own moments/gradient:
`moments! = melitz_moments_adapter_outer!`, `moments_jacobian! =`
`make_melitz_moments_jacobian_b(h)` or `_d()` depending on `gradient_backend`,
`d = ctx.moment_layout.num_moments`, `outer_constr_index = d+1` (all moments internal --
see this file's header), `l = length(theta_free_init)`, `U = z_draws` (the SAME reference
draws the inner Delta(theta) solve uses).

`delta_evaluation_cap` (2026-07-24 evaluation-cap-correction session, governing prompt
Sections 2-3, superseding this docstring's own prior `delta`-coupled description): `cc_algo`'s
shared functor (`PsiObjectiveBundle.jl`'s `(Q::PsiObjectiveBundleImplicit)(...)`) already
contains an objective-threshold early-stop mechanism -- `if f <= lower_limit; return
-KNITRO.KN_INFINITY; else; return f; end`, where `f` is the raw dual objective KNITRO
minimizes (`f = -Delta` at the optimum, and `-f(x) <= Delta` for EVERY dual point `x` by weak
duality, since the CC dual here is unconstrained -- see inner_screening.jl's header). An
EARLIER session tied this to the OUTER BUDGET `delta` (`lower_limit = -(delta + margin)`) --
diagnosed as a central conceptual error: a certified lower bound above the CURRENT outer
budget is not evidence the point is unsolvable, only that it exceeds that one budget, so
aborting there threw away an ordinary finite, fully-solvable point (Case A: `FiniteSolved`,
`inner_screening.jl`). This early-stop must instead be gated on the SEPARATE
`delta_evaluation_cap` (governing prompt Section 2's evaluation cap, e.g. `10.0` -- routine
solves are never aborted merely for exceeding `delta`, only for certifiably exceeding this
cap): `lower_limit = -delta_evaluation_cap`, exactly.

2026-07-26 production-closure session (governing prompt Phase 1, "make the evaluation cap
impossible to omit"): CORRECTED activation rule, superseding this docstring's own prior
description. The previous rule left `lower_limit` disabled whenever a separate
`lower_limit_guard` kwarg was omitted, EVEN IF `delta_evaluation_cap` was supplied -- a
caller passing `delta_evaluation_cap=10.0` alone got a silently-disabled cap (confirmed
live, `docs/melitz_production_fast_backend_2026-07-26.md` Section 5.5: a `13.5x` slower
campaign, 31 spurious `NumericalFailure`s). The corrected rule -- and, per direct user
feedback the same session, simplified further by removing the separate `lower_limit_guard`/
`guard` margin entirely (it was inherited from the OLD `delta`-coupled design, which
genuinely needed a margin because aborting exactly AT `delta` was itself the bug; once the
threshold is the evaluation cap itself -- a value chosen deliberately far from any routine
`Delta` -- the cap IS the threshold, and a second number to reason about added nothing):

  - `inner_solve_config::MelitzInnerSolveConfig` given -- use it as-is (preferred going
    forward); an error to also pass `delta_evaluation_cap`.
  - `delta_evaluation_cap !== nothing` -- the cap is ALWAYS active:
    `lower_limit = -delta_evaluation_cap`, exactly.
  - NEITHER kwarg given -- `lower_limit = -KNITRO.KN_INFINITY` (disabled), the one remaining
    way to get an uncapped bundle, and now an all-defaults, self-evident choice rather than
    a trap a caller falls into while believing a cap is active.

The KNITRO-native mid-solve stop this produces is classified
`AboveEvaluationCap(...,:live_dual_threshold,...)` by `melitz_classified_inner_solve`
(inner_screening.jl), never `NumericalFailure` and never `BudgetInfeasible` (that name/type
no longer exists as of this session).
"""
function build_melitz_implicit_bundle(ctx, z_draws::AbstractMatrix, theta_free_init::AbstractVector;
                                       delta::Real, find_smallest::Bool,
                                       gradient_backend::Symbol=:auto, h::Real=1e-4,
                                       inner_loop_opt::AbstractString,
                                       outer_loop_opt::AbstractString,
                                       delta_evaluation_cap::Union{Nothing,Real}=nothing,
                                       inner_solve_config::Union{Nothing,MelitzInnerSolveConfig}=nothing,
                                       backend::Symbol=:auto_from_gradient_backend,
                                       hessian_backend::Symbol=:auto,
                                       forbid_dense_fallback::Bool=false)
    backend in (:matrix_free, :dense_reference, :auto_from_gradient_backend) || throw(ArgumentError(
        "build_melitz_implicit_bundle: backend must be :matrix_free, :dense_reference, or " *
        "(default) :auto_from_gradient_backend, got $backend"))
    d = ctx.moment_layout.num_moments
    l = length(theta_free_init)
    D = ctx.D
    # 2026-07-26 production-port session: `gradient_backend=:auto` (NEW default, replacing
    # the old unconditional `:B`) resolves to the sorted crossing-slice direct backend
    # whenever a sorted-tail context is available (built below for `backend=:matrix_free`,
    # or already present on `ctx` if the paired FC-side bundle used a sorted moment_backend),
    # else the plain direct backend -- never silently falling back to the old dense `:B`
    # finite-difference-of-full-moments backend, which this session's own inventory audit
    # found was still the un-flipped default. `D`/thread-count threshold matches
    # `MelitzBackendConfig`'s own `:auto` rule (backend_config.jl).
    #
    # `backend` DEFAULT (`:auto_from_gradient_backend`) is itself resolved from the CALLER'S
    # OWN `gradient_backend` choice, BEFORE `:auto` is expanded: if the caller explicitly
    # asked for one of the legacy dense-only gradient mechanisms (`:B`, `:B_localized`,
    # `:B_localized_parallel`, `:B_argument_localized_serial`, `:B_argument_localized_parallel`,
    # `:D` -- every one of which reads `obj.H` and has NO matrix-free equivalent), that
    # request is honored by silently building the dense-reference bundle underneath it,
    # rather than throwing -- this is what a caller explicitly requesting `:B` clearly
    # wants, and it is what this repo's own ~40 existing test call sites (comparison/
    # cross-check baselines against `:B`) already assume. `backend=:matrix_free` passed
    # EXPLICITLY still hard-errors on an incompatible gradient_backend below (a genuine,
    # surfaced conflict), and `backend=:dense_reference` passed explicitly is always honored.
    legacy_dense_only_gradient_backends = (:B, :B_localized, :B_localized_parallel,
        :B_argument_localized_serial, :B_argument_localized_parallel, :D)
    if backend == :auto_from_gradient_backend
        backend = gradient_backend in legacy_dense_only_gradient_backends ? :dense_reference : :matrix_free
    end
    cfg = MelitzBackendConfig(inner_backend=backend, outer_gradient_backend=gradient_backend, hessian_backend=hessian_backend)
    have_ctx_sorted_ctx = get(ctx, :sorted_tail_ctx, nothing) !== nothing
    resolved_gradient_backend = gradient_backend != :auto ? gradient_backend :
        (backend == :matrix_free || have_ctx_sorted_ctx) ?
            melitz_resolve_gradient_backend(MelitzBackendConfig(inner_backend=:matrix_free), D) :
            melitz_resolve_gradient_backend(MelitzBackendConfig(inner_backend=:dense_reference), D)
    gradient_backend = resolved_gradient_backend
    # 2026-07-26 closure session (governing prompt Phase 2): strict production-fast callers
    # (MELITZ_PRODUCTION_FAST's own forbid_dense_fallback=true) must fail HERE, at
    # construction, if the resolved backend is dense -- never after an expensive callback
    # begins. `backend==:dense_reference` is the complete condition (a caller who explicitly
    # requested a legacy dense-only gradient_backend already forced backend=:dense_reference
    # above, so this single check also covers that case -- no separate gradient_backend
    # check is needed).
    if forbid_dense_fallback && backend == :dense_reference
        throw(ArgumentError(
            "build_melitz_implicit_bundle: forbid_dense_fallback=true (strict production-fast " *
            "mode) but the resolved backend is :dense_reference (from backend=$backend, " *
            "gradient_backend=$gradient_backend) -- pass backend=:matrix_free and a matrix-free " *
            "gradient_backend (or leave gradient_backend=:auto) for a strict caller, or " *
            "forbid_dense_fallback=false (MELITZ_PRODUCTION_COMPAT) if this dense choice is " *
            "genuinely intended."))
    end
    # 2026-07-26 production-closure session (governing prompt Phase 1, then simplified same
    # day per direct user feedback removing the separate guard/margin -- see this function's
    # own docstring and inner_solve_config.jl's file header for the full history). The ONLY
    # way to end up with a disabled (`-KN_INFINITY`) `lower_limit` is to supply NEITHER
    # `inner_solve_config` NOR `delta_evaluation_cap` at all -- an explicit, all-defaults
    # choice (still the right default for a single bounded D=4/D=20 one-shot solve that never
    # mentions a cap). The MOMENT a caller supplies `delta_evaluation_cap`, the cap is active,
    # exactly at that value: `lower_limit = -delta_evaluation_cap`.
    local lower_limit
    if inner_solve_config !== nothing
        delta_evaluation_cap !== nothing && throw(ArgumentError(
            "build_melitz_implicit_bundle: pass EITHER inner_solve_config OR delta_evaluation_cap, not both"))
        lower_limit = inner_solve_config.lower_limit
    elseif delta_evaluation_cap !== nothing
        lower_limit = melitz_configure_lower_limit(:evaluation_cap;
            delta_evaluation_cap=delta_evaluation_cap, outer_delta=delta)
    else
        lower_limit = -KNITRO.KN_INFINITY
    end

    # ADDITIVE (continuation4, Section 4): the two new "direct" backends never touch
    # moments_jacobian!/jac_h at all -- cb_G! (below) bypasses this bundle's own theta-branch
    # entirely for them, calling melitz_gradient_delta_direct_{serial,parallel}! directly
    # instead. moments_jacobian! is left at its `error` sentinel default (dead code on this
    # path, never invoked) and needs_outer_moment_jacobian=false skips the dense jac_h
    # allocation on this (outer Implicit) bundle -- the genuinely memory-scalable case the
    # governing prompt's Section 4 asks for (no W x K x n tensor anywhere, not even zeroed).
    # 2026-07-26 sorted-tail continuation: :B_direct_argument_sorted_serial/_parallel
    # (src/melitz/sorted_crossing_gradient.jl) join the "direct" family -- same no-jac_h
    # path as the pre-existing two, requiring ctx.sorted_tail_ctx (built via
    # build_melitz_psi_bundle*'s own moment_backend kwarg) to be present; that check happens
    # inside the gradient closure itself (a clear ArgumentError there), not here.
    is_direct = gradient_backend in (:B_direct_argument_serial, :B_direct_argument_parallel,
                                      :B_direct_argument_sorted_serial, :B_direct_argument_sorted_parallel,
                                      :B_direct_argument_touched_row_serial)
    mj! = gradient_backend == :B ? make_melitz_moments_jacobian_b(h) :
          gradient_backend == :B_localized ? make_melitz_moments_jacobian_b_localized(h) :
          gradient_backend == :B_localized_parallel ? make_melitz_moments_jacobian_b_localized_parallel(h) :
          gradient_backend == :B_argument_localized_serial ? make_melitz_moments_jacobian_b_argument_localized_serial(h) :
          gradient_backend == :B_argument_localized_parallel ? make_melitz_moments_jacobian_b_argument_localized_parallel(h) :
          gradient_backend == :D ? make_melitz_moments_jacobian_d() :
          is_direct ? error :
          error("gradient_backend must be :B, :B_localized, :B_localized_parallel, " *
                ":B_argument_localized_serial, :B_argument_localized_parallel, " *
                ":B_direct_argument_serial, :B_direct_argument_parallel, " *
                ":B_direct_argument_sorted_serial, :B_direct_argument_sorted_parallel, " *
                ":B_direct_argument_touched_row_serial, or :D for " *
                "the KNITRO-native Implicit path (Backend R does not fit the moments_jacobian! hook -- see file header)")

    if backend == :matrix_free
        is_direct || throw(ArgumentError(
            "build_melitz_implicit_bundle: backend=:matrix_free requires gradient_backend in " *
            "the :B_direct_argument_* family (resolved gradient_backend=$gradient_backend) -- " *
            "MelitzCCBundle's functor does not implement the jac_h theta-branch (confirmed " *
            "structurally unreachable/vacuous for Melitz, cc_bundle.jl header)."))
        sorted_tail_ctx = have_ctx_sorted_ctx ? ctx.sorted_tail_ctx :
            build_melitz_sorted_tail_context(z_draws, ctx.sigma; theta_star=ctx.theta_star)
        op = build_melitz_moment_operator(sorted_tail_ctx, ctx.moment_layout)
        resolved_hessian_backend = melitz_resolve_hessian_backend(cfg, D)
        obj = build_melitz_cc_bundle(op, ctx; mode=:implicit, U=z_draws,
            outer_constr_index=d + 1, find_smallest=find_smallest, lower_limit=lower_limit,
            inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt,
            hessian_backend=resolved_hessian_backend)
        return obj
    end

    obj = PsiObjectiveBundleImplicit(
        δ=Float64(delta),
        find_smallest=find_smallest,
        γ=ctx,
        (moments!)=melitz_moments_adapter_outer!,
        (moments_jacobian!)=mj!,
        d=d,
        l=l,
        outer_constr_index=d + 1,
        inequality_index=Int64[],
        U=z_draws,
        inner_loop_opt=inner_loop_opt,
        outer_loop_opt=outer_loop_opt,
        lower_limit=lower_limit,
        needs_outer_moment_jacobian=!is_direct,
    )
    return obj
end

# ============================================================================
# Section 2.3: unambiguous outer-feasibility classification. Replaces ad hoc
# `terminal_eval.verified && terminal_eval.Delta <= delta` checks scattered across the
# driver/tests with ONE named predicate per necessary condition, so nothing is ever
# printed/returned as "feasible=true" for a point with Delta > delta or a failed inner
# solve.
# ============================================================================

"""
    MelitzOuterFeasibilityClassification

Section 2.3's explicit, unambiguous feasibility fields for a fixed-outer-point evaluation
(`MelitzDeltaEvalResult`), replacing the single overloaded `feasible`/`verified` booleans
wherever a caller needs to know WHICH condition is (not) satisfied:

  - `inner_verified`: the inner CC dual solve itself converged to KNITRO's strict optimal
    status (`nStatus==0` -- NOT the looser `[0,-100,-101,-103]` acceptance set used
    elsewhere in this codebase for continuing a trajectory; a value reported as a
    candidate INCUMBENT must clear the strict bar).
  - `inner_moment_feasible`: `lfd_ok` -- the recovered LFD's own internal consistency
    checks (normalization, moment residuals, primal-dual gap) all pass.
  - `cutoff_feasible`: the Section 1.3 deterministic cutoff/export-selection inequalities
    hold (`min_slack >= 0`).
  - `gravity_feasible`: the two gravity restrictions hold to `gravity_tol` (structural, by
    pivot construction -- should ALWAYS be machine-precision-true; checked explicitly
    rather than assumed, since `false` here would indicate a genuine construction bug).
  - `budget_feasible`: `Delta(theta) <= delta` -- the ACTUAL divergence budget this
    milestone's program constrains on.
  - `outer_feasible`: the conjunction of all five -- the ONLY condition under which a
    point may be reported as a valid economic incumbent.
"""
struct MelitzOuterFeasibilityClassification
    inner_verified::Bool
    inner_moment_feasible::Bool
    cutoff_feasible::Bool
    gravity_feasible::Bool
    budget_feasible::Bool
    outer_feasible::Bool
end

"""
    melitz_classify_outer_feasibility(r::MelitzDeltaEvalResult, delta; gravity_tol=1e-6)
        -> MelitzOuterFeasibilityClassification

Builds the Section 2.3 classification from an existing `MelitzDeltaEvalResult` (from
either `evaluate_melitz_delta` or `evaluate_melitz_delta_from_solution`) and the outer
program's own `delta` budget.
"""
function melitz_classify_outer_feasibility(r::MelitzDeltaEvalResult, delta::Real;
                                            gravity_tol::Real=1e-6)
    inner_verified = r.nStatus == 0
    inner_moment_feasible = r.lfd_ok
    cutoff_feasible = r.feasible
    gravity_feasible = r.equilibrium_check !== nothing &&
                        abs(r.equilibrium_check.gravity_residual_A) < gravity_tol &&
                        abs(r.equilibrium_check.gravity_residual_f) < gravity_tol
    budget_feasible = isfinite(r.Delta) && r.Delta <= delta
    outer_feasible = inner_verified && inner_moment_feasible && cutoff_feasible &&
                      gravity_feasible && budget_feasible
    return MelitzOuterFeasibilityClassification(inner_verified, inner_moment_feasible,
        cutoff_feasible, gravity_feasible, budget_feasible, outer_feasible)
end

"""
    MelitzOuterCandidate

Section 2.1/2.2: one complete, immutable candidate incumbent -- the full
`MelitzDeltaEvalResult` (theta_free, full A/f/gamma_prime, cutoff matrix and slacks,
dual/LFD, Delta, gravity/moment residuals, every verification diagnostic), its Section 2.3
feasibility classification, the signed objective value being optimized (`+theta[1]` for
the upper direction, `-theta[1]` for the lower -- so "smaller is better" uniformly,
matching what KNITRO itself always minimizes, Section 3.1), and a `source` tag recording
how this candidate was obtained.
"""
struct MelitzOuterCandidate
    objective::Float64
    eval::MelitzDeltaEvalResult
    classification::MelitzOuterFeasibilityClassification
    source::Symbol   # :initial, :live, :cold_verified
end

# ============================================================================
# Section 3.4/13.F: the finite-delta outer result, with unambiguous incumbent fields.
# ============================================================================

"""
    MelitzFiniteDeltaOuterResult

`terminal_eval`/`terminal_classification` describe whatever KNITRO's own trajectory ended
at (may be infeasible/unverified -- reported honestly, never hidden). `initial_incumbent`
is `theta_init`'s own cold-verified outer-feasibility evaluation (Section 2.1 -- installed
BEFORE `KN_solve` is ever called, so it survives even a 0/1-iteration KNITRO run).
`best_live_incumbent` is the best (smallest signed objective) outer-feasible point observed
during the trajectory itself (Section 2.2, WARM -- not yet independently reverified).
`cold_verified_incumbent` is the best live candidate that survives an independent cold
reverification (a fresh KNITRO solve from a cleared warm start, Section 2.2 steps 2-4),
falling back to `initial_incumbent` if none does -- THIS is the field callers should treat
as "the answer."
"""
struct MelitzFiniteDeltaOuterResult
    theta_init::Vector{Float64}
    delta::Float64
    direction::Symbol
    gradient_backend::Symbol
    terminal_theta::Vector{Float64}
    terminal_eval::MelitzDeltaEvalResult
    terminal_classification::MelitzOuterFeasibilityClassification
    initial_incumbent::Union{Nothing,MelitzOuterCandidate}
    best_live_incumbent::Union{Nothing,MelitzOuterCandidate}
    cold_verified_incumbent::Union{Nothing,MelitzOuterCandidate}
    nStatus::Int
    inner_solve_count::Int
    inner_infeas_count::Int
    inner_eval_failures::Int
    wall_time::Float64
    cutoff_constraint_backend::Symbol   # Section 3.3/4: :linear or :nonlinear_reference
    n_fc_calls::Int                     # Section 4 benchmark: total cb_F! invocations
    n_ga_calls::Int                     # Section 4 benchmark: total cb_G! invocations
    # 2026-07-24 evaluation-cap-correction session: the cap this specific solve used, carried
    # on the result itself so a report never has to cross-reference the call site to know
    # which threshold produced these counts. There is no "policy" field: an
    # AboveEvaluationCap/InfiniteDeltaCertified point ALWAYS reports the SAME fixed sentinel
    # constraint value (`delta_evaluation_cap/delta`) with a ZERO gradient, as an ordinary
    # successful evaluation -- see `melitz_build_finite_delta_callbacks`'s own docstring for
    # why this fixed, self-consistent sentinel is not the same class of error as the original
    # bug (which reported the certificate's own path-dependent value/gradient). Only
    # `NumericalFailure` remains a genuine KNITRO evaluation error.
    delta_evaluation_cap::Float64
    # Addendum Section 16, RENAMED this session (governing prompt Section 11: never keep the
    # old `n_moment_infeasible_reject`/`n_budget_infeasible_reject` names -- they described
    # the pre-correction semantics). Every FC/GA call that is not an exact-cache hit resolves
    # to exactly one of: n_inner_solved, n_infinite_delta_reject, n_above_cap_reject,
    # n_numerical_failure_reject.
    n_inner_solved::Int
    n_infinite_delta_reject::Int
    n_above_cap_reject::Int
    n_numerical_failure_reject::Int
    # 2026-07-27 continuation: Melitz-owned, backend-agnostic replacement for a prior
    # unconditional `CounterfactualSensitivity.INNER_SOLVE_COUNT[]` read (see
    # run_diagnostics.jl's own header for the full incident writeup). `inner_solve_count`/
    # `inner_infeas_count` above are now DERIVED from this same object (not from CS), so this
    # field is the authoritative source; the two scalar fields are kept for backward
    # compatibility with existing callers/tests that read them by name.
    diagnostics::MelitzRunDiagnostics
end

"""
    MelitzExactPointCache(max_size=256)

Continuation session (2026-07-23), Section 4.2: a SHAREABLE upgrade of the prior session's
single-slot `exact_cache_theta`/`exact_cache_result`/`exact_cache_H` mechanism (Section 5.1)
from "remembers only the MOST RECENTLY verified theta" to "remembers EVERY verified theta
seen so far, across however many `melitz_build_finite_delta_callbacks` calls share the SAME
cache object." Valid across DIFFERENT outer budgets `delta`: `Delta(theta)` (and its
optimal dual/moment state) is purely a function of `theta`, never of the outer budget --
see `inner_screening.jl`'s own header on weak duality, and this file's own
`build_melitz_implicit_bundle` docstring. A caller wanting cross-delta reuse (e.g. a
continuation workflow: solve at `delta=1e-3`, then reuse verified points at `delta=1e-2`)
constructs ONE `MelitzExactPointCache()` and passes it to every `solve_melitz_finite_delta_bound`
call in the sequence; omitting it (the default `nothing` everywhere this type is accepted)
allocates a fresh, empty, call-local cache exactly as before -- strictly reproducing the
pre-existing single-call-scoped behavior, since a fresh Dict-based cache started empty for
one call behaves identically to a single forgetful slot from that call's own point of view
(this session's own test suite confirms the pre-existing "Section 5.1"/"Section 7 A/B/A"
tests, which call `melitz_build_finite_delta_callbacks` without this new kwarg, are
unaffected). Insert-only-on-a-verified-solve, matching the mechanism it replaces -- a
non-finite or failed solve is never written here.

Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
D") Section 4: bounded via the shared `MelitzLRUOrder` machinery (`bounded_cache.jl`),
default capacity `256` -- a documented DEVIATION from that session's literal "compact
cache" spec (theta/Delta/dual/status/context-fingerprint only, no moment matrix): this
cache's own correctness fix (see `inner_solve_verified_or_fail` below) requires restoring
`obj.H` on a hit, so each entry HERE still carries a `copy(obj.H)` (a `(W, num_moments+2)`
matrix), unlike the deliberately tiny `MelitzDeltaEvalCache` (`delta_star.jl`), which is
this codebase's actual "heavy-state" tier. The capacity here is set much larger (256, not
1-4) because this cache's HIT RATE is what makes the duplicate `cb_F!`/`cb_G!`-at-the-same-
iterate elimination free (documented 26/26 hits per cell in the continuation2 report) --
capping it as small as the heavy-state tier would defeat that. Both are BOUNDED now, per
the governing prompt's "do not retain every full moment matrix indefinitely," at capacities
suited to their own distinct access patterns; see docs/melitz_optimization_report_2026-07-23_continuation3.md
Section C for the full before/after memory accounting and rationale.

`fp` is a fingerprint stored alongside every COMPACT entry and checked on lookup: a hit
whose fingerprint does not match the QUERYING call's own is treated as a miss (and the
stale entry is dropped) rather than silently restoring `obj.H` from a different context's
moment layout -- a defensive guard against exactly the class of bug this session's own
stale-context test exercises (see `test/melitz/runtests.jl`).

Closure-audit session (2026-07-24), Phase D: two changes on top of the memory-scalability
continuation's own design above.

1. **Content-based fingerprint, not `objectid(ctx)` alone** (governing prompt: "Replace
   `objectid(ctx)` as the sole key with a stable content fingerprint covering: draws, data,
   economic version, parameterization, divergence, moment scaling, solver options").
   `melitz_context_fingerprint(ctx, U=nothing)` (`bounded_cache.jl`) hashes `ctx`'s own
   `D`/`sigma`/`theta_star`/`target_country`/`tau`/`w`/`X_data`/`outer_parameterization`/
   `inner_loop_opt`/`outer_loop_opt` fields (everything Phase D's list names EXCEPT the
   Monte Carlo draws, which are not a field of `ctx` in this codebase's data model -- `U`
   lives on the bundle, one level up -- so `U` is threaded through as an OPTIONAL extra
   argument instead: the production call site below passes `obj.U`, giving full coverage,
   while a caller that only has a bare `ctx` still gets a genuine content fingerprint over
   everything else). A `ctx`-like object missing the expected fields (e.g. this file's own
   synthetic test doubles, `Ref(:ctxA)`) falls back to `objectid(ctx)` -- `objectid` is
   RETAINED, just no longer the ONLY mechanism: two DIFFERENT (freshly reconstructed)
   real Melitz `ctx` objects with IDENTICAL content now fingerprint identically (a
   "recreated-context" cache hit becomes possible), which `objectid` alone could never do.

2. **Compact/heavy-state split** (governing prompt Phase D: "Compact cache: theta, Delta,
   dual, status/residual summary, stable context fingerprint. Heavy-state cache: G or
   compact moment state... Use capacity 1-4 for heavy states and a configurable byte
   limit."). `cache.store` (COMPACT: `theta -> (Delta, x, nStatus, fp)`) keeps its own
   large default capacity (`max_size=256`) -- these entries are a few dozen bytes each, not
   the ~206GB-at-D=20/W=80,000 problem Phase D's own memory projection is about. The full
   `H` matrix (the actual "heavy state," `~W*(d+2)*8` bytes -- ~258MB per entry at
   D=20/W=80,000) now lives in a SEPARATE `heavy_store`, bounded independently by BOTH
   `heavy_max_size` (default 4, per Phase D's literal "capacity 1-4") and `heavy_max_bytes`
   (a configurable byte budget, default 4GB -- whichever limit binds first evicts). A
   COMPACT hit whose heavy entry has since been evicted is NOT a full cache miss: `H` is a
   pure DETERMINISTIC function of `theta` (`obj.moments!` needs no KNITRO dual, only
   `theta`/`obj.U`/`ctx`), so `melitz_exact_cache_get` recomputes it cheaply from `theta`
   alone when an `obj` is supplied (`heavy_recomputes` counts this path) -- preserving the
   COMPACT tier's entire benefit (skipping the ~milliseconds-to-minutes KNITRO solve) even
   when the heavy tier's small capacity has evicted the matching `H`. Without an `obj`
   (e.g. the pre-existing synthetic unit tests below, which never construct a real bundle),
   a heavy-miss safely degenerates to a full cache miss -- never wrong, just not maximally
   efficient, and byte-for-byte the pre-existing behavior for every call site that does not
   opt into the new `obj=` keyword.
"""
mutable struct MelitzExactPointCache
    store::Dict{Vector{Float64},Tuple{Float64,Vector{Float64},Int,UInt}}
    order::MelitzLRUOrder
    max_size::Int
    evictions::Int
    # 2026-07-26 production-port session: widened from Dict{...,Matrix{Float64}} to
    # Dict{...,Any} so a heavy entry can hold EITHER a dense H copy (legacy bundles) or a
    # MelitzOperatorSnapshot (MelitzCCBundle, cc_bundle.jl) -- see melitz_heavy_snapshot/
    # melitz_heavy_restore!/melitz_heavy_bytes/melitz_heavy_recompute (cc_bundle.jl) for the
    # per-bundle-type dispatch this enables.
    heavy_store::Dict{Vector{Float64},Any}
    heavy_order::MelitzLRUOrder
    heavy_max_size::Int
    heavy_max_bytes::Int
    heavy_bytes::Int
    heavy_evictions::Int
    heavy_recomputes::Int
end
function MelitzExactPointCache(max_size::Int=256; heavy_max_size::Int=4,
                                heavy_max_bytes::Int=4_000_000_000)
    MelitzExactPointCache(
        Dict{Vector{Float64},Tuple{Float64,Vector{Float64},Int,UInt}}(), MelitzLRUOrder(), max_size, 0,
        Dict{Vector{Float64},Any}(), MelitzLRUOrder(), heavy_max_size, heavy_max_bytes, 0, 0, 0)
end

"""
    melitz_heavy_evict_until!(cache) -> n_evicted

Pops LRU heavy entries until BOTH `heavy_max_size` and `heavy_max_bytes` are satisfied
(mirrors `melitz_lru_evict_until!`, but also tracks `heavy_bytes` since heavy entries are not
uniformly sized across different `(W, K)` fixtures sharing one cache -- unlikely in practice,
guarded against anyway).
"""
function melitz_heavy_evict_until!(cache::MelitzExactPointCache)
    n_evicted = 0
    while length(cache.heavy_order.keys) > 0 &&
          (length(cache.heavy_order.keys) > cache.heavy_max_size || cache.heavy_bytes > cache.heavy_max_bytes)
        oldest = popfirst!(cache.heavy_order.keys)
        H_old = pop!(cache.heavy_store, oldest, nothing)
        H_old === nothing || (cache.heavy_bytes -= melitz_heavy_bytes(H_old))
        n_evicted += 1
    end
    return n_evicted
end

"""
    melitz_exact_cache_get(cache, key, ctx, U=nothing; obj=nothing)
        -> Union{Nothing,Tuple{Float64,Vector{Float64},Int,Matrix{Float64}}}

Looks up `key` in `cache`, applying the stale-context guard (see `MelitzExactPointCache`'s
own docstring): a compact hit whose stored fingerprint disagrees with
`melitz_context_fingerprint(ctx, U)` is dropped and treated as `nothing`. A genuine compact
hit then tries the heavy tier; on a heavy MISS, if `obj` is supplied, `H` is recomputed
directly from `key`/`obj.U`/`ctx` (no KNITRO) and reinserted; otherwise the lookup degrades
to `nothing` (a full miss). A genuine full hit moves both tiers to MRU position.
"""
function melitz_exact_cache_get(cache::MelitzExactPointCache, key::Vector{Float64}, ctx,
                                 U::Union{Nothing,AbstractMatrix}=nothing; obj=nothing)
    hit = get(cache.store, key, nothing)
    hit === nothing && return nothing
    Delta_hit, x_hit, nStatus_hit, fp = hit
    if fp != melitz_context_fingerprint(ctx, U)
        delete!(cache.store, key)
        melitz_lru_forget!(cache.order, key)
        delete!(cache.heavy_store, key)
        melitz_lru_forget!(cache.heavy_order, key)
        return nothing
    end
    melitz_lru_touch!(cache.order, key)

    H_hit = get(cache.heavy_store, key, nothing)
    if H_hit === nothing
        obj === nothing && return nothing   # compact hit, heavy miss, no way to recompute: full miss
        # 2026-07-26 production-port session: melitz_heavy_recompute (cc_bundle.jl) dispatches
        # per bundle type -- dense rebuild via obj.moments! (legacy), or a cheap
        # (no-KNITRO) operator re-equilibration + snapshot for MelitzCCBundle.
        H_hit = melitz_heavy_recompute(obj, key, ctx)
        cache.heavy_recomputes += 1
        cache.heavy_store[key] = H_hit
        cache.heavy_bytes += melitz_heavy_bytes(H_hit)
        melitz_lru_touch!(cache.heavy_order, key)
        cache.heavy_evictions += melitz_heavy_evict_until!(cache)
    else
        melitz_lru_touch!(cache.heavy_order, key)
    end
    return (Delta_hit, x_hit, nStatus_hit, H_hit)
end

"""
    melitz_exact_cache_insert!(cache, key, Delta, x, nStatus, H, ctx, U=nothing) -> nothing

Inserts/overwrites `key` in BOTH tiers, moves both to MRU, and evicts LRU entries beyond
each tier's own capacity independently (`cache.max_size` for the compact tier,
`cache.heavy_max_size`/`cache.heavy_max_bytes` for the heavy tier).
"""
function melitz_exact_cache_insert!(cache::MelitzExactPointCache, key::Vector{Float64},
                                     Delta::Float64, x::Vector{Float64}, nStatus::Int,
                                     H, ctx, U::Union{Nothing,AbstractMatrix}=nothing)
    cache.store[key] = (Delta, x, nStatus, melitz_context_fingerprint(ctx, U))
    melitz_lru_touch!(cache.order, key)
    cache.evictions += melitz_lru_evict_until!(cache.order, cache.store, cache.max_size)

    haskey(cache.heavy_store, key) && (cache.heavy_bytes -= melitz_heavy_bytes(cache.heavy_store[key]))
    cache.heavy_store[key] = H
    cache.heavy_bytes += melitz_heavy_bytes(H)
    melitz_lru_touch!(cache.heavy_order, key)
    cache.heavy_evictions += melitz_heavy_evict_until!(cache)
    return nothing
end

"""
    melitz_build_finite_delta_callbacks(obj, ctx, delta, find_smallest;
        delta_evaluation_cap=10.0,
        n_live_candidates_tracked=5, exact_cache=nothing) -> NamedTuple

Section 6/7: factors the finite-delta outer NLP's combined callback pair (objective +
divergence-budget + cutoff constraints, Sections 3.1/4.1/5.2) out of
`solve_melitz_finite_delta_bound` so a SEPARATE fixed-point test driver
(`melitz_fixed_point_probe`) can register the EXACT SAME production `cb_F!`/`cb_G!`
closures against a degenerate (0-degree-of-freedom) KNITRO problem -- Section 6's own
requirement ("these tests must pass through the exact production combined callback and
registered bounds, not merely call helper functions") -- rather than duplicating this
logic a second time, which would risk exactly the production/test drift Section 6 warns
against.

2026-07-24 evaluation-cap-correction session (governing prompt Sections 3-4): `delta`
remains what it always was -- the OUTER budget, used ONLY to form the dimensionless
constraint row `c(theta) = DeltaStar(theta)/delta` from a GENUINELY-solved `FiniteSolved`
result (Case A, `inner_screening.jl`), whether that solved value is within or over this
budget. `delta_evaluation_cap` (new, default `10.0` -- the governing prompt's own initial
production candidate) is the SEPARATE threshold that gates early-abort inside the routine
inner evaluation (`inner_solve_verified_or_fail` below, and threaded down into
`melitz_classified_inner_solve`/`build_melitz_implicit_bundle`'s `lower_limit`) -- `delta`
itself plays NO role in whether an inner solve is aborted early, only in how a solved
value is scaled for the outer constraint.

For an `AboveEvaluationCap`/`InfiniteDeltaCertified` trial point, `cb_F!`/`cb_G!` (below)
install a FIXED SENTINEL constraint value `delta_evaluation_cap/delta` (a CONSTANT,
identical for every such point, independent of `theta`/solve path/iteration/screen) with a
ZERO gradient, reported to KNITRO as an ORDINARY SUCCESSFUL evaluation -- not an eval-error.
This is deliberately analogous to the pre-existing Ricardian model's own `lower_limit=-50`
convention (`cc_algo/ccOuter.jl`/`ccInner.jl`): a fixed, numerically-motivated threshold
beyond which the model is treated as certainly-bad, reported as a real (if extreme)
constraint violation -- giving the outer search actual magnitude/direction information
rather than a blind eval-error backtrack. This repo's own prior (pre-this-session) work
found empirically that switching FROM eval-errors TO a real finite value here "measurably
changed exploration (a real, evolving trajectory rather than backtrack-to-near-zero)" --
motivating this choice over a bare eval-error.

This is NOT a reintroduction of the original bug. The original bug reported the
CERTIFICATE's own value (`result.certified_lower_bound`, which varies depending on which
screen/iteration happened to trip -- two points with the SAME true, possibly-infinite
`DeltaStar` could report different numbers) paired with the certificate's own fixed-dual
gradient AT THAT ARBITRARY point -- a real derivative of an unstable, path-dependent
quantity, mismatched with the value it accompanied. Here, the SAME constant is reported for
every point in this bucket, and the gradient (zero) is the EXACT, self-consistent derivative
of "always report this constant" -- value and gradient agree with each other, and neither
claims to know anything about the true `DeltaStar(theta)` beyond "this point is at least as
bad as the cap."

Only `NumericalFailure` (no certificate of any kind -- not even a lower bound) remains a
genuine eval-error: the callback throws a `DomainError`, caught by KNITRO.jl's own
`_try_catch_handler` and converted to a proper evaluation-error return code, matching the
pre-existing Ricardian convention for genuinely-unresolved failures.

Returns a `NamedTuple` `(cb_F!, cb_G!, live_candidates, n_inner_eval_failures,
signed_objective)`. `live_candidates`/`n_inner_eval_failures` are mutated in place by the
callbacks as KNITRO calls them -- the caller reads them AFTER `KN_solve` returns.
`n_inner_eval_failures` counts every trial point whose `DeltaStar` was NOT resolved to an
exact value (`AboveEvaluationCap`+`InfiniteDeltaCertified`+`NumericalFailure`) -- as of this
session, only the `NumericalFailure` subset of that count corresponds to an actual KNITRO
evaluation error; the other two are successful (sentinel) evaluations.
"""
function melitz_build_finite_delta_callbacks(obj, ctx, delta::Float64, find_smallest::Bool;
                                              delta_evaluation_cap::Float64=10.0,
                                              n_live_candidates_tracked::Int=5,
                                              cutoff_constraint_backend::Symbol=:nonlinear_reference,
                                              on_inner_result=nothing,
                                              dual_polish_screen::Bool=false,
                                              dual_polish_steps::Int=3,
                                              origin_block_screen::Bool=false,
                                              screen_order::Symbol=:A,
                                              warm_start_source::Symbol=:previous,
                                              exact_cache::Union{Nothing,MelitzExactPointCache}=nothing,
                                              dual_bank_max_size::Int=8,
                                              gradient_backend::Symbol=:auto,
                                              h::Real=1e-4,
                                              divergence_constraint_scaling::Symbol=:dimensionless,
                                              objective_scale::Union{Nothing,Real}=nothing)
    cutoff_constraint_backend in (:linear, :nonlinear_reference) || throw(ArgumentError(
        "cutoff_constraint_backend must be :linear or :nonlinear_reference, got $cutoff_constraint_backend"))
    divergence_constraint_scaling in (:dimensionless, :legacy_1e10) || throw(ArgumentError(
        "melitz_build_finite_delta_callbacks: divergence_constraint_scaling must be :dimensionless " *
        "or :legacy_1e10, got $divergence_constraint_scaling"))
    # 2026-07-26 closure session (governing prompt Phase 5): `:dimensionless`
    # (c_delta(theta)=DeltaStar(theta)/delta<=1) is this codebase's OWN production default
    # since the 2026-07-23 correctness-repair session (this file's own header, "OPAQUE 1e10
    # CONSTRAINT SCALING") -- NOT a new mode introduced this session. `:legacy_1e10`
    # reproduces the OLD, pre-2026-07-23 registration (`1e10*Delta(theta) <= 1e10*delta`) for
    # direct side-by-side comparison/conditioning study only; it is never the default and this
    # session does not change what any existing caller gets by omitting this kwarg.
    #
    # The shared functor's raw `constr[1]`/theta-gradient are ALWAYS `1e10*DeltaStar(theta)`/
    # `d(1e10*DeltaStar)/dtheta` (an internal cc_algo-shared convention, unaffected by this
    # kwarg). Both modes divide that SAME raw pair by one `divisor`, so the registered
    # constraint/bound/sentinel are always `raw/divisor <= (1e10*delta)/divisor` -- an
    # IDENTICAL feasible set and search direction for any `divisor`, only the numeric scale
    # differs (`divisor=1e10*delta` -> `:dimensionless`'s `<=1`; `divisor=1` -> `:legacy_1e10`'s
    # `<=1e10*delta`).
    divergence_divisor = divergence_constraint_scaling == :dimensionless ? (1e10 * delta) : 1.0
    divergence_bound = (1e10 * delta) / divergence_divisor
    divergence_sentinel = (1e10 * delta_evaluation_cap) / divergence_divisor
    # 2026-07-26 production-port session: resolve `gradient_backend=:auto` HERE using the
    # SAME rule `build_melitz_implicit_bundle` used to build `obj` (matrix-free bundle, or a
    # sorted-tail context present on `ctx`, -> the sorted direct family; else the plain direct
    # family) -- `obj` was just constructed from this exact `ctx` by the caller
    # (solve_melitz_finite_delta_bound/melitz_fixed_point_probe), so the two resolutions
    # necessarily agree. Without this, `:auto` would silently fall through to `nothing` below
    # and reintroduce the legacy jac_h theta-branch even though `obj` itself is matrix-free/
    # sorted-ready.
    resolved_gradient_backend = gradient_backend != :auto ? gradient_backend :
        (obj isa MelitzCCBundle || get(ctx, :sorted_tail_ctx, nothing) !== nothing) ?
            melitz_resolve_gradient_backend(MelitzBackendConfig(inner_backend=:matrix_free), ctx.D) :
            melitz_resolve_gradient_backend(MelitzBackendConfig(inner_backend=:dense_reference), ctx.D)
    # ADDITIVE (continuation4, Section 4): when gradient_backend is one of the two "direct"
    # backends, cb_G! (below) calls this closure directly instead of the shared
    # PsiObjectiveBundleImplicit functor's own theta-branch (obj(x, dummy_g, theta; jac=...)),
    # which would otherwise touch jac_h. `nothing` (the ordinary case) leaves cb_G! exactly as
    # before -- purely additive, zero behavior change for every existing gradient_backend value.
    direct_gradient_fn = resolved_gradient_backend == :B_direct_argument_serial ? make_melitz_gradient_delta_direct_serial(h) :
                         resolved_gradient_backend == :B_direct_argument_parallel ? make_melitz_gradient_delta_direct_parallel(h) :
                         resolved_gradient_backend == :B_direct_argument_sorted_serial ? make_melitz_gradient_delta_direct_sorted_serial(h) :
                         resolved_gradient_backend == :B_direct_argument_sorted_parallel ? make_melitz_gradient_delta_direct_sorted_parallel(h) :
                         resolved_gradient_backend == :B_direct_argument_touched_row_serial ? make_melitz_gradient_delta_direct_touched_row_serial(h) :
                         nothing
    signed_objective(theta) = find_smallest ? theta[1] : -theta[1]
    # 2026-07-28 outer-search gamma-profile session (governing prompt Phase 1): `objective_scale`
    # is a Melitz-owned, purely KNITRO-facing divisor on evalResult.obj[1]/.objGrad[1] --
    # orthogonal to `var_scale`/`var_center` (2026-07-25, KN_set_var_scalings_all), which
    # rescales VARIABLES only. KNITRO's own variable scaling does not touch the objective/
    # gradient the callback returns (callback-facing x/obj/objGrad are always raw, confirmed
    # 2026-07-25/2026-07-27 audits) -- so with objective = raw theta[1] (linear, gradient
    # exactly +-1 in raw units) and a tiny variable scale s_g (e.g. 1e-4, this session's own
    # Phase 7 choice), KNITRO's own INTERNAL scaled-space gradient (raw objGrad .* xScaleFactors,
    # the chain rule KNITRO applies itself) is only +-s_g -- order 1e-4, not order 1 -- a
    # plausible, previously undiagnosed contributor to a scaled trust-region algorithm reporting
    # spurious xtol convergence at (or near) the starting point. `objective_scale` corrects this:
    # `evalResult.obj[1]`/`.objGrad[1]` are divided by it before being registered with KNITRO,
    # so the EFFECTIVE scaled-space objective gradient KNITRO sees is `+-var_scale[1]/objective_scale`
    # -- choosing `objective_scale ~ var_scale[1]` (e.g. 1e-4) restores an order-1 scaled gradient.
    # `nothing` (default) is an EXACT no-op (`obj_scale_divisor=1.0`), preserving every existing
    # caller's behavior byte-for-byte -- this is purely additive, opt-in.
    #
    # Deliberately NOT applied to `signed_objective` (used for live-candidate/incumbent
    # comparison, cold-verified-incumbent selection, and reported `g`) -- those must stay in
    # raw economic theta[1] units regardless of this KNITRO-facing scale, so a caller can never
    # observe a different ANSWER purely from choosing a different `objective_scale`, only a
    # different KNITRO SEARCH TRAJECTORY getting there.
    obj_scale_divisor = objective_scale === nothing ? 1.0 : Float64(objective_scale)
    scaled_objective(theta) = signed_objective(theta) / obj_scale_divisor
    live_candidates = MelitzOuterCandidate[]
    n_inner_eval_failures = Ref(0)
    n_fc_calls = Ref(0)
    n_ga_calls = Ref(0)

    # Addendum Sections 3-6: front-loaded screens + stored-dual bank + no-routine-cold-retry
    # typed classification (inner_screening.jl). `dual_bank` is fresh per outer KNITRO solve
    # (never shared across solves), seeded lazily as verified inner solves accumulate.
    # Continuation session (2026-07-23, memory-scalability): `dual_bank_max_size` (default 8,
    # matching the pre-existing `MelitzDualBank()` default exactly -- this kwarg is purely
    # additive) makes this tier's already-bounded capacity CONFIGURABLE from the outer-solve
    # entry points too (`MelitzDualBank` itself has always been bounded via `max_size`/
    # `policy` eviction, Phase I.6 -- this just threads the knob through one more layer).
    dual_bank = MelitzDualBank(dual_bank_max_size)
    n_infinite_delta_reject = Ref(0)
    n_above_cap_reject = Ref(0)
    n_numerical_failure_reject = Ref(0)
    n_inner_solved = Ref(0)

    # Section 5.1 (2026-07-23 continuation session): exact-point single-slot inner-solve
    # cache, modeled on the Ricardian `OuterEvalCache.ensure_inner!`
    # (`cc_algo/outer_eval_cache.jl` -- that file's own header documents the SAME
    # `eval_fcga=no` pattern eliminating "44% duplicate-inner-solve elimination, bit-
    # identical, 1.36x wall time at D=4" for the Ricardian model). `melitz_outer_finite_delta.opt`
    # sets `eval_fcga no` too, so KNITRO calls `cb_F!` then `cb_G!` SEPARATELY at the same
    # accepted trial `theta` for most outer iterates -- both currently call
    # `inner_solve_verified_or_fail` independently, solving the IDENTICAL inner CC dual
    # problem twice. This closure-local cache (freshly created per
    # `melitz_build_finite_delta_callbacks` call, i.e. per outer KNITRO solve -- never
    # shared across solves, matching the Ricardian scoping discipline) reuses the verified
    # solution when `theta` recurs EXACTLY (value equality, no rounding). Per the governing
    # prompt's own Section 5.1/13 acceptance criteria: a FAILED inner solve is never cached
    # (`inner_solve_verified_or_fail` only updates this on the branch that is about to
    # `return`, i.e. a verified `nStatus`).
    #
    # CORRECTNESS FIX (caught live by the existing "Section 7: A/B/A repeated evaluation"
    # test, which this cache's first version broke): `obj.H` is a SEPARATE mutable field
    # that `inner_loop_internal` overwrites via `obj.moments!` BEFORE it even knows whether
    # the solve will succeed -- so an intervening FAILED solve at a different theta (e.g.
    # the A/B/A test's `theta_B`) still clobbers `obj.H` with `G(theta_B)`, even though it
    # correctly never updates THIS cache. A subsequent cache HIT back at `theta_A` would
    # then return the correct cached `(objSol, x, nStatus)` but leave `obj.H` holding the
    # WRONG (theta_B's) moment matrix -- silently corrupting every downstream consumer that
    # reads `obj.H` directly rather than through this cache's return value: the raw functor
    # calls in `cb_F!`/`cb_G!` (`obj(x, constr=...)`, `obj(x, g, theta; jac=...)`, both of
    # which read `H[:,2:end]` internally) and `register_live_candidate!`'s
    # `G_precomputed = select_G_from_H(obj, obj.H)` reuse (Section 4 above). Fix: cache a
    # COPY of `obj.H` alongside the dual solution, and restore it into `obj.H` on every hit,
    # before returning -- a ~`W*(d+2)*8` byte copy (trivial vs. a KNITRO solve) that keeps
    # `obj.H` and this cache's logical state consistent regardless of what happened at any
    # OTHER theta in between.
    # Section 4.2 (this continuation session): `exact_cache` defaults to a fresh, empty,
    # call-local `MelitzExactPointCache` when the caller passes nothing -- reproducing the
    # prior single-slot behavior's own scoping exactly (this specific call's own cache,
    # never shared). A caller wanting cross-delta reuse passes in the SAME cache object
    # across multiple `solve_melitz_finite_delta_bound`/`melitz_build_finite_delta_callbacks`
    # calls instead.
    exact_cache = exact_cache === nothing ? MelitzExactPointCache() : exact_cache
    n_exact_cache_hits = Ref(0)
    n_exact_cache_misses = Ref(0)

    function register_live_candidate!(theta::Vector{Float64}, Delta_val::Float64,
                                       x::Vector{Float64}, nStatus::Integer)
        # Section 4 (2026-07-23 continuation session): `obj.H`'s G columns were populated
        # by the `inner_loop_internal` call `inner_solve_verified_or_fail` JUST made at this
        # exact `theta` (the caller, cb_F!, calls this immediately afterward with nothing in
        # between that touches `obj.H` -- the intervening raw `obj(x, constr=local_c)` call
        # passes no `theta`, so it reads the already-fixed H without rebuilding it). Reusing
        # it here avoids a second, otherwise-identical O(W*(D^2+1)) moment build inside
        # `evaluate_melitz_delta_from_solution` -> `melitz_recover_lfd_from_solution` --
        # this WAS the dominant cost of `fc_candidate_registration` (~65-75ms/call,
        # `docs/melitz_optimization_report_2026-07-23.md` Section A.3).
        # melitz_bundle_current_G (cc_bundle.jl) returns the dense G view for the legacy
        # bundles (unchanged), or `nothing` for MelitzCCBundle -- a fully supported value for
        # `G_precomputed` (the matrix-free `melitz_recover_lfd_from_solution` method ignores
        # it entirely, always using mul_G!/mul_Gt! on the operator already at this theta).
        G_now = melitz_bundle_current_G(obj)
        # Governing prompt Phase 2 (2026-07-XX outer-search session): `full_equilibrium_check=
        # false` -- this LIVE per-trial registration path only ever needs `gravity_feasible`
        # (via `melitz_classify_outer_feasibility`, called immediately below), which reads
        # ONLY `gravity_residual_A`/`gravity_residual_f` from the returned `.equilibrium_check`
        # -- both computed identically either way (see `evaluate_melitz_delta_from_solution`'s
        # own comment). Skips ~99.7% of a real-D20 FC's wall time (confirmed live,
        # `docs/melitz_outer_search_scaling_and_profile_2026-07-XX.md` Phase 2) with NO change
        # to which points get classified outer-feasible or registered as live candidates. The
        # eventual COLD-VERIFIED incumbent this outer solve reports is always re-derived from
        # scratch via `evaluate_melitz_delta(...; cold=true)` (below, unaffected by this kwarg,
        # full diagnostic detail) -- never this live registration's own (deliberately partial)
        # `check`.
        r = evaluate_melitz_delta_from_solution(theta, ctx, obj, Delta_val, x, nStatus;
            G_precomputed=G_now, full_equilibrium_check=false)
        cls = melitz_classify_outer_feasibility(r, delta)
        cls.outer_feasible || return nothing
        push!(live_candidates, MelitzOuterCandidate(signed_objective(theta), r, cls, :live))
        sort!(live_candidates; by=c -> c.objective)
        while length(live_candidates) > n_live_candidates_tracked
            pop!(live_candidates)
        end
        return nothing
    end

    # 2026-07-24 evaluation-cap-correction session, SECOND revision (user review):
    # `AboveEvaluationCap`/`InfiniteDeltaCertified` report a FIXED SENTINEL constraint value
    # `delta_evaluation_cap/delta` (a CONSTANT, independent of `theta`/solve path/iteration)
    # with a ZERO gradient, as an ORDINARY SUCCESSFUL evaluation -- NOT an eval-error. This
    # is deliberately analogous to the pre-existing Ricardian `lower_limit=-50` convention
    # (`cc_algo/ccOuter.jl`/`ccInner.jl`): a fixed, numerically-motivated threshold beyond
    # which the model is treated as certainly-bad, reported to KNITRO as a real (if extreme)
    # constraint violation, giving the outer search actual magnitude/direction information
    # rather than a blind eval-error backtrack -- this repo's own prior session found
    # empirically that switching FROM eval-errors TO a real finite value "measurably changed
    # exploration (a real, evolving trajectory rather than backtrack-to-near-zero)."
    #
    # This is NOT the original bug. The original bug reported `result.certified_lower_bound`
    # -- a value that depends on WHICH screen/iteration happened to trip, so two points with
    # the SAME true (possibly infinite) `DeltaStar` could report different numbers -- paired
    # with the fixed-dual gradient AT THAT ARBITRARY certifying point, a real derivative of
    # an unstable, path-dependent quantity. The fix here reports the SAME constant for every
    # point in this bucket, with a gradient (zero) that is the EXACT, self-consistent
    # derivative of "always report this constant" -- value and gradient agree with each
    # other and with nothing else, unlike the original bug's mismatched pairing.
    #
    # `NumericalFailure` remains a genuine KNITRO evaluation failure -- thrown as a
    # `DomainError`, caught by KNITRO.jl's own `_try_catch_handler` and converted to a proper
    # evaluation-error return code (KN_RC_EVAL_ERR), no routine cold retry (addendum Section
    # 1, unaffected). This is the ONE case with NO certificate of any kind, not even a lower
    # bound -- matching the pre-existing Ricardian convention for genuinely-unresolved
    # failures (`full_aod_diag/d4_exact/c9_phase8_d20_pilot.jl`).
    function inner_solve_verified_or_fail(theta::AbstractVector)
        # Section 5.1/4.2: exact-point cache check -- before touching KNITRO at all. Restores
        # `obj.H` (see the correctness-fix comment above this closure's cache declaration)
        # so every downstream reader of `obj.H` sees state consistent with `theta`, exactly
        # as if a real solve had just run at this point. `key` uses a concrete `Vector{Float64}`
        # (not the possibly-view `theta` itself) so Dict hashing/equality is well-defined and
        # consistent between the lookup here and the insert below. Cache entries are ONLY ever
        # inserted from a verified FiniteSolved result (below), so a hit is always `:solved`.
        key = Vector{Float64}(theta)
        hit = melitz_exact_cache_get(exact_cache, key, ctx, obj.U; obj=obj)
        if hit !== nothing
            n_exact_cache_hits[] += 1
            melitz_record_seconds_outcome!(:inner_solve, :cache_hit, 0.0)
            Delta_hit, x_hit, nStatus_hit, H_hit = hit
            # melitz_heavy_restore! (cc_bundle.jl) dispatches: obj.H .= H_hit for the legacy
            # bundles (unchanged), or a zero-rebuild operator-field restore (+ MELITZ_FC_TO_GA_
            # CACHE_HITS increment) for MelitzCCBundle -- this IS the FC-to-GA state-reuse path
            # (Phase 3.1/9): a cb_G! call at a theta cb_F! JUST solved hits this branch.
            melitz_heavy_restore!(obj, H_hit)
            MELITZ_EXACT_POINT_CACHE_HITS[] += 1
            return (Delta_hit, x_hit, nStatus_hit, :solved)
        end
        n_exact_cache_misses[] += 1

        # Addendum Section 3/6, CORRECTED this session: `melitz_classified_inner_solve`
        # (inner_screening.jl) now takes `delta_evaluation_cap`, NEVER `delta` -- the routine
        # inner evaluation solves fully for every finite value up to the cap, regardless of
        # the outer budget (governing prompt Section 3). Front-loaded range + stored-dual
        # screens run BEFORE attempting KNITRO; exactly ONE KNITRO attempt (no cold retry) if
        # neither screen rejects.
        t0 = time_ns()
        result = melitz_classified_inner_solve(obj, theta, ctx; delta_evaluation_cap=delta_evaluation_cap,
            bank=dual_bank, on_result=on_inner_result, dual_polish_screen=dual_polish_screen,
            dual_polish_steps=dual_polish_steps, origin_block_screen=origin_block_screen,
            screen_order=screen_order, warm_start_source=warm_start_source)
        elapsed = (time_ns() - t0) / 1e9

        if result isa FiniteSolved
            n_inner_solved[] += 1
            melitz_record_seconds_outcome!(:inner_solve, :warm_success, elapsed)
            # Section 5.1/4.2: cache ONLY a verified result. `obj.H` is snapshotted too (see
            # the correctness-fix comment above) -- it was JUST populated at this exact
            # `theta` by `melitz_classified_inner_solve`. Keyed into the SHARED `exact_cache`
            # (Section 4.2), so a later call passed the same cache object -- even at a
            # DIFFERENT outer `delta`/`delta_evaluation_cap` -- can reuse this entry
            # (`DeltaStar(theta)` does not depend on the outer budget OR the evaluation cap).
            @melitz_profile :fc_cache_insert melitz_exact_cache_insert!(exact_cache, key, result.Delta, result.x, result.nStatus,
                melitz_heavy_snapshot(obj), ctx, obj.U)
            return (result.Delta, result.x, result.nStatus, :solved)
        elseif result isa InfiniteDeltaCertified
            n_infinite_delta_reject[] += 1
            n_inner_eval_failures[] += 1
            melitz_record_seconds_outcome!(:inner_solve, :infinite_delta_certified, elapsed)
            # A PROVEN Delta*(theta)=+infinity is, if anything, an even STRONGER certificate
            # than AboveEvaluationCap -- reported via the SAME fixed-sentinel path (below),
            # not thrown. `x`/`nStatus` are unused placeholders on this branch (`kind` alone
            # tells cb_F!/cb_G! what to do; no dual point exists for this certificate at all).
            return (NaN, Float64[], -1, :certified_bad)
        elseif result isa AboveEvaluationCap
            n_above_cap_reject[] += 1
            n_inner_eval_failures[] += 1
            melitz_record_seconds_outcome!(:inner_solve, :above_evaluation_cap, elapsed)
            # `result.certified_lower_bound`/`result.x` are DELIBERATELY NOT propagated here
            # -- cb_F!/cb_G! install the SAME fixed sentinel for every `:certified_bad` point
            # regardless of the exact certificate level, never this path-dependent value.
            return (NaN, Float64[], -1, :certified_bad)
        else   # NumericalFailure
            n_numerical_failure_reject[] += 1
            n_inner_eval_failures[] += 1
            melitz_record_seconds_outcome!(:inner_solve, :numerical_failure, elapsed)
            throw(DomainError(theta[1],
                "melitz finite-delta outer callback: NumericalFailure -- inner CC dual " *
                "solve returned nStatus=$(result.nStatus) with no certificate obtained (not a " *
                "finite optimum, not an evaluation-cap certificate -- possibly a routine " *
                "inner time/iteration cap reached with no certificate already in hand, " *
                "governing prompt Section 7), no routine cold retry (addendum Section 1) -- " *
                "rejecting this trial point, no DeltaStar value invented"))
        end
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        # Section 3.2 (2026-07-23 continuation session): the ENTIRE callback body is now
        # wrapped in try/finally so `:fc_total_*` records real elapsed wall time whether
        # the call completes normally OR throws (the eval-error convention this callback
        # itself uses, Section 5.2) -- the OLD version only recorded `:fc_total` via a
        # `melitz_record_seconds!` call at the very END of the function body, so any call
        # that threw (a genuine inner-solve failure surviving the cold retry) recorded
        # ZERO time, even though real wall-clock work happened before the throw. This was
        # the single largest documented instrumentation gap
        # (`docs/melitz_optimization_report_2026-07-23.md` Section B/H).
        t_fc0 = time_ns()
        n_fc_calls[] += 1
        outcome = :callback_success
        try
            theta = collect(evalRequest.x)
            objSol, x, nStatus, kind = inner_solve_verified_or_fail(theta)

            # Section 3.1: the outer objective is ALWAYS the finite, deterministic gamma
            # coordinate -- never the inner solve's own return value or a failure sentinel.
            # 2026-07-28 session (Phase 1): registered in KNITRO-facing SCALED units
            # (scaled_objective = signed_objective/obj_scale_divisor); `obj_scale_divisor=1.0`
            # (objective_scale=nothing, the default) makes this byte-identical to the prior
            # `signed_objective(theta)` registration.
            evalResult.obj[1] = scaled_objective(theta)

            # `kind == :certified_bad` (AboveEvaluationCap/InfiniteDeltaCertified):
            # install the FIXED sentinel `delta_evaluation_cap/delta` -- the SAME constant
            # for every such point, regardless of the exact certificate level (never
            # `result.certified_lower_bound`, which is path-dependent -- see
            # `inner_solve_verified_or_fail`'s own comment for the full reasoning). Reported
            # as an ORDINARY successful evaluation (return 0), not an eval-error.
            if kind == :certified_bad
                evalResult.c[1] = divergence_sentinel
                if cutoff_constraint_backend == :nonlinear_reference
                    g_d, g_e = @melitz_profile :fc_cutoff_nonlinear melitz_cutoff_constraints_at(theta, obj.γ)
                    nd = length(g_d)
                    evalResult.c[2:1+nd] .= g_d
                    evalResult.c[2+nd:end] .= g_e
                end
                return 0
            end

            # `kind == :solved` from here on: `inner_solve_verified_or_fail` throws ONLY for
            # `NumericalFailure` (no certificate of any kind) -- every other outcome is
            # handled above. `x` is the GENUINE optimum -- `obj(x, constr=local_c)` computes
            # `1e10*(-f(x;obj.H))` under the CURRENTLY-loaded `obj.H` (already fresh at
            # `theta`), which at the true optimum equals `1e10*DeltaStar(theta)` exactly
            # (Section 18).
            local_c = zeros(1)
            obj(x, constr=local_c)
            Delta_theta = local_c[1] / 1e10
            # Section 4.1: c_delta(theta) = DeltaStar(theta)/delta <= 1 -- dimensionless, O(1)
            # at the budget boundary regardless of delta's own scale. This is now the TRUE,
            # fully-optimized DeltaStar(theta) at every point that reaches here -- including
            # points genuinely OVER the outer budget (Delta_theta/delta > 1) but still below
            # delta_evaluation_cap (governing prompt Case A: these are FiniteSolved, not
            # intercepted early) -- giving KNITRO's own line search real, non-path-dependent
            # magnitude/direction information at every evaluated point, on-budget or not.
            # (governing prompt Phase 5: generalized to `local_c[1]/divergence_divisor` --
            # `:dimensionless` reduces to the ORIGINAL `Delta_theta/delta` exactly, since
            # `local_c[1]==1e10*Delta_theta` and `divergence_divisor==1e10*delta` in that mode.)
            evalResult.c[1] = local_c[1] / divergence_divisor

            # Section 3.3/4 (backend comparison): under :linear, the D+D*(D-1) cutoff rows are
            # NOT evaluated here at all -- they are registered as true KNITRO linear
            # constraints (constant coefficients, no per-iterate cost) by the caller
            # (`solve_melitz_finite_delta_bound`/`melitz_fixed_point_probe`), and this
            # callback's own `evalResult.c` is sized to exactly 1 (the divergence row only).
            # Under :nonlinear_reference (the OLD, trusted-comparison-only default), the
            # cutoff rows are still evaluated through the generic nonlinear callback exactly as
            # before.
            if cutoff_constraint_backend == :nonlinear_reference
                g_d, g_e = @melitz_profile :fc_cutoff_nonlinear melitz_cutoff_constraints_at(theta, obj.γ)
                nd = length(g_d)
                evalResult.c[2:1+nd] .= g_d
                evalResult.c[2+nd:end] .= g_e
            end

            @melitz_profile :fc_candidate_registration register_live_candidate!(theta, Delta_theta, collect(x), nStatus)
            return 0
        catch e
            outcome = :callback_eval_error
            rethrow(e)
        finally
            melitz_record_seconds_outcome!(:fc_total, outcome, (time_ns() - t_fc0) / 1e9)
        end
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        t_ga0 = time_ns()
        n_ga_calls[] += 1
        outcome = :callback_success
        try
            theta = collect(evalRequest.x)
            n_ = length(theta)
            objSol, x, nStatus, kind = inner_solve_verified_or_fail(theta)

            # Section 3.1: d(±theta[1])/dtheta -- exact, trivial, independent of the inner
            # solve (which is still needed below, for the constraint Jacobian only).
            # 2026-07-28 session (Phase 1): divided by the SAME obj_scale_divisor as
            # evalResult.obj[1] above -- consistent value/gradient scaling, exact no-op at
            # obj_scale_divisor=1.0.
            evalResult.objGrad .= 0.0
            evalResult.objGrad[1] = (find_smallest ? 1.0 : -1.0) / obj_scale_divisor

            # `kind == :certified_bad`: the SAME fixed sentinel's own gradient is EXACTLY
            # ZERO -- the true derivative of "always report the constant
            # delta_evaluation_cap/delta," self-consistent with cb_F!'s matching branch
            # (never the certificate's own fixed-dual gradient at an arbitrary, path-
            # dependent point).
            if kind == :certified_bad
                evalResult.jac[1:n_] .= 0.0
                if cutoff_constraint_backend == :nonlinear_reference
                    J_d, J_e = @melitz_profile :ga_cutoff_jacobian_nonlinear melitz_cutoff_constraint_jacobian(theta, obj.γ)
                    nd, ne = size(J_d, 1), size(J_e, 1)
                    @inbounds for kk in 1:nd
                        evalResult.jac[kk*n_+1:(kk+1)*n_] .= @view J_d[kk, :]
                    end
                    off = 1 + nd
                    @inbounds for kk in 1:ne
                        evalResult.jac[(off+kk-1)*n_+1:(off+kk)*n_] .= @view J_e[kk, :]
                    end
                end
                return 0
            end

            # `kind == :solved` from here on: `x` is the GENUINE optimum, so both gradient
            # paths below compute the EXACT derivative of the true `DeltaStar(theta)` via the
            # envelope theorem (no fixed-dual/suboptimal-point caveat applies anymore: the
            # prior session's `:budget_infeasible` branch, which computed this same formula
            # at a possibly-suboptimal certifying dual, no longer exists on this path -- see
            # `inner_solve_verified_or_fail`'s own comment).
            local_jac = zeros(n_)
            if direct_gradient_fn === nothing
                dummy_g = zeros(n_)
                @melitz_profile :ga_divergence_gradient obj(x, dummy_g, theta; jac=local_jac)   # local_jac == d(1e10*DeltaStar)/dtheta
            else
                # Section 4: direct fixed-dual gradient-vector backend -- computes
                # d(1e10*Delta)/dtheta directly (direct_gradient.jl), bypassing the shared
                # functor's theta-branch/jac_h entirely. `obj.H`/`x` are already the correct
                # base state at this `theta` (inner_solve_verified_or_fail just ensured it).
                @melitz_profile :ga_divergence_gradient direct_gradient_fn(local_jac, theta, ctx, obj, x)
            end
            evalResult.jac[1:n_] .= local_jac ./ divergence_divisor   # Section 4.2/Phase 5: same divisor as the value

            if cutoff_constraint_backend == :nonlinear_reference
                J_d, J_e = @melitz_profile :ga_cutoff_jacobian_nonlinear melitz_cutoff_constraint_jacobian(theta, obj.γ)
                nd, ne = size(J_d, 1), size(J_e, 1)
                @inbounds for kk in 1:nd
                    evalResult.jac[kk*n_+1:(kk+1)*n_] .= @view J_d[kk, :]
                end
                off = 1 + nd
                @inbounds for kk in 1:ne
                    evalResult.jac[(off+kk-1)*n_+1:(off+kk)*n_] .= @view J_e[kk, :]
                end
            end
            return 0
        catch e
            outcome = :callback_eval_error
            rethrow(e)
        finally
            melitz_record_seconds_outcome!(:ga_total, outcome, (time_ns() - t_ga0) / 1e9)
        end
    end

    return (cb_F! = cb_F!, cb_G! = cb_G!, live_candidates = live_candidates,
            n_inner_eval_failures = n_inner_eval_failures, signed_objective = signed_objective,
            cutoff_constraint_backend = cutoff_constraint_backend,
            delta_evaluation_cap = delta_evaluation_cap,
            divergence_constraint_scaling = divergence_constraint_scaling,
            divergence_constraint_upbnd = divergence_bound,
            n_fc_calls = n_fc_calls, n_ga_calls = n_ga_calls,
            n_exact_cache_hits = n_exact_cache_hits, n_exact_cache_misses = n_exact_cache_misses,
            exact_cache = exact_cache,
            dual_bank = dual_bank, n_infinite_delta_reject = n_infinite_delta_reject,
            n_above_cap_reject = n_above_cap_reject,
            n_numerical_failure_reject = n_numerical_failure_reject,
            n_inner_solved = n_inner_solved)
end

"""
    melitz_register_finite_delta_knitro_problem!(kc, ctx, cbset, xIndices, n, D;
        cutoff_constraint_backend=:nonlinear_reference) -> cIndices

Section 3.3/4: registers the `m = 1 + D + D*(D-1)` constraint block on an already-created
KNITRO problem `kc` (variables already added via `xIndices`), branching on
`cutoff_constraint_backend`:

  - `:nonlinear_reference` (the OLD, trusted-comparison-only path): every one of the `m`
    rows is registered against ONE combined eval callback (`cbset.cb_F!`/`cbset.cb_G!`),
    exactly as before this session -- the generic nonlinear FC/GA machinery evaluates the
    D+D*(D-1) deterministic cutoff rows fresh at every KNITRO iterate, even though they are
    affine in `theta_free` and never change value/Jacobian shape.
  - `:linear` (Section 3.3's fix): ONLY the divergence-budget row (`c_delta(theta) =
    Delta(theta)/delta <= 1`, genuinely nonlinear) is registered against the eval
    callback (`cIndices[1:1]`, so `evalResult.c`/`.jac` inside `cbset.cb_F!`/`cb_G!` are
    sized to exactly 1 row -- see that function's own `cutoff_constraint_backend` branch).
    The D+D*(D-1) cutoff rows are registered as TRUE KNITRO linear constraints
    (`KN_add_con_linear_struct`, constant coefficients from
    `build_melitz_affine_cutoff_system`) -- evaluated natively by KNITRO with NO per-
    iterate callback cost at all, not merely a cheaper callback. Uses the SAME single eval-
    callback CONTEXT as the old combined callback did (only now covering 1 row instead of
    `m`) -- deliberately avoiding Section 17.C's documented "two separate callback
    contexts" KNITRO crash, since linear constraints registered via
    `KN_add_con_linear_struct` are not a second callback context at all.

Returns `cIndices` (length `m`, row 1 = divergence, rows `2:end` = domestic/export cutoff
rows in `melitz_deterministic_cutoff_constraints`'s own order) and, when
`cutoff_constraint_backend==:linear`, also returns the `MelitzAffineCutoffSystem` used
(`nothing` under `:nonlinear_reference`) so a caller can report row-scaling diagnostics.
"""
function melitz_register_finite_delta_knitro_problem!(kc, ctx, cbset, xIndices, n::Int, D::Int,
                                                        obj_for_user_params;
                                                        cutoff_constraint_backend::Symbol=:nonlinear_reference)
    D2 = D * (D - 1)
    n_cutoff = D + D2
    m = 1 + n_cutoff
    cIndices = melitz_kn_add_cons!(kc, m)
    # Section 4.1: c_delta <= 1 under the (default) :dimensionless scaling, replacing the old
    # 1e10*delta magnitude. Governing prompt Phase 5: the bound now comes from `cbset` itself
    # (`melitz_build_finite_delta_callbacks`'s own `divergence_constraint_upbnd`), so the
    # registered bound always matches whatever scaling that function's callbacks actually use
    # -- `:legacy_1e10` registers `1e10*delta` here instead, an IDENTICAL feasible set, just
    # unscaled (see that function's own header comment for the full equivalence argument).
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], cbset.divergence_constraint_upbnd)

    cutoff_sys = nothing
    if cutoff_constraint_backend == :nonlinear_reference
        KNITRO.KN_set_con_lobnds(kc, n_cutoff, cIndices[2:end], zeros(n_cutoff))
        cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cbset.cb_F!)
        KNITRO.KN_set_cb_grad(kc, cb, cbset.cb_G!,
            jacIndexCons=repeat(cIndices, inner=n), jacIndexVars=repeat(xIndices, outer=m))
    else   # :linear
        cutoff_sys = build_melitz_affine_cutoff_system(ctx)
        # C*theta_free + b >= 0  <=>  C*theta_free >= -b  -- lower bound -b (scaled), no
        # upper bound (default +infinity, matching the old row's own one-sided sense).
        KNITRO.KN_set_con_lobnds(kc, n_cutoff, cIndices[2:end], -cutoff_sys.b)
        nnz = n_cutoff * n
        indexCons_lin = repeat(cIndices[2:end], inner=n)
        indexVars_lin = repeat(xIndices, outer=n_cutoff)
        coefs_lin = vec(permutedims(cutoff_sys.C))   # row-major flatten of C, matching (con,var) pair order
        KNITRO.KN_add_con_linear_struct(kc, nnz, indexCons_lin, indexVars_lin, coefs_lin)

        cb = KNITRO.KN_add_eval_callback(kc, true, [cIndices[1]], cbset.cb_F!)
        KNITRO.KN_set_cb_grad(kc, cb, cbset.cb_G!,
            jacIndexCons=fill(cIndices[1], n), jacIndexVars=xIndices)
    end
    KNITRO.KN_set_cb_user_params(kc, cb, obj_for_user_params)
    return cIndices, cutoff_sys
end

"""
    solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; delta, direction,
        gradient_backend=:B, h=1e-4, theta_box=2.0, n_live_candidates_tracked=5,
        outer_loop_opt=<default>) -> MelitzFiniteDeltaOuterResult

Section 3/4 (2026-07-23 correctness-repair session): solves
`minimize/maximize theta_free[1] s.t. Delta(theta)<=delta` and the Section 1.3 cutoff
inequalities, reusing `PsiObjectiveBundleImplicit` (constructed by
`build_melitz_implicit_bundle`) for the inner CC dual solve/envelope-gradient machinery,
but registering its OWN KNITRO problem/callbacks directly (see this file's header for why
`outer_loop`/`outer_loop_constraints!` are not reused for the outer NLP itself, unlike the
inner-bundle construction).

Three corrections vs. the prior (2026-07-23, pre-repair) version of this function, per
`docs/melitz_delta_star.md`'s governing correction:

  1. **Objective** (Section 3.1): `evalResult.obj[1] = find_smallest ? theta[1] :
     -theta[1]`, ALWAYS -- independent of the inner solve's own success/failure.
  2. **Constraint scaling** (Section 4.1): the divergence row is
     `c_delta(theta) = Delta(theta)/delta <= 1`, not the old `1e10*Delta(theta) <=
     1e10*delta` -- the SAME underlying `1e10*Delta` raw functor value, consistently
     rescaled (value AND Jacobian) by `1/(1e10*delta)`.
  3. **Inner failure handling** (Section 5): a genuine inner-solve failure (bad `nStatus`
     even after one cold retry) throws a `DomainError`, caught by KNITRO.jl's own callback
     wrapper and converted to a proper evaluation-error return code -- matching
     `full_aod_diag/d4_exact/c9_phase8_d20_pilot.jl`'s documented convention for the
     Ricardian model, not a hand-invented constraint value.

2026-07-24 evaluation-cap-correction session (fourth correction, layered on top of the three
above): `delta_evaluation_cap` (default `10.0`) is a NEW kwarg threaded straight into
`build_melitz_implicit_bundle`/`melitz_build_finite_delta_callbacks` -- see those functions'
own docstrings for the full Case A/B/C/D semantics this introduces.

2026-07-26 production-closure session (governing prompt Phase 1): `delta_evaluation_cap` is
ALWAYS active on this driver now, exactly (`lower_limit = -delta_evaluation_cap`, see
`build_melitz_implicit_bundle`'s docstring) -- there is no separate guard/margin kwarg to
omit (removed the same session, per direct user feedback: an earlier version of this fix
added a small additive margin on top of the cap, inherited from an unrelated older design
that genuinely needed one; once the threshold IS the evaluation cap, no margin is needed).
Prior to this session, calling this function with no cap-related kwargs at all left the cap
looking active (`delta_evaluation_cap=10.0`, the default) while `lower_limit` stayed
disabled -- confirmed live to cost a `13.5x` wall-clock regression and 31 spurious
`NumericalFailure` results in a real D=20 campaign
(`docs/melitz_production_fast_backend_2026-07-26.md` Section 5.5).

In one sentence: `delta`
no longer plays any role in whether an inner evaluation is aborted early (only
`delta_evaluation_cap` does), so a trial point with genuine `DeltaStar` between `delta` and
`delta_evaluation_cap` is now solved to a real optimum and returned to KNITRO as an
ordinary, informative, over-budget constraint value -- not intercepted and replaced with a
path-dependent certificate the way the pre-correction session's `BudgetInfeasible` handling
did. A point that is `AboveEvaluationCap` or `InfiniteDeltaCertified` is reported to outer
KNITRO as a FIXED sentinel constraint value `delta_evaluation_cap/delta` with a ZERO
gradient (an ordinary successful evaluation, not an eval-error) -- see
`melitz_build_finite_delta_callbacks`'s own docstring for the full reasoning (this constant,
self-consistent sentinel is NOT the original bug's path-dependent certificate value; only
`NumericalFailure`, the one case with no certificate of any kind, remains a genuine
eval-error).

Also implements Section 2's incumbent bookkeeping: `theta_init` is cold-evaluated and
installed as `initial_incumbent` BEFORE `KN_solve` runs (survives even a
zero/one-iteration KNITRO run, Section 2.1); every outer-feasible point evaluated during
the trajectory is classified via `evaluate_melitz_delta_from_solution` (NO extra KNITRO
solve -- reuses the dual `x` this SAME callback call already computed) and the best
`n_live_candidates_tracked` are kept (Section 2.2); at the end, live candidates are
cold-reverified best-first until one survives, falling back to `initial_incumbent`
(Section 2.2 steps 2-4).

`obj_inner`: the EXISTING `PsiObjectiveBundleDelta` bundle from `build_melitz_psi_bundle`
(shares `ctx`/reference draws with the new `PsiObjectiveBundleImplicit` this function
builds internally) -- used for every cold-verification call (initial incumbent, end-of-run
reverification).

`theta_box`: the outer decision vector has no natural box (the governing prompt imposes
none) -- a symmetric `theta_init .± theta_box` bound is set purely for KNITRO
well-posedness (an ARTIFICIAL bound, flagged per Section 8.F, not an economic
restriction). May be a scalar (uniform box, the original behavior) OR a length-`n` vector
(Section 9's restricted nuisance-coordinate searches: a `0.0` entry pins that coordinate
exactly at `theta_init`'s own value -- KNITRO sees `lobnd==upbnd` for that variable, the
same "zero degrees of freedom" mechanism `melitz_fixed_point_probe` already uses for every
coordinate -- while a nonzero entry frees that coordinate within the usual symmetric
range; both branches of the broadcasted `.-`/`.+` below already work for either shape, no
separate code path needed).

`exact_cache`/`eval_cache` (Section 4.2, this continuation session): both default `nothing`
(fresh, call-local, empty caches -- exactly the pre-existing behavior). Pass the SAME
`MelitzExactPointCache`/`MelitzDeltaEvalCache` object across multiple calls (e.g. a
continuation workflow solving `delta=1e-3` then `delta=1e-2` from the same `theta`
neighborhood) to reuse verified inner-solve/evaluation results across DIFFERENT outer
budgets -- valid because `Delta(theta)` (and everything derived from it: dual, LFD,
moments, equilibrium checks) is a function of `theta` alone, never of the outer budget
`delta` itself.

`external_incumbent` (2026-07-24 outer-benchmark-correction session, main prompt Section 4):
an optional, separately-known feasible `theta_free` point (e.g. the fixed-A/f scalar
profile's own verified boundary point) to be cold-evaluated and folded into the SAME
incumbent-selection logic as `theta_init`'s own initial incumbent and the trajectory's best
cold-reverified live candidate -- `cold_verified_incumbent` (the field callers already treat
as "the answer") is the argmin of SIGNED objective over every outer-feasible candidate among
these three, so the full flexible search can never be reported as worse than a
DELIBERATELY-passed known-good restricted incumbent, even if KNITRO's own trajectory never
rediscovers it (main prompt Section 4's explicit requirement). Default `nothing`: no
externally-supplied incumbent, byte-for-byte the pre-existing selection logic (argmin over
just the trajectory-derived and initial candidates).

`var_scale`/`var_center` (2026-07-25 scaled-KNITRO session): ADDITIVE, length-`n` vectors
passed straight to KNITRO's own native `KN_set_var_scalings_all(kc, var_scale, var_center)`
(`theta[i] = var_center[i] + var_scale[i]*y[i]`, KNITRO's own documented convention --
`include/knitro.h`'s own header comment on `KN_set_var_scalings`). Both default `nothing`:
no call is made at all, byte-for-byte the pre-existing unscaled behavior. Verified live
(standalone synthetic-NLP audit, 2026-07-25 session) that this is a PURE reparameterization
of KNITRO's own internal step/trust-region machinery -- callbacks (`cb_F!`/`cb_G!`, already
built by `melitz_build_finite_delta_callbacks` above) always receive/return `theta` in RAW
economic units regardless of `var_scale`/`var_center`, so NO change to any callback,
Jacobian, or the affine cutoff system's `C`/`b` registration (`melitz_register_finite_delta_knitro_problem!`)
is needed -- KNITRO performs the chain rule internally. `var_scale` entries must be strictly
positive (KNITRO's own convention: a non-positive entry silently disables scaling for that
one coordinate rather than erroring -- this function does not additionally validate that,
matching KNITRO's own documented behavior).
"""
function solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init::AbstractVector;
                                          delta::Real, direction::Symbol,
                                          delta_evaluation_cap::Real=10.0,
                                          gradient_backend::Symbol=:auto, h::Real=1e-4,
                                          theta_box::Union{Real,AbstractVector}=2.0,
                                          n_live_candidates_tracked::Int=5,
                                          cutoff_constraint_backend::Symbol=:nonlinear_reference,
                                          inner_loop_opt::AbstractString,
                                          outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta.opt"),
                                          on_inner_result=nothing,
                                          dual_polish_screen::Bool=false,
                                          dual_polish_steps::Int=3,
                                          origin_block_screen::Bool=false,
                                          screen_order::Symbol=:A,
                                          warm_start_source::Symbol=:previous,
                                          exact_cache::Union{Nothing,MelitzExactPointCache}=nothing,
                                          eval_cache::Union{Nothing,MelitzDeltaEvalCache}=nothing,
                                          dual_bank_max_size::Int=8,
                                          external_incumbent::Union{Nothing,AbstractVector}=nothing,
                                          var_scale::Union{Nothing,AbstractVector}=nothing,
                                          var_center::Union{Nothing,AbstractVector}=nothing,
                                          backend::Symbol=:auto_from_gradient_backend,
                                          forbid_dense_fallback::Bool=false,
                                          divergence_constraint_scaling::Symbol=:dimensionless,
                                          objective_scale::Union{Nothing,Real}=nothing)
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower"))
    t0 = time()
    find_smallest = direction == :upper   # minimize g for the upper GT bound, maximize for lower
    D = ctx.D
    n = length(theta_init)
    delta = Float64(delta)
    delta_evaluation_cap = Float64(delta_evaluation_cap)

    # 2026-07-24 evaluation-cap-correction session: `delta_evaluation_cap` is threaded to
    # `build_melitz_implicit_bundle` (never the outer budget `delta`) -- see that function's
    # docstring for why the two must not be conflated.
    #
    # 2026-07-26 production-closure session (governing prompt Phase 1, simplified same day
    # per direct user feedback removing the separate guard/margin entirely): `delta_evaluation_cap`
    # here is `::Real` (never `Union{Nothing,Real}`) -- this driver's cap is ALWAYS meant to be
    # active, by construction, never an opt-in a caller can forget. Combined with this
    # session's fix to `build_melitz_implicit_bundle` (a supplied `delta_evaluation_cap`
    # always activates `lower_limit = -delta_evaluation_cap`, exactly), simply calling this
    # function with NO cap-related kwargs at all -- the exact confirmed-live incident (Section
    # 5.5 of docs/melitz_production_fast_backend_2026-07-26.md: `delta_evaluation_cap=10.0`'s
    # own default, previously left disabled, 13.5x slower campaign, 31 spurious
    # `NumericalFailure`s) -- now yields an ACTIVE cap automatically. The assertion below is a
    # live, unconditional runtime guarantee of that fact (not merely a documentation claim): if
    # a future refactor ever reintroduces a silent disabled-cap path here, this fails loudly
    # the very first call, not just in a test file that might not be re-run.
    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_init; delta=delta,
        find_smallest=find_smallest, gradient_backend=gradient_backend, h=h,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt,
        delta_evaluation_cap=delta_evaluation_cap, backend=backend,
        forbid_dense_fallback=forbid_dense_fallback)
    @assert isfinite(obj.lower_limit) (
        "solve_melitz_finite_delta_bound: obj.lower_limit=$(obj.lower_limit) is not finite -- " *
        "the evaluation cap (delta_evaluation_cap=$delta_evaluation_cap) failed to activate. " *
        "This should be impossible after the 2026-07-26 production-closure fix; see " *
        "build_melitz_implicit_bundle's docstring.")

    signed_objective(theta) = find_smallest ? theta[1] : -theta[1]

    # ------------------------------------------------------------------------
    # Section 2.1: install the initial incumbent BEFORE KN_solve is ever called, from a
    # COLD evaluation of theta_init -- this must be returned even if KNITRO's own
    # trajectory never produces a verified point (Section 12's zero/one-iteration
    # regression test pins exactly this).
    # ------------------------------------------------------------------------
    # Section 4.2 (this continuation session): `eval_cache`, if given, lets this cold
    # evaluation reuse a verified `MelitzDeltaEvalResult` (Delta, dual, LFD, moments,
    # equilibrium checks -- the FULL state, not just the FC/GA path's lighter dual-only
    # `exact_cache`) computed by an EARLIER `solve_melitz_finite_delta_bound` call that
    # shared the same `eval_cache`/`obj_inner` -- valid across different `delta` values for
    # the same reason `exact_cache` is (`Delta(theta)` does not depend on the outer budget).
    # Default `nothing`: no caching, exactly the pre-existing behavior.
    initial_eval = evaluate_melitz_delta(collect(theta_init), ctx, obj_inner; cold=true, cache=eval_cache)
    initial_classification = melitz_classify_outer_feasibility(initial_eval, delta)
    initial_incumbent = initial_classification.outer_feasible ?
        MelitzOuterCandidate(signed_objective(theta_init), initial_eval, initial_classification, :initial) :
        nothing

    # Section 4 (this session): cold-evaluate the externally-supplied known-good incumbent
    # (if any) BEFORE KN_solve too, on the exact same footing as theta_init's own initial
    # incumbent -- see this function's own docstring.
    external_incumbent_candidate = nothing
    if external_incumbent !== nothing
        external_theta = collect(Float64.(external_incumbent))
        external_eval = evaluate_melitz_delta(external_theta, ctx, obj_inner; cold=true, cache=eval_cache)
        external_classification = melitz_classify_outer_feasibility(external_eval, delta)
        external_incumbent_candidate = external_classification.outer_feasible ?
            MelitzOuterCandidate(signed_objective(external_theta), external_eval, external_classification, :external) :
            nothing
    end

    # Section 6/7: the SAME callback pair a fixed-point test would register directly.
    # `on_inner_result`, if given, is a diagnostic hook (see inner_screening.jl) invoked
    # on every classified inner-solve outcome -- default `nothing`, zero risk to production
    # callers.
    cbset = melitz_build_finite_delta_callbacks(obj, ctx, delta, find_smallest;
        delta_evaluation_cap=delta_evaluation_cap,
        n_live_candidates_tracked=n_live_candidates_tracked,
        cutoff_constraint_backend=cutoff_constraint_backend, on_inner_result=on_inner_result,
        dual_polish_screen=dual_polish_screen, dual_polish_steps=dual_polish_steps,
        origin_block_screen=origin_block_screen, screen_order=screen_order,
        warm_start_source=warm_start_source, exact_cache=exact_cache,
        dual_bank_max_size=dual_bank_max_size, gradient_backend=gradient_backend, h=h,
        divergence_constraint_scaling=divergence_constraint_scaling, objective_scale=objective_scale)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = melitz_kn_add_vars!(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, collect(theta_init) .- theta_box)
    KNITRO.KN_set_var_upbnds_all(kc, collect(theta_init) .+ theta_box)
    KNITRO.KN_set_var_primal_init_values_all(kc, collect(theta_init))
    if var_scale !== nothing
        length(var_scale) == n || throw(ArgumentError("var_scale must have length n=$n, got $(length(var_scale))"))
        vc = var_center === nothing ? collect(Float64.(theta_init)) : collect(Float64.(var_center))
        length(vc) == n || throw(ArgumentError("var_center must have length n=$n, got $(length(vc))"))
        KNITRO.KN_set_var_scalings_all(kc, collect(Float64.(var_scale)), vc)
    end

    cIndices, cutoff_sys = melitz_register_finite_delta_knitro_problem!(kc, ctx, cbset, xIndices, n, D, obj;
        cutoff_constraint_backend=cutoff_constraint_backend)

    # 2026-07-27 continuation: Melitz-owned counter snapshot bracketing the solve, replacing a
    # prior unconditional `CounterfactualSensitivity.INNER_SOLVE_COUNT[]` reset/read that (a)
    # threw `UndefVarError` when `cc_algo` was not loaded and (b) was silently always-zero for
    # the matrix-free backend even when `cc_algo` WAS loaded (see run_diagnostics.jl header).
    counters_before = melitz_backend_counters_snapshot()
    KNITRO.KN_solve(kc)
    nStatus, _, theta_final_raw, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)
    counters_after = melitz_backend_counters_snapshot()
    diagnostics = melitz_run_diagnostics(cbset, counters_before, counters_after)
    solve_inner_count = diagnostics.inner_solve_attempts
    solve_infeas_count = diagnostics.inner_numerical_failure_reject

    theta_final = collect(theta_final_raw)
    terminal_eval = evaluate_melitz_delta(theta_final, ctx, obj_inner; cold=true, cache=eval_cache)
    terminal_classification = melitz_classify_outer_feasibility(terminal_eval, delta)

    # ------------------------------------------------------------------------
    # Section 2.2 steps 2-4: rank the live candidates best-first (smallest signed
    # objective), cold-reverify from a cleared warm start until one survives; fall back to
    # the initial Pareto incumbent if none does. Never restart/report from an unverified
    # terminal point (Section 10's own "never restart from an infeasible terminal point").
    # ------------------------------------------------------------------------
    live_candidates = cbset.live_candidates
    cold_verified_incumbent = nothing
    for cand in live_candidates
        cold_eval = evaluate_melitz_delta(cand.eval.theta_free, ctx, obj_inner; cold=true)
        cold_cls = melitz_classify_outer_feasibility(cold_eval, delta)
        if cold_cls.outer_feasible
            cold_verified_incumbent = MelitzOuterCandidate(signed_objective(cand.eval.theta_free),
                cold_eval, cold_cls, :cold_verified)
            break
        end
    end
    if cold_verified_incumbent === nothing
        cold_verified_incumbent = initial_incumbent
    end

    # Section 4 (this session): fold the externally-supplied known-good incumbent (if any)
    # into the SAME final selection -- argmin of signed objective over every outer-feasible
    # candidate in hand (trajectory-derived, theta_init's own initial point, and the external
    # one). This is what makes "the result returned by the full procedure must never be worse
    # than a known feasible incumbent" an ENFORCED property of this function's return value,
    # not merely something a caller has to remember to check afterward.
    all_candidates = filter(!isnothing, (cold_verified_incumbent, initial_incumbent, external_incumbent_candidate))
    cold_verified_incumbent = isempty(all_candidates) ? nothing :
        reduce((a, b) -> b.objective < a.objective ? b : a, all_candidates)

    best_live_incumbent = isempty(live_candidates) ? nothing : first(live_candidates)

    return MelitzFiniteDeltaOuterResult(collect(theta_init), delta, direction,
        gradient_backend, theta_final, terminal_eval, terminal_classification,
        initial_incumbent, best_live_incumbent, cold_verified_incumbent,
        nStatus, solve_inner_count, solve_infeas_count, cbset.n_inner_eval_failures[], time() - t0,
        cutoff_constraint_backend, cbset.n_fc_calls[], cbset.n_ga_calls[],
        delta_evaluation_cap,
        cbset.n_inner_solved[], cbset.n_infinite_delta_reject[],
        cbset.n_above_cap_reject[], cbset.n_numerical_failure_reject[], diagnostics)
end

# ============================================================================
# Section 6: fixed-point KNITRO integration tests. `melitz_fixed_point_probe` builds a
# degenerate (0-degree-of-freedom, var lobnd==upbnd) KNITRO problem around a single given
# `theta_probe` and registers the EXACT SAME production callback pair
# (`melitz_build_finite_delta_callbacks`) `solve_melitz_finite_delta_bound` itself uses --
# so `KN_solve` performs exactly one production-path evaluation, exercising the real
# registered objective/constraint/bounds/Jacobian wiring end to end, not a bypass helper
# call.
# ============================================================================

"""
    MelitzFixedPointProbeResult

Result of `melitz_fixed_point_probe`. `nStatus` is KNITRO's own overall solve status at
the (single, fixed) evaluation point. `eval_failed` is `true` iff the inner CC dual solve
failed even after the Section 5.2 cold retry (the callback threw, KNITRO caught it and
reported a `KN_RC_CALLBACK_ERR`/`KN_RC_EVAL_ERR`-family status: `-500` or `-502`) -- in
that case `obj_value`/`c` are `NaN`/empty, since KNITRO never recorded a real evaluation.
When `!eval_failed`, `obj_value` is the registered objective (Section 3.1: `±theta[1]`
exactly) and `c` is the full `m`-vector of registered constraint values (`c[1]` = the
Section 4.1 normalized `Delta(theta)/delta` divergence row, `c[2:end]` = the Section 1.3
deterministic cutoff rows). `c[1]`/`obj_value` are read directly off KNITRO
(`KN_get_obj_value`/`KN_get_con_values_all`). Under `cutoff_constraint_backend=:linear`,
`c[2:end]` is instead RECOMPUTED directly in Julia from `cutoff_sys.C*theta+cutoff_sys.b`
-- KNITRO's own post-solve `KN_get_con_values_all` for natively-registered linear rows was
found (Section 3.4) to be unreliable once this problem's callback has triggered a nested
KN_new/KN_solve/KN_free cycle (as it always does here, for the real CC inner solve), even
though solve-time constraint enforcement itself is unaffected (cross-checked against
KNITRO's own presolve-deduced-infeasibility message, which matched the true slack exactly).
`live_candidates` holds whatever the shared incumbent-tracking logic recorded for this one
evaluation (0 or 1 entries).
"""
struct MelitzFixedPointProbeResult
    nStatus::Int
    eval_failed::Bool
    obj_value::Float64
    c::Vector{Float64}
    live_candidates::Vector{MelitzOuterCandidate}
end

const MELITZ_KNITRO_EVAL_ERROR_STATUSES = (-500, -501, -502, -503, -504, -505, -506,
                                            -515, -518, -520, -522, -600)

"""
    melitz_fixed_point_probe(ctx, obj_inner, theta_probe; delta, direction,
        gradient_backend=:B, h=1e-4, inner_loop_opt, outer_loop_opt=<default>)
        -> MelitzFixedPointProbeResult

Section 6 Tests A-D: fixes ALL outer variables at `theta_probe` (KNITRO lower bound ==
upper bound, zero degrees of freedom) and runs the finite-delta outer NLP's real,
production combined callback (`melitz_build_finite_delta_callbacks`) exactly once.
"""
function melitz_fixed_point_probe(ctx, obj_inner, theta_probe::AbstractVector;
                                   delta::Real, direction::Symbol,
                                   gradient_backend::Symbol=:auto, h::Real=1e-4,
                                   cutoff_constraint_backend::Symbol=:nonlinear_reference,
                                   inner_loop_opt::AbstractString,
                                   outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta.opt"),
                                   backend::Symbol=:auto_from_gradient_backend,
                                   divergence_constraint_scaling::Symbol=:dimensionless)
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower"))
    find_smallest = direction == :upper
    D = ctx.D
    n = length(theta_probe)
    delta = Float64(delta)
    theta_probe_v = collect(Float64.(theta_probe))

    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_probe_v; delta=delta,
        find_smallest=find_smallest, gradient_backend=gradient_backend, h=h,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt, backend=backend)

    cbset = melitz_build_finite_delta_callbacks(obj, ctx, delta, find_smallest;
        cutoff_constraint_backend=cutoff_constraint_backend, gradient_backend=gradient_backend, h=h,
        divergence_constraint_scaling=divergence_constraint_scaling)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = melitz_kn_add_vars!(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, theta_probe_v)
    KNITRO.KN_set_var_upbnds_all(kc, theta_probe_v)
    KNITRO.KN_set_var_primal_init_values_all(kc, theta_probe_v)

    cIndices, cutoff_sys = melitz_register_finite_delta_knitro_problem!(kc, ctx, cbset, xIndices, n, D, obj;
        cutoff_constraint_backend=cutoff_constraint_backend)
    m = length(cIndices)

    KNITRO.KN_solve(kc)
    nStatus, objVal, _, _ = KNITRO.KN_get_solution(kc)

    eval_failed = nStatus in MELITZ_KNITRO_EVAL_ERROR_STATUSES
    c = Float64[]
    if !eval_failed
        c = zeros(m)
        KNITRO.KN_get_con_values_all(kc, c)
        # KNOWN KNITRO LIMITATION (found+documented this session, Section 3.4): once this
        # problem's own eval callback has triggered >=1 NESTED KN_new/KN_solve/KN_free
        # cycle (exactly what happens here -- every cb_F!/cb_G! call runs the real CC inner
        # solve as its own separate KNITRO instance), a POST-SOLVE `KN_get_con_values_all`
        # query for NATIVELY-registered linear constraints (no eval callback at all) comes
        # back corrupted/stale -- verified via a minimal 2-constraint reproduction outside
        # Melitz entirely (nesting a trivial second KN instance inside the SAME callback
        # reproduces the corruption; without nesting, the identical linear-registration code
        # reports exact values). The callback-computed divergence row (`c[1]`, written
        # directly into `evalResult.c[1]` during the SAME callback call) is UNAFFECTED --
        # only the natively-computed rows are. Cross-checked against KNITRO's own
        # presolve-deduced infeasibility message for a constructed infeasible point: the
        # DEDUCED value there (computed DURING solve/presolve, not via this post-solve
        # query) matched the true slack exactly, confirming solve-time enforcement is
        # correct and ONLY the post-hoc reporting call is unreliable. Fix: recompute the
        # cutoff rows directly in Julia (`cutoff_sys.C*theta+cutoff_sys.b`, exact, and the
        # authoritative source of truth per Section 3.4's own equivalence tests) rather than
        # trusting KNITRO's post-solve report for them.
        if cutoff_constraint_backend == :linear && cutoff_sys !== nothing
            c[2:end] .= cutoff_sys.C * theta_probe_v .+ cutoff_sys.b
        end
    end
    KNITRO.KN_free(kc)

    return MelitzFixedPointProbeResult(nStatus, eval_failed, eval_failed ? NaN : objVal, c,
                                        cbset.live_candidates)
end
