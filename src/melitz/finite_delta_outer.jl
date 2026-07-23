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
        gradient_backend=:B, h=1e-4, inner_loop_opt=..., outer_loop_opt=...)
        -> PsiObjectiveBundleImplicit

Constructs the SAME `PsiObjectiveBundleImplicit` struct the Ricardian model uses
(`cc_algo/PsiObjectiveBundle.jl`, unmodified), wired to Melitz's own moments/gradient:
`moments! = melitz_moments_adapter_outer!`, `moments_jacobian! =`
`make_melitz_moments_jacobian_b(h)` or `_d()` depending on `gradient_backend`,
`d = ctx.moment_layout.num_moments`, `outer_constr_index = d+1` (all moments internal --
see this file's header), `l = length(theta_free_init)`, `U = z_draws` (the SAME reference
draws the inner Delta(theta) solve uses).
"""
function build_melitz_implicit_bundle(ctx, z_draws::AbstractMatrix, theta_free_init::AbstractVector;
                                       delta::Real, find_smallest::Bool,
                                       gradient_backend::Symbol=:B, h::Real=1e-4,
                                       inner_loop_opt::AbstractString,
                                       outer_loop_opt::AbstractString)
    d = ctx.moment_layout.num_moments
    l = length(theta_free_init)

    mj! = gradient_backend == :B ? make_melitz_moments_jacobian_b(h) :
          gradient_backend == :D ? make_melitz_moments_jacobian_d() :
          error("gradient_backend must be :B or :D for the KNITRO-native Implicit path " *
                "(Backend R does not fit the moments_jacobian! hook -- see file header)")

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
end

"""
    melitz_build_finite_delta_callbacks(obj, ctx, delta, find_smallest;
        n_live_candidates_tracked=5) -> NamedTuple

Section 6/7: factors the finite-delta outer NLP's combined callback pair (objective +
divergence-budget + cutoff constraints, Sections 3.1/4.1/5.2) out of
`solve_melitz_finite_delta_bound` so a SEPARATE fixed-point test driver
(`melitz_fixed_point_probe`) can register the EXACT SAME production `cb_F!`/`cb_G!`
closures against a degenerate (0-degree-of-freedom) KNITRO problem -- Section 6's own
requirement ("these tests must pass through the exact production combined callback and
registered bounds, not merely call helper functions") -- rather than duplicating this
logic a second time, which would risk exactly the production/test drift Section 6 warns
against.

Returns a `NamedTuple` `(cb_F!, cb_G!, live_candidates, n_inner_eval_failures,
signed_objective)`. `live_candidates`/`n_inner_eval_failures` are mutated in place by the
callbacks as KNITRO calls them -- the caller reads them AFTER `KN_solve` returns.
"""
function melitz_build_finite_delta_callbacks(obj, ctx, delta::Float64, find_smallest::Bool;
                                              n_live_candidates_tracked::Int=5,
                                              cutoff_constraint_backend::Symbol=:nonlinear_reference)
    cutoff_constraint_backend in (:linear, :nonlinear_reference) || throw(ArgumentError(
        "cutoff_constraint_backend must be :linear or :nonlinear_reference, got $cutoff_constraint_backend"))
    signed_objective(theta) = find_smallest ? theta[1] : -theta[1]
    live_candidates = MelitzOuterCandidate[]
    n_inner_eval_failures = Ref(0)
    n_fc_calls = Ref(0)
    n_ga_calls = Ref(0)

    function register_live_candidate!(theta::Vector{Float64}, Delta_val::Float64,
                                       x::Vector{Float64}, nStatus::Integer)
        r = evaluate_melitz_delta_from_solution(theta, ctx, obj, Delta_val, x, nStatus)
        cls = melitz_classify_outer_feasibility(r, delta)
        cls.outer_feasible || return nothing
        push!(live_candidates, MelitzOuterCandidate(signed_objective(theta), r, cls, :live))
        sort!(live_candidates; by=c -> c.objective)
        while length(live_candidates) > n_live_candidates_tracked
            pop!(live_candidates)
        end
        return nothing
    end

    # Section 5.2: a genuine inner-solve failure is an EVALUATION failure, not a model
    # value. Retry once from a neutral cold start; if that also fails, throw a
    # `DomainError` -- KNITRO.jl's own `_try_catch_handler` catches this and converts it to
    # a proper evaluation-error return code (KN_RC_EVAL_ERR), telling KNITRO to
    # reject/backtrack from this trial point -- matching
    # `full_aod_diag/d4_exact/c9_phase8_d20_pilot.jl`'s documented convention for the
    # Ricardian model.
    function inner_solve_verified_or_fail(theta::AbstractVector)
        objSol, x, nStatus = CounterfactualSensitivity.inner_loop_internal(obj, theta)
        if nStatus in (0, -100, -101, -103)
            return objSol, x, nStatus
        end
        was_cached = obj.use_cached_x
        obj.use_cached_x = false
        objSol2, x2, nStatus2 = CounterfactualSensitivity.inner_loop_internal(obj, theta)
        obj.use_cached_x = was_cached
        if nStatus2 in (0, -100, -101, -103)
            return objSol2, x2, nStatus2
        end
        n_inner_eval_failures[] += 1
        throw(DomainError(theta[1],
            "melitz finite-delta outer callback: inner CC dual solve failed even after a " *
            "cold retry (warm nStatus=$nStatus, cold nStatus=$nStatus2) -- rejecting this " *
            "trial point (Section 5.2)"))
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        t_fc0 = time_ns()
        n_fc_calls[] += 1
        theta = collect(evalRequest.x)
        objSol, x, nStatus = @melitz_profile :fc_inner_solve inner_solve_verified_or_fail(theta)

        # Section 3.1: the outer objective is ALWAYS the finite, deterministic gamma
        # coordinate -- never the inner solve's own return value or a failure sentinel.
        evalResult.obj[1] = signed_objective(theta)

        local_c = zeros(1)
        obj(x, constr=local_c)   # raw functor call: local_c[1] == +1e10*Delta(theta) (Section 18)
        Delta_theta = local_c[1] / 1e10
        # Section 4.1: c_delta(theta) = Delta(theta)/delta <= 1 -- dimensionless, O(1) at
        # the budget boundary regardless of delta's own scale (replaces the old
        # 1e10-scaled row).
        evalResult.c[1] = Delta_theta / delta

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
        melitz_record_seconds!(:fc_total, (time_ns() - t_fc0) / 1e9)
        return 0
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        t_ga0 = time_ns()
        n_ga_calls[] += 1
        theta = collect(evalRequest.x)
        n_ = length(theta)
        objSol, x, nStatus = @melitz_profile :ga_inner_solve inner_solve_verified_or_fail(theta)

        # Section 3.1: d(±theta[1])/dtheta -- exact, trivial, independent of the inner
        # solve (which is still needed below, for the constraint Jacobian only).
        evalResult.objGrad .= 0.0
        evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0

        dummy_g = zeros(n_)
        local_jac = zeros(n_)
        @melitz_profile :ga_divergence_gradient obj(x, dummy_g, theta; jac=local_jac)   # local_jac == d(1e10*Delta)/dtheta at fixed x
        evalResult.jac[1:n_] .= local_jac ./ (1e10 * delta)   # Section 4.2: same scaling as the value

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
        melitz_record_seconds!(:ga_total, (time_ns() - t_ga0) / 1e9)
        return 0
    end

    return (cb_F! = cb_F!, cb_G! = cb_G!, live_candidates = live_candidates,
            n_inner_eval_failures = n_inner_eval_failures, signed_objective = signed_objective,
            cutoff_constraint_backend = cutoff_constraint_backend,
            n_fc_calls = n_fc_calls, n_ga_calls = n_ga_calls)
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
    cIndices = KNITRO.KN_add_cons(kc, m)
    # Section 4.1: c_delta <= 1 -- dimensionless, replaces the old 1e10*delta magnitude.
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1.0)

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
"""
function solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init::AbstractVector;
                                          delta::Real, direction::Symbol,
                                          gradient_backend::Symbol=:B, h::Real=1e-4,
                                          theta_box::Union{Real,AbstractVector}=2.0,
                                          n_live_candidates_tracked::Int=5,
                                          cutoff_constraint_backend::Symbol=:nonlinear_reference,
                                          inner_loop_opt::AbstractString,
                                          outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta.opt"))
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower"))
    t0 = time()
    find_smallest = direction == :upper   # minimize g for the upper GT bound, maximize for lower
    D = ctx.D
    n = length(theta_init)
    delta = Float64(delta)

    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_init; delta=delta,
        find_smallest=find_smallest, gradient_backend=gradient_backend, h=h,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt)

    signed_objective(theta) = find_smallest ? theta[1] : -theta[1]

    # ------------------------------------------------------------------------
    # Section 2.1: install the initial incumbent BEFORE KN_solve is ever called, from a
    # COLD evaluation of theta_init -- this must be returned even if KNITRO's own
    # trajectory never produces a verified point (Section 12's zero/one-iteration
    # regression test pins exactly this).
    # ------------------------------------------------------------------------
    initial_eval = evaluate_melitz_delta(collect(theta_init), ctx, obj_inner; cold=true)
    initial_classification = melitz_classify_outer_feasibility(initial_eval, delta)
    initial_incumbent = initial_classification.outer_feasible ?
        MelitzOuterCandidate(signed_objective(theta_init), initial_eval, initial_classification, :initial) :
        nothing

    # Section 6/7: the SAME callback pair a fixed-point test would register directly.
    cbset = melitz_build_finite_delta_callbacks(obj, ctx, delta, find_smallest;
        n_live_candidates_tracked=n_live_candidates_tracked,
        cutoff_constraint_backend=cutoff_constraint_backend)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, collect(theta_init) .- theta_box)
    KNITRO.KN_set_var_upbnds_all(kc, collect(theta_init) .+ theta_box)
    KNITRO.KN_set_var_primal_init_values_all(kc, collect(theta_init))

    cIndices, cutoff_sys = melitz_register_finite_delta_knitro_problem!(kc, ctx, cbset, xIndices, n, D, obj;
        cutoff_constraint_backend=cutoff_constraint_backend)

    CS = CounterfactualSensitivity
    CS.INNER_SOLVE_COUNT[] = 0; CS.INNER_INFEAS_COUNT[] = 0; CS.INNER_ITERS_TOTAL[] = 0
    KNITRO.KN_solve(kc)
    nStatus, _, theta_final_raw, _ = KNITRO.KN_get_solution(kc)
    solve_inner_count = CS.INNER_SOLVE_COUNT[]
    solve_infeas_count = CS.INNER_INFEAS_COUNT[]
    KNITRO.KN_free(kc)

    theta_final = collect(theta_final_raw)
    terminal_eval = evaluate_melitz_delta(theta_final, ctx, obj_inner; cold=true)
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

    best_live_incumbent = isempty(live_candidates) ? nothing : first(live_candidates)

    return MelitzFiniteDeltaOuterResult(collect(theta_init), delta, direction,
        gradient_backend, theta_final, terminal_eval, terminal_classification,
        initial_incumbent, best_live_incumbent, cold_verified_incumbent,
        nStatus, solve_inner_count, solve_infeas_count, cbset.n_inner_eval_failures[], time() - t0,
        cutoff_constraint_backend, cbset.n_fc_calls[], cbset.n_ga_calls[])
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
                                   gradient_backend::Symbol=:B, h::Real=1e-4,
                                   cutoff_constraint_backend::Symbol=:nonlinear_reference,
                                   inner_loop_opt::AbstractString,
                                   outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta.opt"))
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower"))
    find_smallest = direction == :upper
    D = ctx.D
    n = length(theta_probe)
    delta = Float64(delta)
    theta_probe_v = collect(Float64.(theta_probe))

    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_probe_v; delta=delta,
        find_smallest=find_smallest, gradient_backend=gradient_backend, h=h,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt)

    cbset = melitz_build_finite_delta_callbacks(obj, ctx, delta, find_smallest;
        cutoff_constraint_backend=cutoff_constraint_backend)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, n)
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
