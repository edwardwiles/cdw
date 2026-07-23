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
# Everything else -- `PsiObjectiveBundleImplicit`'s struct/functor, `outer_loop`,
# `outer_loop_constraints!`, the KNITRO F/G callback dispatch, `inner_loop_internal`'s
# nested real KNITRO solve, the `-1e10*f <= 1e10*delta` constraint-1 convention -- is
# used completely UNCHANGED, matching the Ricardian model's own `ccOuter.jl` usage
# pattern (`counterType==1` branch) line for line. `outer_constr_index = d+1` (every
# active moment used INSIDE the inner CC problem, none held out as extra outer equality
# rows) -- matching this closure's own D^2+1 system exactly (there is no separate
# "extra" moment beyond what the inner Delta(theta) solve already enforces), so the
# `ift!`-based total-derivative correction (`PsiObjectiveBundle.jl:331-343`, only active
# when `outer_constr_index <= d`) is never exercised here, exactly as in the
# already-existing `sequential_gravity/PsiObjectiveBundleImplicitMethodB.jl` precedent
# for a DIFFERENT model (that file's own `check_methodB_valid` documents the identical
# condition).
#
# The deterministic cutoff inequalities (Section 1.3) do NOT fit this constraint
# vocabulary (`outer_loop_constraints!` only supports the divergence-budget row plus
# EQUALITY-bound held-out moments) -- they are added as a SEPARATE KNITRO constraint
# block via a second `KN_add_eval_callback` (KNITRO natively supports multiple callbacks
# over disjoint constraint-index subsets; the deterministic cutoff Jacobian
# (`melitz_cutoff_constraint_jacobian`, Section 1.3, exact/ForwardDiff-validated) is
# reused unchanged for it) -- `outer_loop` itself is not modified; this file's own driver
# mirrors `outer_loop`'s body, adding the one extra callback registration `outer_loop`
# has no hook for.

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
reads only `H[1,1]` as kappa.
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
"""
function make_melitz_moments_jacobian_b(h::Real)
    function melitz_moments_jacobian_b!(K_jac, G_jac, theta, U, obj)
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
            Gp = fixed_active_set_moments(theta .+ h .* ei, ctx, obj)
            Gm = fixed_active_set_moments(theta .- h .* ei, ctx, obj)
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

    A, f, gamma_prime_j, f_jj = expand_free_theta(theta, ctx)
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
# Bundle construction + driver, reusing PsiObjectiveBundleImplicit/outer_loop AS-IS.
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
# Deterministic cutoff constraints (Section 1.3): genuinely NEW -- no analogue exists in
# the Ricardian model (it has no gravity-pivot/cutoff-feasibility system), so there is
# nothing to adapt here; registered as a SECOND, independent KNITRO eval-callback block
# (KNITRO natively supports multiple callbacks over disjoint constraint-index subsets)
# alongside the FIRST, UNMODIFIED `callbackEval_and_ConsF/G_outer!` block that owns the
# objective and the divergence-budget constraint.
# ============================================================================

"""
    melitz_combined_callback_F!/_G!

A SINGLE combined callback covering all `1+D^2` constraints (the divergence budget,
index 1, plus the cutoff inequalities, indices `2:end`), registered ONCE on the outer
`kc`. Earlier versions registered the divergence budget and cutoff blocks as TWO
separate `KN_add_eval_callback` contexts on the same `kc` (one reusing cc_algo's own
`callbackEval_and_ConsF/G_outer!` unchanged, one new for cutoffs); that combination
reproducibly crashed KNITRO with a `-500` ("could not evaluate objective or
constraints") at the very first evaluation, even though EACH block, registered ALONE
on its own `kc`, ran flawlessly (the delta-only block completed 60 real outer
iterations; the cutoff-only block converged in 6). The cause was not pinned down
exactly (an earlier hypothesis -- a second, independent nested inner KNITRO solve
running inside the cutoff callback, on top of the delta callback's own -- was tested
and ruled out: removing it left the identical crash), but is evidently specific to
having TWO separate KNITRO callback CONTEXTS on one `kc` where one of them also
triggers a NESTED inner KNITRO solve. Merging into one callback context (this
function) sidesteps the issue entirely rather than chasing it further, and is also
simply a smaller, more obviously correct piece of new code: it computes the delta-row
by calling `inner_loop_internal`/the functor directly (the SAME calls
`callbackEval_and_ConsF/G_outer!` make, just inlined into one callback rather than a
second registered context) and appends the (KNITRO-free, cheap) cutoff constraints/
Jacobian in the same call.
"""
function melitz_combined_callback_F!(kc, cb, evalRequest, evalResult, userParams)
    obj = userParams
    theta = evalRequest.x
    objSol, x, nStatus = CounterfactualSensitivity.inner_loop_internal(obj, theta)
    evalResult.obj[1] = -objSol

    local_c = zeros(1)
    obj(x, constr=local_c)
    evalResult.c[1] = local_c[1]
    if abs(objSol) == 1e10
        evalResult.c[1] = 1e9
    end

    g_d, g_e = melitz_cutoff_constraints_at(theta, obj.γ)
    nd = length(g_d)
    evalResult.c[2:1+nd] .= g_d
    evalResult.c[2+nd:end] .= g_e
    return 0
end

function melitz_combined_callback_G!(kc, cb, evalRequest, evalResult, userParams)
    obj = userParams
    theta = evalRequest.x
    n = length(theta)
    objSol, x, nStatus = CounterfactualSensitivity.inner_loop_internal(obj, theta)

    local_jac = zeros(n)
    obj(x, evalResult.objGrad, theta; jac=local_jac)
    evalResult.objGrad .*= -1.0
    evalResult.jac[1:n] .= local_jac

    J_d, J_e = melitz_cutoff_constraint_jacobian(theta, obj.γ)
    nd, ne = size(J_d, 1), size(J_e, 1)
    @inbounds for kk in 1:nd
        evalResult.jac[kk*n+1:(kk+1)*n] .= @view J_d[kk, :]
    end
    off = 1 + nd
    @inbounds for kk in 1:ne
        evalResult.jac[(off+kk-1)*n+1:(off+kk)*n] .= @view J_e[kk, :]
    end
    return 0
end

"""
    MelitzFiniteDeltaOuterResult

Section 3.4/8.C required report fields. `terminal` is whatever KNITRO's own solution
was (may be infeasible/unverified); `cold_verified_incumbent` is the best VERIFIED
feasible point tracked during the run, INDEPENDENTLY re-solved from a cleared warm start
at the very end (Section 6: never trust a warm-started `verified=true` flag as the final
word) -- `nothing` if the run never found one.
"""
struct MelitzFiniteDeltaOuterResult
    theta_init::Vector{Float64}
    delta::Float64
    direction::Symbol
    gradient_backend::Symbol
    terminal_theta::Vector{Float64}
    terminal_eval::MelitzDeltaEvalResult
    cold_verified_incumbent::Union{Nothing,MelitzDeltaEvalResult}
    nStatus::Int
    inner_solve_count::Int
    inner_infeas_count::Int
    wall_time::Float64
end

"""
    solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; delta, direction,
        gradient_backend=:B, h=1e-4, theta_box=10.0,
        outer_loop_opt=<default>) -> MelitzFiniteDeltaOuterResult

Section 3/4: solves `minimize/maximize theta_free[1] s.t. Delta(theta)<=delta` and the
Section 1.3 cutoff inequalities. Mirrors `cc_algo/ccOuter.jl`'s own `counterType==1`
usage of `outer_loop`/`PsiObjectiveBundleImplicit` almost line for line -- see this
file's header for the precise list of what is reused unchanged vs. newly added (the
cutoff-constraint block has no Ricardian analogue).

`obj_inner`: the EXISTING `PsiObjectiveBundleDelta` bundle from `build_melitz_psi_bundle`
(shares `ctx`/reference draws with the new `PsiObjectiveBundleImplicit` this function
builds internally) -- used for the final cold-verification gate.

`theta_box`: the outer decision vector has no natural box (the governing prompt imposes
none) -- a symmetric `theta_init .± theta_box` bound is set purely for KNITRO
well-posedness (an ARTIFICIAL bound, flagged per Section 8.F, not an economic
restriction). Default `2.0`, deliberately modest (not the "generous ±10" first tried):
an unconstrained smoke run with a wide box and an APPROXIMATE (Method B) constraint
gradient ran theta far enough from the start that the inner CC problem entered a
numerically pathological region (a real, reproducible failure mode, not hypothetical --
the verified-success gate correctly refused to report that terminal point as a
result). A tighter box is a cheap extra safety net on top of Section 4's own
continuation strategy (small delta steps from an already-good start), not a substitute
for it.
"""
function solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init::AbstractVector;
                                          delta::Real, direction::Symbol,
                                          gradient_backend::Symbol=:B, h::Real=1e-4,
                                          theta_box::Real=2.0,
                                          inner_loop_opt::AbstractString,
                                          outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_outer_finite_delta.opt"))
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower"))
    t0 = time()
    find_smallest = direction == :upper   # minimize g for the upper GT bound, maximize for lower
    D = ctx.D
    n = length(theta_init)

    obj = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_init; delta=delta,
        find_smallest=find_smallest, gradient_backend=gradient_backend, h=h,
        inner_loop_opt=inner_loop_opt, outer_loop_opt=outer_loop_opt)

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, obj.outer_loop_opt)

    xIndices = KNITRO.KN_add_vars(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, collect(theta_init) .- theta_box)
    KNITRO.KN_set_var_upbnds_all(kc, collect(theta_init) .+ theta_box)
    KNITRO.KN_set_var_primal_init_values_all(kc, collect(theta_init))

    # ONE combined constraint block/callback covering the divergence budget (index 1)
    # AND the cutoff inequalities (indices 2:end) -- see melitz_combined_callback_F!'s
    # own docstring for why this is a single callback context rather than two (a real,
    # reproducible KNITRO crash with two separate contexts, root cause not fully pinned
    # down but confirmed independent of the reused cc_algo callbacks' own correctness).
    #
    # Constraint 1's bound: the functor computes `constr[1] = -1e10*f` where
    # `f=Delta(theta)>=0` always (convex duality). `Delta(theta)<=delta` is therefore
    # `f<=delta`, i.e. `constr[1] >= -1e10*delta` -- a LOWER bound. (The Ricardian
    # model's own `outer_loop_constraints!` sets an UPPER bound of `1e10*delta` for the
    # analogous Implicit-bundle row; that reduces to the vacuous `f>=-delta`, always
    # true, confirmed empirically here: a smoke run using that bound let theta drift for
    # 60 iterations with feasibility error pinned at 0.000 regardless of where theta
    # went. Not reused for that reason -- this is the one place this file's constraint
    # convention deliberately differs from the reused cc_algo pattern.) A tight
    # `delta=1e-3` test with the corrected lower-bound convention pushed the search hard
    # enough to reach a numerically pathological inner-dual region (`nStatus=-102`),
    # correctly rejected by the verified-success gate below rather than silently
    # accepted -- motivating Section 4's continuation strategy (start from a looser
    # delta).
    D2 = D * (D - 1)
    n_cutoff = D + D2
    m = 1 + n_cutoff
    cIndices = KNITRO.KN_add_cons(kc, m)
    KNITRO.KN_set_con_lobnd(kc, cIndices[1], -1e10 * delta)
    KNITRO.KN_set_con_lobnds(kc, n_cutoff, cIndices[2:end], zeros(n_cutoff))

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, melitz_combined_callback_F!)
    KNITRO.KN_set_cb_grad(kc, cb, melitz_combined_callback_G!,
        jacIndexCons=repeat(cIndices, inner=n), jacIndexVars=repeat(xIndices, outer=m))
    KNITRO.KN_set_cb_user_params(kc, cb, obj)

    CS = CounterfactualSensitivity
    CS.INNER_SOLVE_COUNT[] = 0; CS.INNER_INFEAS_COUNT[] = 0; CS.INNER_ITERS_TOTAL[] = 0
    KNITRO.KN_solve(kc)
    nStatus, _, theta_final_raw, _ = KNITRO.KN_get_solution(kc)
    solve_inner_count = CS.INNER_SOLVE_COUNT[]
    solve_infeas_count = CS.INNER_INFEAS_COUNT[]
    KNITRO.KN_free(kc)

    theta_final = collect(theta_final_raw)
    terminal_eval = evaluate_melitz_delta(theta_final, ctx, obj_inner; cold=true)

    # Section 3.4: only the COLD-verified terminal point may be reported as the
    # incumbent -- an unverified/infeasible/failed terminal point is returned to the
    # caller as `terminal_eval` (so nothing hides a failure) but `cold_verified_incumbent`
    # stays `nothing` (Section 6 does further, separate local polling on top of this).
    cold_verified_incumbent = nothing
    if terminal_eval.verified && terminal_eval.Delta <= delta + 1e-6
        cold_verified_incumbent = terminal_eval
    end

    return MelitzFiniteDeltaOuterResult(collect(theta_init), Float64(delta), direction,
        gradient_backend, theta_final, terminal_eval, cold_verified_incumbent,
        nStatus, solve_inner_count, solve_infeas_count, time() - t0)
end
