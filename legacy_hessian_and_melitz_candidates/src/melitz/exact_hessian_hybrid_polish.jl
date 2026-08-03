# Exact-HVP-stabilized-polish hybrid middle-loop driver (2026-07-31 rescue-experiment task).
#
# Governing prompt: does exact curvature (`exact_reduced_hessian.jl`, treated as VALIDATED --
# not re-derived or re-audited here) accelerate the FINAL phase of the existing constrained
# middle KNITRO solve (`solve_melitz_fixed_q_A_profile_v2`, `fixed_q_a_middle_loop.jl`)? This
# file does NOT touch the HVP formulas (`melitz_mixed_B_mul!`/`melitz_mixed_Bt_mul!`/
# `melitz_explicit_phi_aa_mul!`/`melitz_profiled_A_hvp_full!`, `exact_reduced_hessian.jl`) or the
# v2 driver's own correctness logic -- it is purely additive:
#
#   1. A rank-revealing eigendecomposition + stability report of the per-point dual Hessian
#      `H_eta_eta` (raw/equilibrated condition numbers, numerical rank at several thresholds,
#      near-null-space projection of a probe direction) -- Section 4/13 of the prior session's
#      own flagged follow-up.
#   2. Two regularization strategies tied to the point's OWN `eigmax` (never a fixed absolute
#      shift): a truncated eigen-pseudo-inverse, and eigenvalue-scaled Levenberg-Marquardt
#      damping -- both operating on the SAME eigendecomposition the stability report computes.
#   3. A KNITRO Hessian-VECTOR-PRODUCT callback (`hessopt=KN_HESSOPT_PRODUCT`,
#      `algorithm in (KN_ALG_BAR_CG, KN_ALG_ACT_CG)`) wired into a SECOND, short KNITRO solve
#      that reuses the SAME linear ordering/same-bin constraint rows, bounds, cap handling, and
#      cached FC/GA evaluator (`melitz_middle_objective_and_gradient_cached!`) as
#      `solve_melitz_fixed_q_A_profile_v2` -- a "polish" phase entered ONLY after a verified
#      L-BFGS incumbent (phase 1, unchanged v2 call) clears a configurable gradient/KKT gate AND
#      the per-point stability gate (rank/null-space/symmetry/conditioning).
#   4. A strict-incumbent-retention hybrid orchestrator
#      (`solve_melitz_fixed_q_A_profile_hybrid_hvp`) that can never return a worse verified
#      result than phase 1 alone -- the same cold-reverification convention `_v2` already uses.
#
# Continuation dual warm start (governing prompt requirement 8): `session.obj.use_cached_x`/
# `.x` are NEVER reset between phase 1 and phase 2 by this file -- phase 2 begins from
# whatever dual phase 1's own search left warm, exactly like `solve_melitz_fixed_q_A_profile_v2`
# itself never resets between successive trial points. Only the FINAL cold-reverification
# candidates use `warm_start_source=:neutral`, matching the pre-existing convention.

using LinearAlgebra: dot, eigen, Symmetric, Diagonal, norm, diag, I

# ----------------------------------------------------------------------------------------
# Rank-revealing eigendecomposition + stability report of H_eta_eta = -(1/M)*Hfull.
# ----------------------------------------------------------------------------------------

"""
    MelitzHxxStabilityReport

Governing prompt requirement 5's own required diagnostics, computed from ONE full symmetric
eigendecomposition of `P = (1/M)*Hfull` (`pt.Hfull`/`pt.M`, `exact_reduced_hessian.jl` --
cheap, `K+1<=402` at `D<=20`). `eigvals`/`eigvecs` are kept (ascending order, `LinearAlgebra.eigen`'s
own convention) so a caller can build BOTH regularization strategies from the SAME
decomposition without a second `eigen` call.

  - `cond_raw`: `eigmax/max(eigmin,eps())` (matches `melitz_hxx_report`, kept for continuity).
  - `cond_equilibrated`: condition number of the DIAGONALLY EQUILIBRATED matrix
    `D*P*D`, `D=Diagonal(1 ./ sqrt.(diag(P)))` (Van der Sluis equilibration) -- a DIAGNOSTIC
    lens only (the actual HVP solve below still operates in `P`'s own native eigenbasis, never
    re-solved in equilibrated coordinates), reported because raw conditioning at real D=20 can
    be dominated by a few cells' own units/scale rather than genuine rank deficiency.
  - `ranks`: `Dict(tau => count(eigvals .>= tau*eigmax))` for every `tau` in `rel_thresholds`
    -- the numerical rank of `P` at each relative threshold.
  - `nullspace_frac`: `norm(Q_null' * probe) / norm(probe)`, `Q_null` the eigenvectors with
    `eigvals < null_threshold*eigmax` (`null_threshold` below), `probe` a caller-supplied
    direction (the gate uses the incumbent's own gradient direction; the live HVP callback
    uses the actual `Bv` it just computed) -- requirement 5's "projection of Bv onto the
    near-null space."
  - `symmetric_ok`: `norm(Hfull - Hfull')/max(norm(Hfull),eps()) < sym_tol` -- `Hfull` is
    mirrored from its own upper triangle by `build_melitz_exact_hessian_point_state`, so this
    should always pass; checked anyway per requirement 7 ("HVP is not symmetric/stable").
"""
struct MelitzHxxStabilityReport
    K1::Int
    eigvals::Vector{Float64}     # ascending, length K+1
    eigvecs::Matrix{Float64}     # columns = eigenvectors, matching eigvals order
    eigmin::Float64
    eigmax::Float64
    cond_raw::Float64
    cond_equilibrated::Float64
    ranks::Dict{Float64,Int}
    null_threshold::Float64
    nullspace_frac::Float64
    symmetric_ok::Bool
    sym_residual::Float64
end

"""
    melitz_hxx_stability_report(pt::MelitzExactHessianPointState, probe::AbstractVector{Float64};
        rel_thresholds=(1e-6,1e-8,1e-10,1e-12), null_threshold=1e-10, sym_tol=1e-8) -> MelitzHxxStabilityReport

Builds the full stability report at the point already cached in `pt`
(`build_melitz_exact_hessian_point_state`, `exact_reduced_hessian.jl`, unchanged). `probe`
(length `K+1`, SAME layout as the dual `x`) is the direction the near-null-space projection is
measured against -- the caller decides what is economically meaningful (the incumbent's own
`B*grad_free` at the gate stage, or the live `Bv` inside the HVP callback).
"""
function melitz_hxx_stability_report(pt::MelitzExactHessianPointState, probe::AbstractVector{Float64};
                                      rel_thresholds::NTuple{N,Float64}=(1e-6, 1e-8, 1e-10, 1e-12),
                                      null_threshold::Float64=1e-10, sym_tol::Float64=1e-8) where {N}
    K1 = pt.K + 1
    P = Symmetric(pt.Hfull ./ pt.M)
    F = eigen(P)   # ascending eigvals, LinearAlgebra convention
    ev = F.values
    Q = F.vectors
    eigmin = ev[1]
    eigmax = ev[end]
    cond_raw = eigmax / max(eigmin, eps())

    dscale = 1.0 ./ sqrt.(max.(diag(pt.Hfull ./ pt.M), eps()))
    Peq = Symmetric((dscale .* (pt.Hfull ./ pt.M)) .* dscale')
    ev_eq = eigen(Peq).values
    cond_equilibrated = ev_eq[end] / max(ev_eq[1], eps())

    ranks = Dict{Float64,Int}()
    for tau in rel_thresholds
        ranks[tau] = count(>=(tau * eigmax), ev)
    end

    null_mask = ev .< (null_threshold * eigmax)
    nullspace_frac = if any(null_mask) && norm(probe) > 0
        Qnull = Q[:, null_mask]
        norm(Qnull' * probe) / norm(probe)
    else
        0.0
    end

    sym_residual = norm(pt.Hfull .- pt.Hfull') / max(norm(pt.Hfull), eps())
    symmetric_ok = sym_residual < sym_tol

    return MelitzHxxStabilityReport(K1, ev, Q, eigmin, eigmax, cond_raw, cond_equilibrated,
        ranks, null_threshold, nullspace_frac, symmetric_ok, sym_residual)
end

# ----------------------------------------------------------------------------------------
# Regularized solves of H_eta_eta*z=b (`H_eta_eta=-P`), reusing the SAME eigendecomposition
# the stability report computes -- neither strategy uses a fixed absolute shift (governing
# prompt requirement 6).
# ----------------------------------------------------------------------------------------

"""
    melitz_hxx_solve_stabilized(report::MelitzHxxStabilityReport, b::AbstractVector{Float64};
        strategy=:lm, lm_floor_frac=1e-8, trunc_rel_threshold=1e-8) -> z

Solves `H_eta_eta*z=b` (`H_eta_eta=-P`) via the CACHED eigendecomposition (`report.eigvecs`/
`report.eigvals`), never re-factorizing:

  - `:lm` (default): eigenvalue-scaled Levenberg-Marquardt damping, `reg = max(0,
    lm_floor_frac*eigmax - eigmin)` -- shifts EVERY eigenvalue up by `reg` (tied to the point's
    OWN `eigmax`, never a fixed `1e-10`), then solves in the eigenbasis
    (`z = -Q*((Q'b)./(ev.+reg))`).
  - `:truncated_pinv`: keeps only eigenpairs with `ev >= trunc_rel_threshold*eigmax`, forms a
    genuine truncated Moore-Penrose inverse on that range (`z = -Q_kept*((Q_kept'b)./ev_kept)`),
    zeroing the near-null contribution entirely rather than damping it.
"""
function melitz_hxx_solve_stabilized(report::MelitzHxxStabilityReport, b::AbstractVector{Float64};
                                      strategy::Symbol=:lm, lm_floor_frac::Float64=1e-8,
                                      trunc_rel_threshold::Float64=1e-8)
    ev = report.eigvals
    Q = report.eigvecs
    eigmax = report.eigmax
    eigmin = report.eigmin
    qtb = Q' * b
    if strategy == :lm
        reg = max(0.0, lm_floor_frac * eigmax - eigmin)
        z = Q * (qtb ./ (ev .+ reg))
        return -z
    elseif strategy == :truncated_pinv
        keep = ev .>= (trunc_rel_threshold * eigmax)
        z = zeros(length(b))
        @inbounds for i in eachindex(ev)
            keep[i] || continue
            z .+= (qtb[i] / ev[i]) .* view(Q, :, i)
        end
        return -z
    else
        throw(ArgumentError("melitz_hxx_solve_stabilized: strategy must be :lm or :truncated_pinv, got $strategy"))
    end
end

# ----------------------------------------------------------------------------------------
# Stabilized full/free-coordinate HVP -- reuses melitz_mixed_B_mul!/_Bt_mul!/
# melitz_explicit_phi_aa_mul! VERBATIM (exact_reduced_hessian.jl, unchanged); only the middle
# `H_eta_eta^{-1}` solve step is swapped for the stabilized version above.
# ----------------------------------------------------------------------------------------

"""
    melitz_profiled_A_hvp_full_stabilized!(out_full_A, pt, ctx, v_full_A, report;
        strategy=:lm, lm_floor_frac=1e-8, trunc_rel_threshold=1e-8) -> (out_full_A, Bv, nullspace_frac)

Same formula as `melitz_profiled_A_hvp_full!` (`Hv = phi_aa*v - B'*(H_eta_eta^{-1}*(B*v))`),
but solving the inner `H_eta_eta` system via `melitz_hxx_solve_stabilized` instead of the
un-stabilized cached Cholesky. Returns the computed `Bv` too (so the caller can report its
near-null-space projection without a second `melitz_mixed_B_mul!` call).
"""
function melitz_profiled_A_hvp_full_stabilized!(out_full_A::AbstractMatrix{Float64}, pt::MelitzExactHessianPointState,
                                                 ctx, v_full_A::AbstractMatrix{Float64}, report::MelitzHxxStabilityReport;
                                                 strategy::Symbol=:lm, lm_floor_frac::Float64=1e-8,
                                                 trunc_rel_threshold::Float64=1e-8)
    K = pt.K
    b = zeros(K + 1)
    melitz_mixed_B_mul!(b, pt, ctx, v_full_A)
    z = melitz_hxx_solve_stabilized(report, b; strategy=strategy, lm_floor_frac=lm_floor_frac,
        trunc_rel_threshold=trunc_rel_threshold)
    Btz = zeros(size(v_full_A))
    melitz_mixed_Bt_mul!(Btz, pt, ctx, z)
    phiaa = zeros(size(v_full_A))
    melitz_explicit_phi_aa_mul!(phiaa, pt, ctx, v_full_A)
    @. out_full_A = phiaa - Btz
    return out_full_A, b
end

# ----------------------------------------------------------------------------------------
# Gating logic (governing prompt requirement 3/7): decide ONCE, at the verified L-BFGS
# incumbent, whether to enter the exact-HVP polish phase at all.
# ----------------------------------------------------------------------------------------

"""
    MelitzExactHvpPolishConfig

All thresholds governing `solve_melitz_fixed_q_A_profile_hybrid_hvp`'s gate + polish phase.
Every field has a documented default; all are caller-overridable.

  - `polish_grad_tol`: phase 1 incumbent's own `norm(grad_free)` must be `<=` this to even
    ATTEMPT the stability gate (requirement 3, "configurable gradient/KKT threshold").
  - `accepted_nStatus`: phase 1's own `nStatus` must be in this set (the SAME
    successful/locally-optimal-equivalent codes this codebase already treats as clean
    elsewhere, `inner_screening.jl`/`cc_bundle.jl`/Addendum Part 2 above).
  - `min_rank_frac`: numerical rank at `rank_gate_threshold` divided by `K+1` must be `>=` this.
  - `rank_gate_threshold`: which of `rel_thresholds`' ranks the gate itself uses.
  - `max_cond_equilibrated`: ceiling on the DIAGNOSTIC equilibrated condition number.
  - `max_nullspace_frac`: ceiling on the incumbent gradient's own near-null-space projection.
  - `reg_strategy`/`lm_floor_frac`/`trunc_rel_threshold`: forwarded to
    `melitz_hxx_solve_stabilized` throughout the polish phase.
  - `polish_max_evals`/`polish_box`/`polish_opt_file`: phase 2's own KNITRO setup (`polish_box`
    is a symmetric bound around the phase-1 incumbent, in the SAME coordinate units -- must be
    `<=` the original `box` the caller used for phase 1, never wider).
  - `cap_handling`/`cap_barrier_multiple`: forwarded verbatim to phase 2, matching phase 1.
"""
struct MelitzExactHvpPolishConfig
    polish_grad_tol::Float64
    accepted_nStatus::Vector{Int}
    min_rank_frac::Float64
    rank_gate_threshold::Float64
    max_cond_equilibrated::Float64
    max_nullspace_frac::Float64
    reg_strategy::Symbol
    lm_floor_frac::Float64
    trunc_rel_threshold::Float64
    rel_thresholds::NTuple{4,Float64}
    null_threshold::Float64
    sym_tol::Float64
    polish_max_evals::Int
    polish_box::Float64
    polish_opt_file::String
    cap_handling::Symbol
    cap_barrier_multiple::Float64
end
function MelitzExactHvpPolishConfig(;
        polish_grad_tol::Real=1e-3,
        accepted_nStatus::Vector{Int}=[0, -100, -101, -103],
        min_rank_frac::Real=0.3,
        rank_gate_threshold::Real=1e-8,
        max_cond_equilibrated::Real=1e12,
        max_nullspace_frac::Real=0.2,
        reg_strategy::Symbol=:lm,
        lm_floor_frac::Real=1e-8,
        trunc_rel_threshold::Real=1e-8,
        rel_thresholds::NTuple{4,Float64}=(1e-6, 1e-8, 1e-10, 1e-12),
        null_threshold::Real=1e-10,
        sym_tol::Real=1e-8,
        polish_max_evals::Int=40,
        polish_box::Real=0.02,
        polish_opt_file::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_middle_loop_exact_hvp_polish_opt_2026-07-31.opt"),
        cap_handling::Symbol=:reject,
        cap_barrier_multiple::Real=5.0)
    return MelitzExactHvpPolishConfig(Float64(polish_grad_tol), accepted_nStatus, Float64(min_rank_frac),
        Float64(rank_gate_threshold), Float64(max_cond_equilibrated), Float64(max_nullspace_frac),
        reg_strategy, Float64(lm_floor_frac), Float64(trunc_rel_threshold), rel_thresholds,
        Float64(null_threshold), Float64(sym_tol), polish_max_evals, Float64(polish_box),
        String(polish_opt_file), cap_handling, Float64(cap_barrier_multiple))
end

"""
    MelitzHvpGateResult

Result of the ONE-TIME gate check at the phase-1 incumbent. `attempted=false` means the
gradient/KKT prerequisite (requirement 3) itself failed -- the stability report was never even
built. `ok=true` means phase 2 should run; `reason` records the FIRST failing test, checked in
the order requirement 7 lists them (rank, null-space, KKT, symmetry) with the gradient/KKT
prerequisite checked first (requirement 3 gates requirement 4 itself, per the governing
prompt's own ordering: "before using exact curvature, construct ... THEN use ...").
"""
struct MelitzHvpGateResult
    attempted::Bool
    ok::Bool
    reason::Symbol
    report::Union{Nothing,MelitzHxxStabilityReport}
    grad_norm::Float64
    nStatus::Int
end

"""
    melitz_hvp_gate_check(pt, ctx, grad_free_full, nStatus, cfg) -> MelitzHvpGateResult

Requirement 3/4/7's own gate, checked in order: (1) `nStatus` accepted AND
`norm(grad_free_full) <= cfg.polish_grad_tol` -- else `attempted=false`,
`reason=:poor_kkt_or_gradient`; (2) build the stability report (`probe = vec(grad_free_full)`
zero-padded to length `K+1` with the `zeta`-row set to 0 -- the profiled A-gradient has no
`zeta`/`mu` component of its own, so the probe is embedded as `[0; mu-shaped zero]`... actually
the null-space projection needs a `K+1`-length dual-space vector; since the gate's natural
probe is `B*grad_direction` (a genuine dual-space vector), this function computes `Bv` at
`v=grad_free_full` FIRST via `melitz_mixed_B_mul!` and uses THAT as the null-space probe,
matching exactly what the live HVP callback does with its own `Bv` -- Section 5's "projection
of Bv onto the near-null space" is over `Bv`, not over `v` itself); (3) `rank_frac >=
min_rank_frac`; (4) `nullspace_frac <= max_nullspace_frac`; (5) `cond_equilibrated <=
max_cond_equilibrated`; (6) `symmetric_ok`.
"""
function melitz_hvp_gate_check(pt::MelitzExactHessianPointState, ctx, grad_free_full::AbstractMatrix{Float64},
                                nStatus::Int, cfg::MelitzExactHvpPolishConfig)
    gnorm = norm(grad_free_full)
    kkt_ok = nStatus in cfg.accepted_nStatus
    if !(kkt_ok && gnorm <= cfg.polish_grad_tol)
        return MelitzHvpGateResult(false, false, :poor_kkt_or_gradient, nothing, gnorm, nStatus)
    end

    bv_probe = zeros(pt.K + 1)
    melitz_mixed_B_mul!(bv_probe, pt, ctx, grad_free_full)
    report = melitz_hxx_stability_report(pt, bv_probe; rel_thresholds=cfg.rel_thresholds,
        null_threshold=cfg.null_threshold, sym_tol=cfg.sym_tol)

    rank_frac = report.ranks[cfg.rank_gate_threshold] / report.K1
    if rank_frac < cfg.min_rank_frac
        return MelitzHvpGateResult(true, false, :stable_rank_test_failed, report, gnorm, nStatus)
    end
    if report.nullspace_frac > cfg.max_nullspace_frac
        return MelitzHvpGateResult(true, false, :material_nullspace_component, report, gnorm, nStatus)
    end
    if report.cond_equilibrated > cfg.max_cond_equilibrated
        return MelitzHvpGateResult(true, false, :conditioning_ceiling_exceeded, report, gnorm, nStatus)
    end
    if !report.symmetric_ok
        return MelitzHvpGateResult(true, false, :hvp_not_symmetric, report, gnorm, nStatus)
    end
    return MelitzHvpGateResult(true, true, :accepted, report, gnorm, nStatus)
end

# ----------------------------------------------------------------------------------------
# Phase 2: KNITRO-wired exact-HVP polish, same constrained problem as
# solve_melitz_fixed_q_A_profile_v2 (linear ordering/same-bin rows, bounds, cap handling,
# cached FC/GA evaluator) -- only the Hessian is different (hessvec callback instead of
# hessopt=6 L-BFGS).
# ----------------------------------------------------------------------------------------

"""
    MelitzHvpPolishResult

Result of `solve_melitz_middle_exact_hvp_polish`. Mirrors `MelitzMiddleProfileV2Result`'s own
strict-incumbent-retention fields/invariants so the hybrid orchestrator can treat both phases
uniformly.
"""
struct MelitzHvpPolishResult
    nStatus::Int
    incumbent_source::Symbol
    A_free_incumbent::Vector{Float64}
    theta_free_incumbent::Vector{Float64}
    r_incumbent::MelitzInnerResult
    Delta_incumbent::Float64
    Delta_start_verified::Float64
    Delta_terminal_verified::Float64
    n_fc_calls::Int
    n_ga_calls::Int
    n_hessvec_calls::Int
    n_finite_solved::Int
    n_above_cap::Int
    n_infinite_certified::Int
    wall_s::Float64
    n_point_gate_rejections::Int
end

"""
    solve_melitz_middle_exact_hvp_polish(session, q_fixed, gpj_fixed, x_start, ctx, sys,
        exact_cache, bad_cache, stats, cfg::MelitzExactHvpPolishConfig;
        coordinate=:logA, policy=session.policy, warm_start_source=:previous,
        origin_block_screen=false) -> MelitzHvpPolishResult

Phase 2. Same KNITRO setup as `solve_melitz_fixed_q_A_profile_v2` (SAME `sys.rows_A`/`rows_H`
linear rows, SAME `melitz_middle_objective_and_gradient_cached!` evaluator, SAME
`cap_handling` convention) but with `algorithm=KN_ALG_BAR_CG`/`ACT_CG` + `hessopt=
KN_HESSOPT_PRODUCT` (`cfg.polish_opt_file`) and a Hessian-VECTOR-PRODUCT callback built from
`melitz_profiled_A_hvp_full_stabilized!`. `exact_cache`/`bad_cache`/`stats` are the SAME
instances phase 1 used (passed in, never rebuilt here) -- a repeat request at a point phase 1
already solved (e.g. the polish start itself, which IS phase 1's own incumbent) is a genuine
cache hit, not a second inner solve (requirement: "one inner solve per unique A point," now
spanning BOTH phases). `session.obj.use_cached_x`/`.x` are NEVER touched by this function
except via the normal evaluator path -- the dual warm-starts continuously from wherever phase 1
left it (requirement 8).

The Hessian-vector callback caches its own per-point state (`MelitzExactHessianPointState` +
eigendecomposition) keyed on the SAME `theta_free_middle` fingerprint, rebuilt only when KNITRO
requests a genuinely new `x` (the expected pattern for `KN_ALG_BAR_CG`/`ACT_CG`: several
Hessian-vector products per outer iterate at a FIXED `x`, one `melitz_update_operator_at_theta!`
call unconditionally before ANY point-state build removes any dependence on cache-hit
bookkeeping elsewhere leaving `session.obj.op` in the right state).
"""
function solve_melitz_middle_exact_hvp_polish(session::MelitzInnerSession, q_fixed::AbstractMatrix{Float64},
                                               gpj_fixed::Real, x_start::AbstractVector{Float64}, ctx,
                                               sys::MelitzFixedQMiddleConstraintSystem,
                                               exact_cache::MelitzExactPointCache, bad_cache::MelitzMiddleBadPointCache,
                                               stats::MelitzMiddleCacheStats, cfg::MelitzExactHvpPolishConfig;
                                               coordinate::Symbol=:logA,
                                               policy::MelitzInnerSolvePolicy=session.policy,
                                               warm_start_source::Symbol=:previous,
                                               origin_block_screen::Bool=false)
    t0 = time()
    n = length(x_start)
    x_start_v = Vector{Float64}(x_start)
    D = ctx.D
    M_A = melitz_A_free_linear_map(ctx)

    rows = coordinate == :logH ? sys.rows_H : sys.rows_A
    rhs = coordinate == :logH ? sys.rhs_H : sys.rhs_A

    n_fc_calls = Ref(0); n_ga_calls = Ref(0); n_hv_calls = Ref(0)
    n_finite = Ref(0); n_cap = Ref(0); n_inf = Ref(0)
    n_gate_reject = Ref(0)
    incumbent_Delta = Ref(Inf)
    incumbent_theta = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    function consider_incumbent!(theta_free::Vector{Float64}, Delta::Float64)
        if Delta < incumbent_Delta[] - 1e-12
            incumbent_Delta[] = Delta
            incumbent_theta[] = copy(theta_free)
        end
    end

    function classify!(x::Vector{Float64})
        r = melitz_middle_objective_and_gradient_cached!(session, x, q_fixed, gpj_fixed, ctx,
            exact_cache, bad_cache, stats; coordinate=coordinate, policy=policy,
            warm_start_source=warm_start_source, origin_block_screen=origin_block_screen)
        if r.classification_sym == :FiniteSolved
            n_finite[] += 1
            consider_incumbent!(r.theta_free, r.Delta)
        elseif r.classification_sym == :AboveEvaluationCap
            n_cap[] += 1
        else
            n_inf[] += 1
        end
        return r
    end

    cap_barrier_value = cfg.cap_barrier_multiple * melitz_policy_cap(policy)
    function _eval(x::Vector{Float64})
        r = classify!(x)
        r.classification_sym == :FiniteSolved && return r
        if cfg.cap_handling == :barrier
            return (Delta=cap_barrier_value, grad_free=r.grad_free, classification_sym=r.classification_sym,
                    certified_lower_bound=r.certified_lower_bound, theta_free=r.theta_free, key=r.key,
                    cache_hit=r.cache_hit)
        end
        reason = r.classification_sym == :AboveEvaluationCap ?
            "AboveEvaluationCap (certified_lower_bound=$(r.certified_lower_bound))" : "InfiniteDeltaCertified"
        throw(DomainError(x, "solve_melitz_middle_exact_hvp_polish: $reason trial rejected (Addendum A convention)."))
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        x = collect(evalRequest.x)
        n_fc_calls[] += 1
        r = _eval(x)
        evalResult.obj[1] = r.Delta
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        x = collect(evalRequest.x)
        n_ga_calls[] += 1
        r = _eval(x)
        evalResult.objGrad .= r.grad_free
        return 0
    end

    # Per-point Hessian cache: rebuilt only when the requested x is a genuinely new point.
    last_key = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    last_pt = Ref{Union{Nothing,MelitzExactHessianPointState}}(nothing)
    last_report = Ref{Union{Nothing,MelitzHxxStabilityReport}}(nothing)

    function cb_Hv!(kc2, cb, evalRequest, evalResult, userParams)
        n_hv_calls[] += 1
        x = collect(evalRequest.x)
        v_free = collect(evalRequest.vec)
        sigma = evalRequest.sigma

        A_free = coordinate == :logH ? melitz_A_free_from_h_free(x, ctx) : x
        theta_free_middle = melitz_fixed_q_state_theta(A_free, q_fixed, gpj_fixed, ctx)
        key = Vector{Float64}(theta_free_middle)

        if last_key[] === nothing || last_key[] != key
            # Ensure the live operator reflects THIS point regardless of whatever the most
            # recent FC/GA/cache activity left it at (cheap, O(W*D), no KNITRO solve).
            melitz_update_operator_at_theta!(session.obj.op, theta_free_middle, ctx)
            hit = melitz_exact_cache_get(exact_cache, key, ctx, session.obj.U; obj=session.obj)
            local x_dual
            if hit !== nothing
                _Delta_hit, x_dual, _nStatus_hit, _H_hit = hit
            elseif haskey(bad_cache, key)
                throw(DomainError(x, "solve_melitz_middle_exact_hvp_polish: Hessian-vector product requested at a " *
                    "non-FiniteSolved (cap/infinite-certified) point -- no verified dual to build curvature at."))
            else
                r = classify!(x)
                r.classification_sym == :FiniteSolved || throw(DomainError(x,
                    "solve_melitz_middle_exact_hvp_polish: Hessian-vector product requested at a point that " *
                    "just classified non-FiniteSolved -- no verified dual to build curvature at."))
                melitz_update_operator_at_theta!(session.obj.op, theta_free_middle, ctx)
                hit2 = melitz_exact_cache_get(exact_cache, key, ctx, session.obj.U; obj=session.obj)
                hit2 === nothing && throw(DomainError(x,
                    "solve_melitz_middle_exact_hvp_polish: internal invariant violated -- a FiniteSolved point " *
                    "was not found in exact_cache immediately after being classified."))
                _Delta_hit2, x_dual, _nStatus_hit2, _H_hit2 = hit2
            end

            state = MelitzExpandedState(D)
            ws_exp = MelitzThetaExpansionWorkspace(D)
            melitz_expand_theta!(state, theta_free_middle, ctx, ws_exp)
            pt = build_melitz_exact_hessian_point_state(session.obj, x_dual, ctx, state)
            report = melitz_hxx_stability_report(pt, zeros(pt.K + 1); rel_thresholds=cfg.rel_thresholds,
                null_threshold=cfg.null_threshold, sym_tol=cfg.sym_tol)

            last_key[] = key
            last_pt[] = pt
            last_report[] = report
        end

        pt = last_pt[]::MelitzExactHessianPointState
        report = last_report[]::MelitzHxxStabilityReport

        v_full_vec = M_A * v_free
        v_full = reshape(v_full_vec, D, D)
        h_full = zeros(D, D)
        _, bv = melitz_profiled_A_hvp_full_stabilized!(h_full, pt, ctx, v_full, report;
            strategy=cfg.reg_strategy, lm_floor_frac=cfg.lm_floor_frac, trunc_rel_threshold=cfg.trunc_rel_threshold)
        if norm(bv) > 0
            null_mask = report.eigvals .< (report.null_threshold * report.eigmax)
            if any(null_mask)
                nf = norm(view(report.eigvecs, :, null_mask)' * bv) / norm(bv)
                nf > cfg.max_nullspace_frac * 3 && (n_gate_reject[] += 1)  # diagnostic counter only
            end
        end
        hv_free = M_A' * vec(h_full)
        evalResult.hessVec .= sigma .* hv_free
        return 0
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, cfg.polish_opt_file)
    KNITRO.KN_set_int_param(kc, KNITRO.KN_PARAM_HESSOPT, KNITRO.KN_HESSOPT_PRODUCT)
    xIndices = melitz_kn_add_vars!(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, x_start_v .- cfg.polish_box)
    KNITRO.KN_set_var_upbnds_all(kc, x_start_v .+ cfg.polish_box)
    KNITRO.KN_set_var_primal_init_values_all(kc, x_start_v)

    m = length(rhs)
    if m > 0
        cIndices = melitz_kn_add_cons!(kc, m)
        for i in 1:m
            if sys.sense[i] == :eq
                KNITRO.KN_set_con_eqbnd(kc, cIndices[i], rhs[i])
            else
                KNITRO.KN_set_con_lobnd(kc, cIndices[i], rhs[i])
            end
        end
        nnz = m * n
        indexCons_lin = repeat(cIndices, inner=n)
        indexVars_lin = repeat(xIndices, outer=m)
        coefs_lin = vec(permutedims(rows))
        KNITRO.KN_add_con_linear_struct(kc, nnz, indexCons_lin, indexVars_lin, coefs_lin)
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!; nnzJ=0)
    KNITRO.KN_set_cb_hess(kc, cb, 0, cb_Hv!)
    melitz_apply_policy_to_knitro!(kc, policy)

    nStatus = -999
    x_final = copy(x_start_v)
    try
        KNITRO.KN_solve(kc)
        nStatus_ref, objSol, xSol, _ = KNITRO.KN_get_solution(kc)
        nStatus = Int(nStatus_ref)
        x_final = collect(Float64.(xSol))
    catch e
        @warn "solve_melitz_middle_exact_hvp_polish: KN_solve terminated via exception" exception=(e, catch_backtrace())
    finally
        KNITRO.KN_free(kc)
    end

    function theta_of(x::Vector{Float64})
        Af = coordinate == :logH ? melitz_A_free_from_h_free(x, ctx) : x
        return melitz_fixed_q_state_theta(Af, q_fixed, gpj_fixed, ctx)
    end
    function cold_reverify(theta_free::Vector{Float64})
        return solve_melitz_delta!(session, theta_free, policy; warm_start_source=:neutral,
                                    origin_block_screen=origin_block_screen)
    end

    theta_start = theta_of(x_start_v)
    r_start_cold = cold_reverify(theta_start)
    Delta_start_cold = r_start_cold isa FiniteSolved ? r_start_cold.Delta : Inf

    candidates = Tuple{Symbol,Vector{Float64},Float64,MelitzInnerResult}[]
    push!(candidates, (:polish_start, theta_start, Delta_start_cold, r_start_cold))
    if incumbent_theta[] !== nothing
        r_trial_cold = cold_reverify(incumbent_theta[])
        Delta_trial_cold = r_trial_cold isa FiniteSolved ? r_trial_cold.Delta : Inf
        push!(candidates, (:polish_trial, incumbent_theta[], Delta_trial_cold, r_trial_cold))
    end
    theta_terminal = theta_of(x_final)
    r_terminal_cold = cold_reverify(theta_terminal)
    Delta_terminal_cold = r_terminal_cold isa FiniteSolved ? r_terminal_cold.Delta : Inf
    push!(candidates, (:polish_terminal, theta_terminal, Delta_terminal_cold, r_terminal_cold))

    deltas = [c[3] for c in candidates]
    best_i = argmin(deltas)
    best_source, best_theta, best_Delta, best_r = candidates[best_i]

    nA = length(ctx.A_pivot.other)
    A_free_incumbent = best_theta[2:1+nA]

    return MelitzHvpPolishResult(nStatus, best_source, A_free_incumbent, best_theta, best_r,
        best_Delta, Delta_start_cold, Delta_terminal_cold, n_fc_calls[], n_ga_calls[], n_hv_calls[],
        n_finite[], n_cap[], n_inf[], time() - t0, n_gate_reject[])
end

# ----------------------------------------------------------------------------------------
# Top-level hybrid orchestrator (governing prompt's own required comparison "B"): Phase 1
# L-BFGS (unchanged solve_melitz_fixed_q_A_profile_v2 call) -> gate check at the incumbent ->
# Phase 2 exact-HVP polish, only if the gate passes -> strict incumbent retention across BOTH
# phases plus the original input start.
# ----------------------------------------------------------------------------------------

"""
    MelitzHybridHvpResult

Full result of `solve_melitz_fixed_q_A_profile_hybrid_hvp`. `phase1` is the ordinary
`MelitzMiddleProfileV2Result` (unchanged L-BFGS driver). `gate` records the one-time
gate-check outcome. `phase2` is `nothing` iff the gate did not pass (`gate.ok==false`).
`Delta_final`/`A_free_final`/`theta_free_final`/`r_final`/`final_source` are the OVERALL best
verified candidate across phase 1's own start/trial/terminal AND (if run) phase 2's own
start/trial/terminal -- `Delta_final <= phase1.Delta_start_verified` is asserted (never worse
than the verified input start, matching `_v2`'s own guarantee). `unique_A_points`/
`unique_inner_solves` are CUMULATIVE across both phases (shared caches); `wall_s` is the total
wall time of this ONE hybrid call (phase 1 + gate + phase 2, if run).
"""
struct MelitzHybridHvpResult
    phase1::MelitzMiddleProfileV2Result
    gate::MelitzHvpGateResult
    phase2::Union{Nothing,MelitzHvpPolishResult}
    final_source::Symbol
    Delta_final::Float64
    A_free_final::Vector{Float64}
    theta_free_final::Vector{Float64}
    r_final::MelitzInnerResult
    unique_A_points::Int
    unique_inner_solves::Int
    cache_hits::Int
    wall_s::Float64
end

"""
    solve_melitz_fixed_q_A_profile_hybrid_hvp(session, q_fixed, gamma_prime_j_fixed, x_start, ctx;
        coordinate=:logA, policy=session.policy, max_evals=120, box=0.1,
        outer_loop_opt=<v2 default>, warm_start_source=:previous, origin_block_screen=false,
        sys=nothing, theta_fixed_q_for_constraints=nothing, cap_handling=:reject,
        cap_barrier_multiple=5.0, hvp_cfg=MelitzExactHvpPolishConfig()) -> MelitzHybridHvpResult

The full hybrid (governing prompt's required comparison "B"). Runs phase 1 EXACTLY as
`solve_melitz_fixed_q_A_profile_v2` would (same signature, same defaults for every phase-1
kwarg); the only difference from calling `_v2` directly is that this function ALSO evaluates
the gate at phase 1's own incumbent and, if it passes, runs phase 2
(`solve_melitz_middle_exact_hvp_polish`) starting from that SAME incumbent, sharing phase 1's
own `exact_cache`/`bad_cache`/`stats` (never resetting `session.obj.use_cached_x`/`.x` in
between -- requirement 8's continuation warm start). The returned `Delta_final` is the best
across ALL cold-verified candidates from both phases plus the original input start.
"""
function solve_melitz_fixed_q_A_profile_hybrid_hvp(session::MelitzInnerSession, q_fixed::AbstractMatrix{Float64},
                                                     gamma_prime_j_fixed::Real, x_start::AbstractVector{Float64}, ctx;
                                                     coordinate::Symbol=:logA,
                                                     policy::MelitzInnerSolvePolicy=session.policy,
                                                     max_evals::Int=120,
                                                     box::Real=0.1,
                                                     outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..", "melitz_middle_loop_opt_2026-07-30.opt"),
                                                     warm_start_source::Symbol=:previous,
                                                     origin_block_screen::Bool=false,
                                                     sys::Union{Nothing,MelitzFixedQMiddleConstraintSystem}=nothing,
                                                     theta_fixed_q_for_constraints::Union{Nothing,AbstractVector}=nothing,
                                                     cap_handling::Symbol=:reject,
                                                     cap_barrier_multiple::Real=5.0,
                                                     hvp_cfg::MelitzExactHvpPolishConfig=MelitzExactHvpPolishConfig())
    t0 = time()
    gpj_fixed = Float64(gamma_prime_j_fixed)

    if sys === nothing
        theta_fixed_q_for_constraints === nothing && throw(ArgumentError(
            "solve_melitz_fixed_q_A_profile_hybrid_hvp: pass either `sys` or `theta_fixed_q_for_constraints`."))
        sys = melitz_fixed_q_middle_constraint_system(theta_fixed_q_for_constraints, ctx, session.obj)
    end

    exact_cache = MelitzExactPointCache()
    bad_cache = MelitzMiddleBadPointCache()
    stats = MelitzMiddleCacheStats()

    r1 = solve_melitz_fixed_q_A_profile_v2(session, q_fixed, gpj_fixed, x_start, ctx;
        coordinate=coordinate, policy=policy, max_evals=max_evals, box=box, outer_loop_opt=outer_loop_opt,
        warm_start_source=warm_start_source, origin_block_screen=origin_block_screen, sys=sys,
        cap_handling=cap_handling, cap_barrier_multiple=cap_barrier_multiple,
        exact_cache=exact_cache, bad_cache=bad_cache, stats=stats)

    candidates = Tuple{Symbol,Vector{Float64},Float64,MelitzInnerResult}[]
    push!(candidates, (Symbol("phase1_", r1.incumbent_source), r1.theta_free_incumbent, r1.Delta_incumbent, r1.r_incumbent))

    gate = if !(r1.r_incumbent isa FiniteSolved)
        MelitzHvpGateResult(false, false, :phase1_not_finite, nothing, NaN, r1.nStatus)
    else
        D = ctx.D
        state = MelitzExpandedState(D)
        ws_exp = MelitzThetaExpansionWorkspace(D)
        melitz_expand_theta!(state, r1.theta_free_incumbent, ctx, ws_exp)
        melitz_update_operator_at_theta!(session.obj.op, r1.theta_free_incumbent, ctx)
        ws_grad = MelitzExactAGradientWorkspace(session.obj.op)
        grad_full_A = zeros(D, D)
        melitz_exact_a_gradient_full!(grad_full_A, session.obj, r1.r_incumbent.x, state, ctx, ws_grad)
        pt = build_melitz_exact_hessian_point_state(session.obj, r1.r_incumbent.x, ctx, state)
        melitz_hvp_gate_check(pt, ctx, grad_full_A, r1.nStatus, hvp_cfg)
    end

    phase2 = nothing
    if gate.ok
        nA = length(ctx.A_pivot.other)
        x_start_polish = coordinate == :logH ?
            melitz_h_free_from_A_free(r1.A_free_incumbent, ctx) : r1.A_free_incumbent
        phase2 = solve_melitz_middle_exact_hvp_polish(session, q_fixed, gpj_fixed, x_start_polish, ctx, sys,
            exact_cache, bad_cache, stats, hvp_cfg; coordinate=coordinate, policy=policy,
            warm_start_source=warm_start_source, origin_block_screen=origin_block_screen)
        push!(candidates, (Symbol("phase2_", phase2.incumbent_source), phase2.theta_free_incumbent,
            phase2.Delta_incumbent, phase2.r_incumbent))
    end

    deltas = [c[3] for c in candidates]
    best_i = argmin(deltas)
    final_source, best_theta, best_Delta, best_r = candidates[best_i]

    @assert best_Delta <= r1.Delta_start_verified + 1e-6 (
        "solve_melitz_fixed_q_A_profile_hybrid_hvp: strict incumbent retention violated -- " *
        "returned Delta=$best_Delta > verified start Delta=$(r1.Delta_start_verified)")

    nA = length(ctx.A_pivot.other)
    A_free_final = best_theta[2:1+nA]

    return MelitzHybridHvpResult(r1, gate, phase2, final_source, best_Delta, A_free_final, best_theta,
        best_r, length(stats.seen_keys), stats.n_actual_inner_solves, stats.n_cache_hits + stats.n_bad_point_hits,
        time() - t0)
end
