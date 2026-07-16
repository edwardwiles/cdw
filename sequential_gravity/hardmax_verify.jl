# Hard-max (rho=0) verification of a converged production point.
#
# Background: the sequential/profiled pipeline matches non-focal ("omitted") destinations'
# trade shares by inverting for a competitiveness vector u under a SMOOTHED (softmax, rho=2e-3)
# model -- necessary for `invert_destination`'s Newton machinery to have a usable gradient. This
# file provides an INDEPENDENT check, run against the TRUE hard-argmax model (rho=0), of any
# point this pipeline is about to report as its answer. See
# derivative_diagnostics/hardmax_inversion_report.md for the full validation writeup (LP-duality
# approach tried and found wrong for this model; rho-continuation homotopy, used here, works and
# was validated on 3 independently-audited real-data example points) and project memory
# `hardmax-inversion-validation` / `optim-trustregion-production-solver`.
#
# `hardmax_invert_destination` here is the validated solver (unchanged from the diagnostics
# version); `verify_hardmax_point` is the new production entry point, called from
# `run_one_bound` after the outer loop settles on a best-feasible theta -- see that function for
# how the result is surfaced (a NEW, separate `hardmax_verified` flag; does NOT change the
# meaning of the existing smoothed `gravity_feasible`/`best_feasible_gravity_ok` fields).

"""
    hardmax_invert_destination(log_x, p, λ̂; ref=1, ρ0=2e-3, ρ_floor=1e-8, u_init=nothing,
                               tol=1e-6, tol_hard=1e-4, maxit=150, ls_iters=100)

Hard-max (rho=0) destination inversion via rho-continuation. Solves `invert_destination` at a
geometrically-decreasing rho schedule from `ρ0` down to `ρ_floor`, warm-starting each step from
the previous solution (or from `u_init` at the first step, defaulting to zeros), then attempts
one final native rho=0 solve warm-started from the `ρ_floor` endpoint.

Returns a NamedTuple: `u` (the final u, gauge u[ref]=0), `hard_shares` (dest_share(...;rho=0) at
`u`), `hard_err` (max abs share error under STRICT hard-argmax -- the honest metric; may be a
genuine nonzero structural floor, NOT necessarily drivable to the `tol_hard` tolerance -- see
the module docstring above), `rho0_converged` (whether the FINAL native rho=0 solve satisfied
its own tol_hard criterion -- false does not imply failure, see above), `homotopy_ok` (whether
every step of the rho>0 homotopy itself converged; a false here IS a genuine red flag), and
`schedule_rows` (per-step diagnostics: rho, converged, iters, hard_err at that step -- useful
for auditing whether the floor was reached early or the destination is unusually hard).
"""
function hardmax_invert_destination(log_x::AbstractMatrix, p::AbstractVector, λ̂::AbstractVector;
        ref::Int = 1, ρ0::Real = 2e-3, ρ_floor::Real = 1e-8,
        u_init::Union{Nothing,AbstractVector} = nothing,
        tol::Real = 1e-6, tol_hard::Real = 1e-4, maxit::Int = 150, ls_iters::Int = 100)
    S, D = size(log_x)
    logp = log.(p)
    # geometric schedule from ρ0 down to ρ_floor, 3 substeps per decade
    n_decades = log10(ρ0 / ρ_floor)
    n_steps = max(1, round(Int, 3 * n_decades))
    schedule = [ρ0 * (ρ_floor / ρ0)^(i / n_steps) for i in 0:n_steps]

    u_cur = u_init === nothing ? zeros(Float64, D) : copy(u_init)
    if u_cur[ref] != 0.0
        u_cur = u_cur .- u_cur[ref]
    end
    rows = NamedTuple[]
    homotopy_ok = true
    for ρi in schedule
        inv = invert_destination(log_x, p, λ̂; ref = ref, ρ = ρi, tol = tol, maxit = maxit,
            ls_iters = ls_iters, u_init = u_cur)
        hard_shares, _ = dest_share(log_x, logp, inv.u_full; ρ = 0.0)
        hard_err = maximum(abs.(hard_shares .- λ̂))
        push!(rows, (ρ = ρi, converged = inv.converged, iters = inv.iterations,
            own_share_err = inv.max_abs_share_error, hard_err = hard_err))
        inv.converged || (homotopy_ok = false)
        u_cur = copy(inv.u_full)
    end

    # final native rho=0 attempt, warm-started, with a LOOSER tolerance (rho=0's achievable
    # floor need not reach the rho>0 tol used above -- see module docstring)
    inv0 = invert_destination(log_x, p, λ̂; ref = ref, ρ = 0.0, tol = tol_hard, maxit = maxit,
        ls_iters = ls_iters, u_init = u_cur)
    hard_shares0, _ = dest_share(log_x, logp, inv0.u_full; ρ = 0.0)
    hard_err0 = maximum(abs.(hard_shares0 .- λ̂))
    push!(rows, (ρ = 0.0, converged = inv0.converged, iters = inv0.iterations,
        own_share_err = inv0.max_abs_share_error, hard_err = hard_err0))

    return (u = inv0.u_full, hard_shares = hard_shares0, hard_err = hard_err0,
        rho0_converged = inv0.converged, homotopy_ok = homotopy_ok, schedule_rows = rows)
end

"""
    verify_hardmax_point(θ, umat, p; ref=1, tol=5e-4)

Independent verification of a converged production point against the TRUE hard-max economic
model, given the (theta, umat, p) a `seq_gravcol` call already produced (no extra KNITRO call
needed -- `p` is reused as-is, not re-derived):

  1. Focal trade shares under hard-argmin (closed form via `dest_share(...;rho=0)` on
     `umat[:,focal]` -- already exact by construction, checked here as a sanity floor, not
     re-derived).
  2. Each non-focal ("omitted") destination's trade shares under a GENUINE hard-max inversion
     (`hardmax_invert_destination`, warm-started from that destination's own smoothed column of
     `umat` -- the realistic, cheap warm start, not a cold start).
  3. The resulting hard-max `u_mat`'s gravity-equation residual.

Returns a NamedTuple with `verified::Bool` (the hard-max `u_mat`'s gravity residual is within
`tol` -- the SAME tolerance and the SAME `abs(R_mean) <= tol` convention `seq_gravcol`'s own
(smoothed) `gravity_ok` uses, so the two are directly comparable), plus full diagnostics:
`focal_err`, `R_mean_hardmax`, `max_hard_err`/`mean_hard_err` (the per-destination hard-max
share-matching gap -- informational: per `hardmax-inversion-validation` project memory, this has
a genuine nonzero structural floor that GROWS with divergence stress, so a nonzero value here is
expected and is NOT by itself evidence of a problem; `verified` deliberately does not gate on
it, only on the gravity-moment consequence, which is what the outer optimization actually
targets), `homotopy_all_ok` (did every destination's rho-continuation itself converge cleanly --
an unexpected false here, unlike a nonzero `max_hard_err`, IS worth investigating), `umat_hard`
(the full hard-max competitiveness matrix, for further inspection/saving), and `wall`.
"""
function verify_hardmax_point(θ::AbstractVector, umat::AbstractMatrix, p::AbstractVector;
        ref::Int = 1, tol::Real = 5e-4)
    t0 = time()
    log_x = build_log_x(Uσ, θ[1])
    logp = log.(p)

    focal_shares, _ = dest_share(log_x, logp, umat[:, focal]; ρ = 0.0)
    focal_err = maximum(abs.(focal_shares .- λData[:, focal]))

    umat_hard = copy(Matrix(umat))
    hard_errs = fill(NaN, D)
    homotopy_ok_vec = trues(D)
    if PARALLEL_INVERSION
        Threads.@threads for i in eachindex(omitted)
            d = omitted[i]
            hm = hardmax_invert_destination(log_x, p, λData[:, d]; ref = ref, u_init = umat[:, d])
            umat_hard[:, d] = hm.u
            hard_errs[d] = hm.hard_err
            homotopy_ok_vec[d] = hm.homotopy_ok
        end
    else
        for d in omitted
            hm = hardmax_invert_destination(log_x, p, λData[:, d]; ref = ref, u_init = umat[:, d])
            umat_hard[:, d] = hm.u
            hard_errs[d] = hm.hard_err
            homotopy_ok_vec[d] = hm.homotopy_ok
        end
    end

    gr_hard = gravity_residual(umat_hard, logτ, logw, σ)
    verified = abs(gr_hard.R_mean) <= tol

    return (verified = verified, focal_err = focal_err, R_mean_hardmax = gr_hard.R_mean,
        max_hard_err = maximum(hard_errs[omitted]), mean_hard_err = sum(hard_errs[omitted]) / (D - 1),
        hard_errs = hard_errs, homotopy_all_ok = all(homotopy_ok_vec[omitted]),
        umat_hard = umat_hard, wall = time() - t0)
end
