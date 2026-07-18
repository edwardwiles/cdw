# ============================================================================
# Diagnose evaluate_fullA's inner_status=-300 at a reconstructed sequential
# point: is it genuine hard-max moment-matching infeasibility (the sequential
# method's destination inversions were solved under rho=2e-3 SOFTMAX, which
# only APPROXIMATELY matches the observed shares under the codebase's HARD
# argmax winner rule -- an expected, documented gap per
# sequential_methodology.tex's own "Hard-max (rho->0) verification" section),
# or a KNITRO-solver-only artifact (in which case the direct LP below --
# independent of KNITRO's dual solve, reusing phaseF_primal_feasibility_lp.jl's
# already-validated lp_feasibility_check, not re-derived) would report
# FEASIBLE_LP despite KNITRO's -300.
# ============================================================================
include(joinpath(@__DIR__, "phase5_sequential_reconstruction.jl"))
include(joinpath(@__DIR__, "winners.jl"))
using JuMP, HiGHS, JLD2, Printf

function lp_feasibility_check(xf::Vector{Float64}, ctx, label::String)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    W = size(ctx.obj.U, 1); d = ctx.obj.d
    K = zeros(W); G = zeros(W, d)
    ctx.obj.moments!(K, G, θ_full, ctx.obj.U, ctx.obj)

    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, m[1:W] >= 0)
    @constraint(model, sum(m) / W == 1)
    @constraint(model, moments[j=1:d], sum(m[s] * G[s, j] for s in 1:W) / W == 0)
    @objective(model, Min, 0)
    optimize!(model)
    status = termination_status(model)
    feasible_exact = status == MOI.OPTIMAL

    phase1_max_resid = NaN
    if !feasible_exact
        model2 = Model(HiGHS.Optimizer); set_silent(model2)
        @variable(model2, m2[1:W] >= 0)
        @variable(model2, t >= 0)
        @constraint(model2, sum(m2) / W == 1)
        @constraint(model2, [j=1:d], sum(m2[s] * G[s, j] for s in 1:W) / W <= t)
        @constraint(model2, [j=1:d], sum(m2[s] * G[s, j] for s in 1:W) / W >= -t)
        @objective(model2, Min, t)
        optimize!(model2)
        phase1_max_resid = termination_status(model2) == MOI.OPTIMAL ? value(t) : NaN
        # also report divergence of the phase-I minimizer, for context (not a delta<=1 certificate,
        # phase-I minimizes moment residual, not divergence -- but informative)
    end

    classification = if feasible_exact
        "FEASIBLE_LP"
    elseif isfinite(phase1_max_resid) && phase1_max_resid > 1e-6
        "CERTIFIED_INFEASIBLE (phase-I min max-residual = $(round(phase1_max_resid, sigdigits=6)) > 0)"
    elseif isfinite(phase1_max_resid)
        "NUMERICALLY_BORDERLINE (phase-I max-residual ~ $(round(phase1_max_resid, sigdigits=6)))"
    else
        "NUMERICALLY_UNRESOLVED"
    end
    @printf("  [%s] LP1_status=%s classification=%s\n", label, string(status), classification)
    return (label = label, feasible_exact = feasible_exact, phase1_max_resid = phase1_max_resid, classification = classification, G = G)
end

const SEQ_BATCH_ROOT = length(ARGS) >= 1 ? ARGS[1] : error("usage: julia phase5_lp_diagnose.jl <batch dir> <start_id> <bound>")
const SID = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const BOUND = length(ARGS) >= 3 ? ARGS[3] : "upper"

path = joinpath(SEQ_BATCH_ROOT, "start$SID", "seq_$(BOUND)_delta1.jld2")
d = JLD2.load(path)
θseq = d["best_feasible_theta"]
@printf("Loaded start%d %s: best_feasible_kappa=%.6f theta_seq=%s\n", SID, BOUND, d["best_feasible_kappa"], string(θseq))

find_smallest = BOUND == "upper"
rc = build_reconstruction_context(; δ = 1.0, find_smallest = find_smallest)
x_free, diag = reconstruct_fullA_point(rc, θseq; δ = 1.0)
@printf("Focal round-trip max abs err = %.3e ; seq re-solve gravity_ok=%s divergence(p)=%.4f\n",
        diag.focal_roundtrip_max_abs_err, diag.seq_gravity_ok, diag.seq_divergence_p)

println("\n--- Direct LP feasibility check on the reconstructed point (independent of KNITRO) ---")
r = lp_feasibility_check(x_free, rc.ctx, "reconstructed_$(BOUND)_start$SID")

θ_full = CS.reconstruct_full(x_free, rc.ctx.m)
K = zeros(rc.W); G = zeros(rc.W, rc.ctx.obj.d)
rc.ctx.obj.moments!(K, G, θ_full, rc.ctx.obj.U, rc.ctx.obj)

# Residual under the SEQUENTIAL METHOD'S OWN converged weights p_seq (diag.p) -- the most direct
# test of "is this a reconstruction-formula bug (even the 'right' p fails) vs a reweighting/
# LP-optimum mismatch (the LP just finds a DIFFERENT, better p than p_seq)". p_seq was calibrated
# to match the REDUCED (sequential) moment system under the SOFTMAX (rho=2e-3) destination
# inversions; evaluating it against the FULL hard-max D^2 moment system here is the direct
# smoothed-vs-hardmax comparison sequential_methodology.tex's own "Hard-max verification" section
# describes as expected to show *some* nonzero gap, not necessarily zero.
resid_pseq = vec(sum(diag.p .* G, dims=1))
@printf("\n  Moment residuals under p_seq (sequential method's own converged LFD weights): max|resid|=%.4e\n", maximum(abs.(resid_pseq)))
println("  Per-moment residual under p_seq:")
for (j, v) in enumerate(resid_pseq)
    @printf("    moment %2d: %+.4e\n", j, v)
end

if !r.feasible_exact
    # Break down: is the mismatch concentrated in a few moments (consistent with hard/soft winner
    # discrepancy at specific near-tied draws) or spread uniformly?
    resid_uniform = vec(sum(G, dims=1)) ./ rc.W
    @printf("\n  Moment residuals at UNIFORM p=1/W (hard-max, no reweighting): max|resid|=%.4e\n", maximum(abs.(resid_uniform)))
    println("  Per-moment residual at uniform p:")
    for (j, v) in enumerate(resid_uniform)
        @printf("    moment %2d: %+.4e\n", j, v)
    end
end
println("\n--- lambda (data trade share) matrix, wHat, tau ---")
println("lambda ="); display(rc.λData); println()
println("wHat = ", rc.wHat)
println("tau ="); display(rc.τ); println()
println("\n--- umat (reconstructed competitiveness matrix, destination-inversion gauge for omitted cols) ---")
display(diag.umat); println()
println("\n--- Aod_theta (reconstructed full-A free parameter matrix) ---")
display(diag.Aod_theta); println()
println("\n--- destination-inversion convergence check (fresh, verbose) for d=4 specifically ---")
log_x = build_log_x(rc.Uσ, θseq[1])
inv4 = invert_destination(log_x, diag.p, rc.λData[:,4]; ref=rc.ref, ρ=rc.ρ, tol=1e-10, maxit=300, ls_iters=80, verbose=true)
@printf("d=4 inversion: converged=%s iters=%d max_abs_share_error=%.3e u_full=%s\n", inv4.converged, inv4.iterations, inv4.max_abs_share_error, string(inv4.u_full))
@printf("d=4 model shares = %s   vs target lambda[:,4] = %s\n", string(inv4.model_shares), string(rc.λData[:,4]))

println("\n--- HARD-MAX (rho=0) re-inversion for d=4, warm-started from the rho=0.002 solution ---")
inv4_hard = invert_destination(log_x, diag.p, rc.λData[:,4]; ref=rc.ref, ρ=0.0, tol=1e-10, maxit=300, ls_iters=80,
                                u_init=inv4.u_full, verbose=true)
@printf("d=4 HARD-MAX inversion: converged=%s iters=%d max_abs_share_error=%.3e u_full=%s\n",
        inv4_hard.converged, inv4_hard.iterations, inv4_hard.max_abs_share_error, string(inv4_hard.u_full))
@printf("u shift (hard - soft) = %s\n", string(inv4_hard.u_full .- inv4.u_full))

println("\n--- Full hard-max re-inversion of ALL omitted columns, rebuild Aod_theta, re-check LP ---")
umat_hard = copy(diag.umat)
for dd in rc.omitted
    invd_hard = invert_destination(log_x, diag.p, rc.λData[:,dd]; ref=rc.ref, ρ=0.0, tol=1e-10, maxit=300, ls_iters=80,
                                    u_init=diag.umat[:,dd])
    @printf("  d=%d hard-max reinversion: converged=%s share_err=%.3e\n", dd, invd_hard.converged, invd_hard.max_abs_share_error)
    umat_hard[:, dd] .= invd_hard.u_full
end
Aod_theta_hard = copy(diag.Aod_theta)
for dd in rc.omitted, o in 1:rc.D
    AodPow_od = exp(-(umat_hard[o, dd]/(rc.σ-1) + rc.logw[o] + rc.logτ[o, dd]))
    Aod_theta_hard[o, dd] = aod_theta_from_AodPow(AodPow_od, o, dd, θseq[1], rc.wHat, rc.τ, rc.λData)
end
θ_full_hard = copy(diag.θ_full)
θ_full_hard[rc.ctx.Aod_offset+1:rc.ctx.Aod_offset+rc.D^2] .= vec(Aod_theta_hard)
x_free_hard = CS.pack_free(θ_full_hard, rc.ctx.m)
r_hard = lp_feasibility_check(x_free_hard, rc.ctx, "reconstructed_HARDMAX_$(BOUND)_start$SID")

K2 = zeros(rc.W); G2 = zeros(rc.W, rc.ctx.obj.d)
rc.ctx.obj.moments!(K2, G2, θ_full_hard, rc.ctx.obj.U, rc.ctx.obj)
resid_pseq_hard = vec(sum(diag.p .* G2, dims=1))
@printf("\n  [HARD-MAX reconstruction] Moment residuals under p_seq: max|resid|=%.4e\n", maximum(abs.(resid_pseq_hard)))
for (j, v) in enumerate(resid_pseq_hard)
    @printf("    moment %2d: %+.4e\n", j, v)
end

println("\n--- Direct hard-winner share check via winners.jl::compute_winners (validated vs hFunction!) ---")
winner, price, gap = compute_winners(diag.θ_full, rc.ctx)
for dd in 1:rc.D
    p_share_hard = [sum(diag.p[winner[:,dd] .== o]) for o in 1:rc.D]
    @printf("  d=%d: hard-max p_seq-weighted shares = %s   vs lambda[:,%d] = %s   (uniform-p hard shares = %s)\n",
            dd, string(round.(p_share_hard, digits=4)), dd, string(round.(rc.λData[:,dd], digits=4)),
            string(round.([count(==(o), winner[:,dd])/rc.W for o in 1:rc.D], digits=4)))
end

println("\nPHASE5_LP_DIAGNOSE_DONE")
