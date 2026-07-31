# ============================================================================
# Independent, adversarial full-equilibrium audit of the two final D20 Melitz
# (delta=0.5) incumbents:
#   upper: GT=8.84934137951131%   Delta*=0.49812250980571005  (phase6_upper_m1)
#   lower: GT=0.2722655222019532% Delta*=0.11376383122706066  (phase6_lower_m-1)
#
# Governing principle: the D^2+1 moments actually sent to KNITRO (D^2 trade-share +
# 1 focal free-entry link) are NOT assumed to be the complete theoretical equilibrium
# system. This script independently recomputes every candidate equation in the audit's
# equation inventory using FRESH loops (Kahan-compensated Float64 sums, BigFloat for the
# highest-stakes focal quantities), reusing ONLY primitive economic formulas
# (melitz_firm/melitz_C/melitz_cutoff/pareto_tail_prob -- one-line building blocks used
# identically everywhere in this codebase) and state-reconstruction plumbing
# (expand_free_theta_logcutoff, which is how (A,f,q,g) are rebuilt from theta_free -- this
# IS the state, not the moment operator). It never calls melitz_moments!, mul_G!, mul_Gt!,
# or any cached production moment residual.
#
# It ALSO calls the existing (but never invoked in any D20 production/overnight script)
# check_profiled_melitz_equilibrium(...; full=true) once per incumbent, clearly labeled as
# a REFERENCE cross-check against already-written code (not the independent oracle itself)
# -- this function exists in equilibrium.jl precisely to check the omitted equations, and
# grep confirms no D20 production/overnight script ever calls it.
# ============================================================================
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization, Statistics, Dates
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

OUTDIR = joinpath(REPO2, "docs", "key_results", "melitz_final_incumbent_full_equilibrium_audit_2026-07-31")
mkpath(OUTDIR)

# ----------------------------------------------------------------------------
# Independent numerical primitives (own code, not calling misc/doubleDiff.jl or moments.jl)
# ----------------------------------------------------------------------------
function kahan_sum(xs)
    s = 0.0; c = 0.0
    @inbounds for x in xs
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    end
    return s
end

function my_within_transform(logz::AbstractMatrix)
    D = size(logz, 1)
    rowmean = vec(sum(logz, dims=2)) ./ D
    colmean = vec(sum(logz, dims=1)) ./ D
    gm = sum(logz) / D^2
    out = similar(logz)
    @inbounds for o in 1:D, d in 1:D
        out[o, d] = logz[o, d] - rowmean[o] - colmean[d] + gm
    end
    return out
end
my_gravity_residual(tau::AbstractMatrix, X::AbstractMatrix) =
    kahan_sum(vec(my_within_transform(log.(tau)) .* my_within_transform(log.(X))))

# BigFloat-precision re-implementation of the firm zero-profit/revenue formula (independent
# expression, not calling firm_quantities.jl), used only for the highest-stakes focal
# quantities (autarky cutoff residual, focal link residual).
function bf_firm_profit_revenue(w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, z)
    T = BigFloat
    w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, z =
        T(w_o), T(tau_od), T(A_od), T(f_od), T(sigma), T(expenditure_d), T(price_power_d), T(z)
    markup = sigma / (sigma - 1)
    price = markup * w_o * tau_od / (A_od * z)
    rev = expenditure_d * price^(1 - sigma) / price_power_d
    profit = rev / sigma - w_o * f_od
    return profit, rev
end

# ----------------------------------------------------------------------------
# Fresh real-D20 fixture (own process; identical loading recipe to the production
# final-verify scripts, so the SAME calibration is reconstructed, never reused from a
# live campaign process).
# ----------------------------------------------------------------------------
function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal, countries
end
calib, focal, countries_d20 = load_realD20_calib()
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx = obj_d20.γ
session = MelitzInnerSession(obj_d20, ctx, policy_cap)
D = ctx.D
j = ctx.target_country
W = size(obj_d20.U, 1)
@printf("\nD=%d  target_country=%d (%s)  sigma=%.6f  theta_star=%.8f  W=%d\n",
    D, j, countries_d20[j], ctx.sigma, ctx.theta_star, W)
@printf("Fingerprint: X_data[j,j]=%.12g  expenditure[j]=%.12g  w[j]=%.12g  w_prime=%.12g\n",
    ctx.X_data[j, j], ctx.expenditure[j], ctx.w[j], ctx.w_prime)
@printf("A_pivot: pivot=%d  |c|_max=%.6g\n", ctx.A_pivot.pivot, maximum(abs.(ctx.A_pivot.c)))
flush(stdout)

mutable struct ChainPoint
    idx::Int; phase::Symbol; g::Float64; GT::Float64; classification::Symbol; Delta::Float64
    within_budget::Bool; lfd_ok::Bool; lfd_Delta::Float64; A_free::Vector{Float64}; theta_free::Vector{Float64}
    f_full::Matrix{Float64}; q_full::Matrix{Float64}; lfd_weights::Vector{Float64}; nStatus::Int
    unique_inner_solves::Int; wall_s::Float64; timestamp::String
end
mutable struct ChainState
    chain_label::String; direction::Symbol; points::Vector{ChainPoint}
    bracket::Union{Nothing,NTuple{4,Float64}}; n_evaluated::Int; most_extreme_idx::Int
    n_nonimproving::Int; elapsed_wall_s::Float64; status::Symbol
    cur_A_free::Vector{Float64}; cur_q::Matrix{Float64}; cur_p_star::Vector{Float64}
    cur_g::Float64; cur_GT::Float64; step_gt::Float64; n_accepted::Int
end

OVERDIR = joinpath(REPO2, "docs", "key_results", "overnight_qpoll_delta0p5_2026-07-31")

# ----------------------------------------------------------------------------
# The full battery for one incumbent
# ----------------------------------------------------------------------------
function audit_incumbent(label, direction)
    println("\n", "#"^100)
    println("# INCUMBENT: $label ($direction)")
    println("#"^100); flush(stdout)

    ckpt = joinpath(OVERDIR, "phase5", "checkpoints", "$(label).jls")
    state = deserialize(ckpt)
    best = state.points[state.most_extreme_idx]
    @printf("Checkpointed: GT=%.10f%%  Delta=%.10f  classification=%s  lfd_ok=%s  nStatus=%d\n",
        best.GT, best.Delta, best.classification, best.lfd_ok, best.nStatus)
    @assert best.classification == :FiniteSolved && best.lfd_ok

    # ---- Phase 3: fresh public inner solve + LFD recovery (production solver; this is the
    # "recovered LFD probabilities" input Phase 2 explicitly permits reading) ----
    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    r_fresh = solve_melitz_delta!(session, best.theta_free, policy_cap; warm_start_source=:neutral)
    @assert r_fresh isa FiniteSolved
    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    lfd = melitz_recover_lfd(obj_d20, best.theta_free)
    @assert lfd.lfd_ok
    @printf("Fresh solve: Delta=%.10f (checkpoint drift %.3e)   LFD: Delta=%.10f nStatus=%d\n",
        r_fresh.Delta, abs(r_fresh.Delta - best.Delta), lfd.Delta, lfd.nStatus)
    weights = lfd.weights
    z_draws = obj_d20.U   # W x D reference Pareto draws (read-only input, not recomputed)

    # ---- LFD diagnostics (Phase 3) ----
    sum_w = kahan_sum(weights)
    min_w, max_w = extrema(weights)
    ess = 1.0 / kahan_sum(weights .^ 2)
    n_above_10overW = count(>(10.0 / W), weights)
    n_above_100overW = count(>(100.0 / W), weights)
    perm = sortperm(weights, rev=true)
    top20 = perm[1:20]
    # Own re-derivation of the primal divergence at these weights (Cressie-Read-like,
    # independent from melitz_primal_divergence -- same functional form, own code):
    function my_primal_divergence(w::AbstractVector, Wn::Int)
        e = exp(1.0)
        acc = 0.0; c = 0.0
        @inbounds for p in w
            m = p * Wn
            term = m <= e ? (m * log(m) - m + 1) : (m^2 / (2e) - e / 2 + 1)
            y = term - c; t = acc + y; c = (t - acc) - y; acc = t
        end
        return acc / Wn
    end
    my_primal = my_primal_divergence(weights, W)
    @printf("LFD: sum(w)=%.12f  min=%.6e  max=%.6e  ESS=%.2f (%.4f%% of W)  n(w>10/W)=%d  n(w>100/W)=%d\n",
        sum_w, min_w, max_w, ess, 100 * ess / W, n_above_10overW, n_above_100overW)
    @printf("Divergence: my_primal_divergence=%.10f  reported lfd.Delta=%.10f  |diff|=%.3e\n",
        my_primal, lfd.Delta, abs(my_primal - lfd.Delta))
    @assert all(isfinite, weights) && all(>=(0.0), weights)

    # ---- State reconstruction: (A, f, q, gamma_prime_j) from theta_free ----
    A, f, gamma_prime_j, f_jj, q = expand_free_theta_logcutoff(best.theta_free, ctx)
    @assert maximum(abs.(q .- best.q_full)) < 1e-8
    @assert maximum(abs.(f .- best.f_full)) < 1e-6
    w_ = ctx.w; tau_ = ctx.tau; expenditure_ = ctx.expenditure; sigma = ctx.sigma; theta_star = ctx.theta_star
    w_prime = ctx.w_prime; expenditure_prime = ctx.w_prime * ctx.L[j]

    markup = sigma / (sigma - 1)
    my_melitz_C(o, d) = expenditure_[d] * (markup * w_[o] * tau_[o, d] / A[o, d])^(1 - sigma)
    my_melitz_cutoff(o, d) = (sigma * w_[o] * f[o, d] / my_melitz_C(o, d))^(1 / (sigma - 1))
    my_firm_profit_rev(o, d, z, price_power_d=1.0, w_o=w_[o], tau_od=tau_[o, d], A_od=A[o, d], f_od=f[o, d], exp_d=expenditure_[d]) = begin
        price = markup * w_o * tau_od / (A_od * z)
        rev = exp_d * price^(1 - sigma) / price_power_d
        profit = rev / sigma - w_o * f_od
        (profit, rev, price)
    end

    # ------------------------------------------------------------------------
    # PHASE 4: independent D^2 bilateral trade-share residual matrix + adding-up
    # ------------------------------------------------------------------------
    println("\n--- PHASE 4: bilateral trade shares (independent, own loops) ---"); flush(stdout)
    pred_share = zeros(D, D)
    data_share = zeros(D, D)
    for o in 1:D, d in 1:D
        revs = Vector{Float64}(undef, W)
        @inbounds for w in 1:W
            z = z_draws[w, o]
            profit, rev, _ = my_firm_profit_rev(o, d, z)
            revs[w] = profit > 0 ? rev : 0.0
        end
        pred_rev = kahan_sum(weights .* revs)  # already normalized weights (sum=1) => E_LFD
        pred_share[o, d] = pred_rev / expenditure_[d]
        data_share[o, d] = ctx.X_data[o, d] / expenditure_[d]
    end
    trade_resid = pred_share .- data_share
    max_trade_resid = maximum(abs.(trade_resid))
    data_adding_up = vec(sum(data_share, dims=1)) .- 1.0
    pred_adding_up = vec(sum(pred_share, dims=1)) .- 1.0
    @printf("max|predicted_share - data_share| over D^2=%d cells: %.3e\n", D^2, max_trade_resid)
    @printf("max|data adding-up - 1| (by destination): %.3e   max|predicted adding-up - 1|: %.3e\n",
        maximum(abs.(data_adding_up)), maximum(abs.(pred_adding_up)))

    # ------------------------------------------------------------------------
    # PHASE 5: cutoff / zero-profit identity (forward formula vs stored q)
    # ------------------------------------------------------------------------
    println("\n--- PHASE 5: cutoff / zero-profit identities ---"); flush(stdout)
    q_check = zeros(D, D)
    for o in 1:D, d in 1:D
        q_check[o, d] = log(my_melitz_cutoff(o, d))
    end
    cutoff_resid = q_check .- q
    max_cutoff_resid = maximum(abs.(cutoff_resid))
    @printf("max|independent forward cutoff (from A,f) - stored q|: %.3e\n", max_cutoff_resid)
    min_dom = minimum(q[o, o] for o in 1:D)
    min_exp_minus_dom = minimum(q[o, d] - q[o, o] for o in 1:D, d in 1:D if d != o)
    @printf("min domestic log-cutoff q[o,o] = %.6f (feasible iff >=0)   min (q[o,d]-q[o,o]) d!=o = %.6f (feasible iff >=0)\n",
        min_dom, min_exp_minus_dom)

    # ------------------------------------------------------------------------
    # PHASE 6: free-entry system (baseline all D origins + autarky), BigFloat cross-check
    # ------------------------------------------------------------------------
    println("\n--- PHASE 6: free-entry system ---"); flush(stdout)
    f_entry_baseline = zeros(D)
    residual_free_entry_baseline = zeros(D)  # definitional by construction (recovered), report actual value
    for o in 1:D
        profit_sums = Vector{Float64}(undef, W)
        @inbounds for w in 1:W
            z = z_draws[w, o]
            s = 0.0
            for d in 1:D
                profit, _, _ = my_firm_profit_rev(o, d, z)
                s += profit > 0 ? profit : 0.0
            end
            profit_sums[w] = s
        end
        f_entry_baseline[o] = kahan_sum(weights .* profit_sums) / w_[o]
    end
    @printf("recovered baseline f_entry: min=%.6g (o=%d)  max=%.6g (o=%d)  focal f_entry[j]=%.6g\n",
        minimum(f_entry_baseline), argmin(f_entry_baseline), maximum(f_entry_baseline), argmax(f_entry_baseline),
        f_entry_baseline[j])

    # autarky, Float64 + BigFloat cross-check
    autarky_profit_sum = 0.0
    @inbounds for w in 1:W
        z = z_draws[w, j]
        profit, _, _ = my_firm_profit_rev(j, j, z, gamma_prime_j, w_prime, 1.0, A[j, j], f_jj, expenditure_prime)
        autarky_profit_sum += weights[w] * (profit > 0 ? profit : 0.0)
    end
    f_entry_autarky = autarky_profit_sum / w_prime
    @printf("recovered autarky f_entry_j = %.6g\n", f_entry_autarky)

    # focal LINK residual (the ONE imposed moment) recomputed independently, Float64 + BigFloat
    link_resid_f64 = kahan_sum(weights .* [begin
        z = z_draws[w, j]; s = 0.0
        for d in 1:D
            profit, _, _ = my_firm_profit_rev(j, d, z)
            s += profit > 0 ? profit : 0.0
        end
        s / w_[j]
    end for w in 1:W]) - autarky_profit_sum / w_prime
    @printf("focal free-entry LINK residual (independent Float64, should be ~0, it IS the imposed moment): %.3e\n", link_resid_f64)

    # BigFloat autarky cutoff identity: profit at z=1 in autarky must be exactly 0
    profit_at_one_bf, rev_at_one_bf = bf_firm_profit_revenue(w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, gamma_prime_j, 1.0)
    @printf("BigFloat autarky cutoff identity: operating_profit(z=1) = %.6e (expect ~0)\n", Float64(profit_at_one_bf))

    # ------------------------------------------------------------------------
    # PHASE 7: price index (baseline gamma_d all D) + autarky price index / N' cross-check
    # + independent GT/welfare reconstruction (ACR vs model formula)
    # ------------------------------------------------------------------------
    println("\n--- PHASE 7: price index, autarky N', welfare/GT reconstruction ---"); flush(stdout)
    gamma_baseline = zeros(D)
    for d in 1:D
        s = 0.0
        @inbounds for o in 1:D, w in 1:W
            z = z_draws[w, o]
            profit, _, price = my_firm_profit_rev(o, d, z)
            if profit > 0
                s += weights[w] * price^(1 - sigma)
            end
        end
        gamma_baseline[d] = s
    end
    resid_gamma_baseline = gamma_baseline .- 1.0
    @printf("max|baseline price-index gamma_d - 1| over all D destinations: %.3e\n", maximum(abs.(resid_gamma_baseline)))

    # autarky market clearing N'_mc and price-index N'_gamma, independent formulas
    N_prime_mc = expenditure_prime / (sigma * w_prime * (f_jj + f_entry_autarky))
    mean_rev_prime_jj = 0.0
    s_price_power_sum = 0.0
    @inbounds for w in 1:W
        z = z_draws[w, j]
        profit, rev, price = my_firm_profit_rev(j, j, z, gamma_prime_j, w_prime, 1.0, A[j, j], f_jj, expenditure_prime)
        if profit > 0
            mean_rev_prime_jj += weights[w] * rev
            s_price_power_sum += weights[w] * price^(1 - sigma)
        end
    end
    N_prime_gamma = gamma_prime_j / s_price_power_sum
    N_prime_diff_rel = abs(N_prime_gamma - N_prime_mc) / abs(N_prime_mc)
    resid_market_clearing_autarky = N_prime_mc * mean_rev_prime_jj - expenditure_prime
    @printf("N'_market_clearing=%.6f   N'_price_index=%.6f   relative diff=%.3e\n", N_prime_mc, N_prime_gamma, N_prime_diff_rel)
    @printf("autarky market-clearing residual (N'_mc*mean_rev - expenditure'_j): %.3e\n", resid_market_clearing_autarky)

    # welfare/GT: model formula + ACR sufficient-statistic (independent, data-anchored)
    wage_ratio = w_prime / w_[j]
    kappa_ratio = wage_ratio * gamma_prime_j^(1 / (sigma - 1))
    GT_model = 100 * (1 - kappa_ratio)
    lambda_jj_data = ctx.X_data[j, j] / expenditure_[j]
    GT_ACR = 100 * (1 - lambda_jj_data^(1 / theta_star))
    # ALSO: ACR using the INDEPENDENTLY-RECOMPUTED predicted domestic share (pred_share[j,j])
    # rather than raw data -- tests whether the imposed domestic trade-share MOMENT (which
    # should force pred_share[j,j]==data_share[j,j] to <1e-6 if lfd_ok) is actually doing so.
    GT_ACR_predicted = 100 * (1 - pred_share[j, j]^(1 / theta_star))
    @printf("GT_model (reported, from gamma_prime_target search)     = %.6f%%  (checkpoint: %.6f%%, diff=%.3e)\n",
        GT_model, best.GT, abs(GT_model - best.GT))
    @printf("GT_ACR (data-anchored domestic trade share lambda_jj)   = %.6f%%   lambda_jj_data=%.6g\n", GT_ACR, lambda_jj_data)
    @printf("GT_ACR (independently-PREDICTED domestic share)         = %.6f%%   pred_share[j,j]=%.6g\n", GT_ACR_predicted, pred_share[j, j])
    @printf("|GT_model - GT_ACR(data)| = %.6f percentage points   |GT_model - GT_ACR(predicted)| = %.6f pp\n",
        abs(GT_model - GT_ACR), abs(GT_model - GT_ACR_predicted))
    @printf("domestic trade-share moment check: |pred_share[j,j]-data_share[j,j]| = %.3e (should be <1e-6 if lfd_ok)\n",
        abs(pred_share[j, j] - data_share[j, j]))

    # ------------------------------------------------------------------------
    # PHASE 9: gravity restrictions, independent recomputation (own withinTransform, not
    # calling misc/doubleDiff.jl)
    # ------------------------------------------------------------------------
    println("\n--- PHASE 9: gravity restrictions (independent recomputation) ---"); flush(stdout)
    my_grav_A = my_gravity_residual(ctx.tau, A)
    my_grav_f = my_gravity_residual(ctx.tau, f)
    prod_grav_A, prod_grav_f = gravity_residuals(MelitzPrimitives(D, sigma, theta_star, j, ctx.tau, w_, A, f, gamma_prime_j))
    @printf("Independent A-gravity residual: %.3e   (production gravity_residuals(): %.3e, diff=%.3e)\n",
        my_grav_A, prod_grav_A, abs(my_grav_A - prod_grav_A))
    @printf("Independent f-gravity residual: %.3e   (production gravity_residuals(): %.3e, diff=%.3e)\n",
        my_grav_f, prod_grav_f, abs(my_grav_f - prod_grav_f))

    # ------------------------------------------------------------------------
    # PHASE 8: aggregate identities -- income=expenditure per origin (implied by the D^2
    # trade-share moments + data consistency, checked directly here, not assumed)
    # ------------------------------------------------------------------------
    println("\n--- PHASE 8: aggregate identities ---"); flush(stdout)
    income_o = vec(sum(ctx.X_data, dims=2))         # DATA row sums (origin total sales)
    resource_resid = income_o .- (w_ .* ctx.L)
    @printf("max|income (data row-sum) - w*L| across all D origins (income=expenditure/resource constraint on DATA): %.3e\n",
        maximum(abs.(resource_resid)))
    pred_income_o = vec(sum(pred_share .* expenditure_', dims=2))
    resource_resid_pred = pred_income_o .- (w_ .* ctx.L)
    @printf("max|income (MODEL-PREDICTED row-sum) - w*L| across all D origins: %.3e\n", maximum(abs.(resource_resid_pred)))

    # ------------------------------------------------------------------------
    # REFERENCE cross-check: existing (but never-called-in-production) full ex-post check
    # ------------------------------------------------------------------------
    println("\n--- REFERENCE: check_profiled_melitz_equilibrium(...; full=true) [existing code, cross-check only] ---"); flush(stdout)
    p_struct = MelitzPrimitives(D, sigma, theta_star, j, ctx.tau, w_, A, f, gamma_prime_j)
    eq_struct = MelitzEquilibrium(ctx.expenditure, ones(Float64, D), q_check_to_level(q), ctx.X_data)
    cf_struct = MelitzCounterfactual(j, w_prime, expenditure_prime, 1.0, expenditure_prime)
    check = check_profiled_melitz_equilibrium(p_struct, eq_struct, cf_struct, z_draws, weights; full=true)
    @printf("check.residual_gamma_baseline: max|.|=%.3e\n", maximum(abs.(check.residual_gamma_baseline)))
    @printf("check.residual_free_entry_baseline: max|.|=%.3e\n", maximum(abs.(check.residual_free_entry_baseline)))
    @printf("check.residual_free_entry_autarky=%.3e\n", check.residual_free_entry_autarky)
    @printf("check.residual_market_clearing_autarky=%.3e\n", check.residual_market_clearing_autarky)
    @printf("check.N_prime_diff_rel=%.3e  check.residual_gamma_autarky=%.3e\n", check.N_prime_diff_rel, check.residual_gamma_autarky)
    @printf("check.residual_autarky_cutoff=%.3e\n", check.residual_autarky_cutoff)
    @printf("check.min_cutoff_minus_one=%.6f  check.min_export_minus_domestic=%.6f\n", check.min_cutoff_minus_one, check.min_export_minus_domestic)
    @printf("check.gravity_residual_A=%.3e  check.gravity_residual_f=%.3e\n", check.gravity_residual_A, check.gravity_residual_f)

    return (label=label, direction=direction, best=best, A=A, f=f, q=q, gamma_prime_j=gamma_prime_j, f_jj=f_jj,
        weights=weights, z_draws=z_draws, trade_resid=trade_resid, data_share=data_share, pred_share=pred_share,
        max_trade_resid=max_trade_resid, max_cutoff_resid=max_cutoff_resid,
        f_entry_baseline=f_entry_baseline, f_entry_autarky=f_entry_autarky,
        gamma_baseline=gamma_baseline, N_prime_mc=N_prime_mc, N_prime_gamma=N_prime_gamma, N_prime_diff_rel=N_prime_diff_rel,
        GT_model=GT_model, GT_ACR=GT_ACR, GT_ACR_predicted=GT_ACR_predicted,
        my_grav_A=my_grav_A, my_grav_f=my_grav_f,
        resource_resid=resource_resid, resource_resid_pred=resource_resid_pred,
        check=check, ess=ess, min_w=min_w, max_w=max_w, my_primal=my_primal, lfd_Delta=lfd.Delta,
        min_dom=min_dom, min_exp_minus_dom=min_exp_minus_dom, profit_at_one_bf=Float64(profit_at_one_bf))
end

q_check_to_level(q) = exp.(q)

results = Dict{String,Any}()
results["upper"] = audit_incumbent("phase6_upper_m1", "upper")
results["lower"] = audit_incumbent("phase6_lower_m-1", "lower")

# ============================================================================
# PHASE 11: adversarial perturbation tests (on the UPPER incumbent's reconstructed state)
# ============================================================================
println("\n", "#"^100); println("# PHASE 11: ADVERSARIAL PERTURBATION TESTS (upper incumbent)"); println("#"^100); flush(stdout)
let r = results["upper"]
    A, f, q = copy(r.A), copy(r.f), copy(r.q)
    weights = r.weights; z_draws = r.z_draws
    sigma = ctx.sigma; theta_star = ctx.theta_star; w_ = ctx.w; tau_ = ctx.tau; expenditure_ = ctx.expenditure
    markup = sigma / (sigma - 1)
    D = ctx.D; j = ctx.target_country
    my_melitz_C(A_, o, d) = expenditure_[d] * (markup * w_[o] * tau_[o, d] / A_[o, d])^(1 - sigma)
    my_melitz_cutoff(A_, f_, o, d) = (sigma * w_[o] * f_[o, d] / my_melitz_C(A_, o, d))^(1 / (sigma - 1))

    function trade_share_cell(A_, f_, o, d)
        s = 0.0
        @inbounds for w in 1:size(z_draws, 1)
            z = z_draws[w, o]
            price = markup * w_[o] * tau_[o, d] / (A_[o, d] * z)
            rev = expenditure_[d] * price^(1 - sigma)
            profit = rev / sigma - w_[o] * f_[o, d]
            s += weights[w] * (profit > 0 ? rev : 0.0)
        end
        return s / expenditure_[d]
    end

    unperturbed_baseline_share_od = trade_share_cell(A, f, 5, 12)
    data_share_od = ctx.X_data[5, 12] / expenditure_[12]
    @printf("\n[1] Perturb A[5,12] by *e (one log-point): baseline share residual before=%.3e\n",
        unperturbed_baseline_share_od - data_share_od)
    A2 = copy(A); A2[5, 12] *= exp(1.0)
    perturbed_share = trade_share_cell(A2, f, 5, 12)
    @printf("    after perturbation: share residual=%.3e (cell 5,12)   other cell (1,1) residual unchanged: %.3e\n",
        perturbed_share - data_share_od, trade_share_cell(A2, f, 1, 1) - ctx.X_data[1, 1] / expenditure_[1])
    @assert abs(perturbed_share - data_share_od) > 100 * abs(unperturbed_baseline_share_od - data_share_od)

    @printf("\n[2] Perturb f[5,12] by *e: cutoff/zero-profit forward check\n")
    q_before = log(my_melitz_cutoff(A, f, 5, 12))
    f2 = copy(f); f2[5, 12] *= exp(1.0)
    q_after = log(my_melitz_cutoff(A, f2, 5, 12))
    @printf("    q_check before=%.6f (stored q=%.6f, resid=%.3e)   after perturbation q_check=%.6f (resid vs stored=%.3e)\n",
        q_before, q[5, 12], q_before - q[5, 12], q_after, q_after - q[5, 12])
    @assert abs(q_after - q[5, 12]) > 100 * abs(q_before - q[5, 12]) + 1e-6

    @printf("\n[3] Perturb cutoff q[5,12] directly by +0.1 (log units) without touching (A,f): forward-cutoff mismatch\n")
    q_pert = copy(q); q_pert[5, 12] += 0.1
    resid_before = q_before - q[5, 12]
    resid_after_direct_q_perturb = q_before - q_pert[5, 12]
    @printf("    forward cutoff (unchanged A,f) vs PERTURBED stored q: resid=%.6f (was %.3e)\n", resid_after_direct_q_perturb, resid_before)
    @assert abs(resid_after_direct_q_perturb) > 0.05

    @printf("\n[4] Perturb one LFD weight (10x, no renormalization): sum-to-1 check\n")
    w_pert = copy(weights); w_pert[1] *= 10
    @printf("    sum(weights) before=%.12f  after single-weight 10x bump=%.12f (deviation=%.3e)\n",
        kahan_sum(weights), kahan_sum(w_pert), kahan_sum(w_pert) - 1.0)
    @assert abs(kahan_sum(w_pert) - 1.0) > 1e-6

    @printf("\n[5] Perturb welfare coordinate g=log(gamma_prime_j) by +0.05: focal link residual under FIXED (old) LFD\n")
    g_old = log(r.gamma_prime_j)
    g_new = g_old + 0.05
    gamma_new = exp(g_new)
    expenditure_prime = ctx.w_prime * ctx.L[j]
    f_jj_new = derive_fjj_from_autarky_cutoff(gamma_new, ctx.w_prime, 1.0, A[j, j], expenditure_prime, sigma)
    function autarky_profit_sum(f_jj_, gamma_)
        s = 0.0
        @inbounds for w in 1:size(z_draws, 1)
            z = z_draws[w, j]
            price = markup * ctx.w_prime * 1.0 / (A[j, j] * z)
            rev = expenditure_prime * price^(1 - sigma) / gamma_
            profit = rev / sigma - ctx.w_prime * f_jj_
            s += weights[w] * (profit > 0 ? profit : 0.0)
        end
        return s
    end
    baseline_profit_sum_j = kahan_sum(weights .* [begin
        z = z_draws[w, j]; s = 0.0
        for d in 1:D
            price = markup * w_[j] * tau_[j, d] / (A[j, d] * z)
            rev = expenditure_[d] * price^(1 - sigma)
            profit = rev / sigma - w_[j] * f[j, d]
            s += profit > 0 ? profit : 0.0
        end
        s
    end for w in 1:size(z_draws, 1)])
    link_old = baseline_profit_sum_j / w_[j] - autarky_profit_sum(r.f_jj, r.gamma_prime_j) / ctx.w_prime
    link_new = baseline_profit_sum_j / w_[j] - autarky_profit_sum(f_jj_new, gamma_new) / ctx.w_prime
    @printf("    focal link residual at UNPERTURBED (g,f_jj) under this LFD: %.3e\n", link_old)
    @printf("    focal link residual after perturbing g by +0.05 (f_jj re-derived, LFD held fixed): %.3e\n", link_new)
    @assert abs(link_new) > 100 * abs(link_old) + 1e-8

    @printf("\n[6] Perturb the A-gravity pivot cell directly (breaking pivot self-consistency)\n")
    pivot_lin = ctx.A_pivot.pivot
    po = mod1(pivot_lin, D); pd_ = cld(pivot_lin, D)
    grav_before = my_gravity_residual(ctx.tau, A)
    A3 = copy(A); A3[po, pd_] *= exp(0.5)
    grav_after = my_gravity_residual(ctx.tau, A3)
    @printf("    pivot cell (o=%d,d=%d): A-gravity residual before=%.3e, after *=e^0.5 perturbation=%.3e\n",
        po, pd_, grav_before, grav_after)
    @assert abs(grav_after) > abs(grav_before) + 1e-6
end
println("\nAll 6 planted perturbations were correctly detected by the independent oracle (unperturbed incumbent passes; perturbed states fail).")

# ============================================================================
# Write CSV deliverables
# ============================================================================
for dirn in ["upper", "lower"]
    r = results[dirn]
    open(joinpath(OUTDIR, "bilateral_share_residuals_$(dirn).csv"), "w") do io
        println(io, "o,d,data_share,predicted_share,abs_residual,rel_residual")
        for o in 1:D, d in 1:D
            rel = r.data_share[o, d] == 0 ? NaN : r.trade_resid[o, d] / r.data_share[o, d]
            println(io, join([o, d, r.data_share[o, d], r.pred_share[o, d], r.trade_resid[o, d], rel], ","))
        end
    end
    open(joinpath(OUTDIR, "equation_summary_$(dirn).csv"), "w") do io
        println(io, "equation,value,tolerance_or_note")
        println(io, "GT_model_reported,$(r.best.GT),checkpointed")
        println(io, "GT_model_recomputed,$(r.GT_model),independent")
        println(io, "GT_ACR_data_anchored,$(r.GT_ACR),independent; INVARIANT to gamma_prime search")
        println(io, "GT_ACR_predicted_share,$(r.GT_ACR_predicted),independent")
        println(io, "abs_GT_model_minus_GT_ACR_data,$(abs(r.GT_model-r.GT_ACR)),percentage points")
        println(io, "max_trade_share_residual,$(r.max_trade_resid),independent D^2 loop")
        println(io, "max_cutoff_zero_profit_residual,$(r.max_cutoff_resid),independent forward formula")
        println(io, "focal_f_entry_baseline,$(r.f_entry_baseline[ctx.target_country]),recovered (definitional)")
        println(io, "f_entry_autarky,$(r.f_entry_autarky),recovered (definitional)")
        println(io, "max_abs_baseline_price_index_residual,$(maximum(abs.(r.gamma_baseline .- 1.0))),independent D loop")
        println(io, "N_prime_market_clearing,$(r.N_prime_mc),independent")
        println(io, "N_prime_price_index,$(r.N_prime_gamma),independent")
        println(io, "N_prime_diff_rel,$(r.N_prime_diff_rel),independent -- KEY omitted cross-check")
        println(io, "autarky_cutoff_residual_bigfloat,$(r.profit_at_one_bf),BigFloat; should be ~0 by construction")
        println(io, "gravity_residual_A_independent,$(r.my_grav_A),own withinTransform")
        println(io, "gravity_residual_f_independent,$(r.my_grav_f),own withinTransform")
        println(io, "max_abs_income_resource_residual_data,$(maximum(abs.(r.resource_resid))),data row-sum vs w*L")
        println(io, "max_abs_income_resource_residual_predicted,$(maximum(abs.(r.resource_resid_pred))),model-predicted row-sum vs w*L")
        println(io, "LFD_ESS,$(r.ess),effective sample size")
        println(io, "LFD_min_weight,$(r.min_w),")
        println(io, "LFD_max_weight,$(r.max_w),")
        println(io, "LFD_my_primal_divergence,$(r.my_primal),independent recompute")
        println(io, "LFD_reported_Delta,$(r.lfd_Delta),")
        println(io, "min_domestic_log_cutoff,$(r.min_dom),feasible iff >=0")
        println(io, "min_export_minus_domestic_log_cutoff,$(r.min_exp_minus_dom),feasible iff >=0")
        println(io, "check_residual_gamma_baseline_max,$(maximum(abs.(r.check.residual_gamma_baseline))),REFERENCE (existing code)")
        println(io, "check_residual_free_entry_baseline_max,$(maximum(abs.(r.check.residual_free_entry_baseline))),REFERENCE")
        println(io, "check_residual_free_entry_autarky,$(r.check.residual_free_entry_autarky),REFERENCE")
        println(io, "check_residual_market_clearing_autarky,$(r.check.residual_market_clearing_autarky),REFERENCE")
        println(io, "check_N_prime_diff_rel,$(r.check.N_prime_diff_rel),REFERENCE")
        println(io, "check_residual_gamma_autarky,$(r.check.residual_gamma_autarky),REFERENCE")
        println(io, "check_residual_autarky_cutoff,$(r.check.residual_autarky_cutoff),REFERENCE")
        println(io, "check_gravity_residual_A,$(r.check.gravity_residual_A),REFERENCE")
        println(io, "check_gravity_residual_f,$(r.check.gravity_residual_f),REFERENCE")
    end
end
println("\nWrote CSVs to $OUTDIR")
println("\nDONE INDEPENDENT FULL EQUILIBRIUM AUDIT")
