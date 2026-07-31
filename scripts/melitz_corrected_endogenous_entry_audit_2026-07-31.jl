# ============================================================================
# CORRECTED full-equilibrium audit: endogenous autarky entrant mass N'_j (NOT required
# to equal N_j), the focal free-entry LINK proven algebraically equivalent to profiling
# out a shared f_Ej, and GT computed from factual/autarky price indices (ACR treated as
# descriptive-only, non-binding away from the exact Pareto calibration point).
#
# This supersedes the interpretation (but not the raw numbers, which are re-verified
# independently here) of scripts/melitz_independent_full_equilibrium_audit_2026-07-31.jl's
# N'-cross-check and ACR sections.
# ============================================================================
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization, Dates
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

OUTDIR = joinpath(REPO2, "docs", "key_results", "melitz_corrected_endogenous_entry_full_equilibrium_audit_2026-07-31")
mkpath(OUTDIR)

function kahan_sum(xs)
    s = 0.0; c = 0.0
    @inbounds for x in xs
        y = x - c; t = s + y; c = (t - s) - y; s = t
    end
    return s
end

function bf_firm(w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, z)
    T = BigFloat
    w_o, tau_od, A_od, f_od, sigma, expenditure_d, price_power_d, z =
        T(w_o), T(tau_od), T(A_od), T(f_od), T(sigma), T(expenditure_d), T(price_power_d), T(z)
    markup = sigma / (sigma - 1)
    price = markup * w_o * tau_od / (A_od * z)
    rev = expenditure_d * price^(1 - sigma) / price_power_d
    profit = rev / sigma - w_o * f_od
    return profit, rev, price
end

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
D = ctx.D; j = ctx.target_country; W = size(obj_d20.U, 1)
@printf("\nD=%d  target_country=%d (%s)  sigma=%.6f  theta_star=%.8f  W=%d\n",
    D, j, countries_d20[j], ctx.sigma, ctx.theta_star, W)
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

function audit_incumbent(label, direction)
    println("\n", "#"^100); println("# INCUMBENT: $label ($direction)"); println("#"^100); flush(stdout)
    ckpt = joinpath(OVERDIR, "phase5", "checkpoints", "$(label).jls")
    state = deserialize(ckpt)
    best = state.points[state.most_extreme_idx]
    @assert best.classification == :FiniteSolved && best.lfd_ok

    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    r_fresh = solve_melitz_delta!(session, best.theta_free, policy_cap; warm_start_source=:neutral)
    @assert r_fresh isa FiniteSolved
    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    lfd = melitz_recover_lfd(obj_d20, best.theta_free)
    @assert lfd.lfd_ok
    weights = lfd.weights
    z_draws = obj_d20.U

    A, f, gamma_prime_j, f_jj, q = expand_free_theta_logcutoff(best.theta_free, ctx)
    w_ = ctx.w; tau_ = ctx.tau; expenditure_ = ctx.expenditure; sigma = ctx.sigma; theta_star = ctx.theta_star
    w_prime = ctx.w_prime; expenditure_prime = ctx.w_prime * ctx.L[j]
    @assert isapprox(expenditure_prime, w_prime * ctx.L[j]; atol=1e-12) "E'_j = w'_j*L_j convention check"
    markup = sigma / (sigma - 1)

    firm(o, d, z, price_power_d=1.0, w_o=w_[o], tau_od=tau_[o, d], A_od=A[o, d], f_od=f[o, d], exp_d=expenditure_[d]) = begin
        price = markup * w_o * tau_od / (A_od * z)
        rev = exp_d * price^(1 - sigma) / price_power_d
        profit = rev / sigma - w_o * f_od
        (profit, rev, price)
    end

    # ------------------------------------------------------------------------
    # PART I / F1: bilateral trade flows (independent, all D^2 cells)
    # ------------------------------------------------------------------------
    pred_share = zeros(D, D); data_share = zeros(D, D)
    for o in 1:D, d in 1:D
        revs = Vector{Float64}(undef, W)
        @inbounds for w in 1:W
            profit, rev, _ = firm(o, d, z_draws[w, o])
            revs[w] = profit > 0 ? rev : 0.0
        end
        pred_share[o, d] = kahan_sum(weights .* revs) / expenditure_[d]
        data_share[o, d] = ctx.X_data[o, d] / expenditure_[d]
    end
    max_F1 = maximum(abs.(pred_share .- data_share))

    # ------------------------------------------------------------------------
    # F2: factual price-index gamma_d for ALL D destinations (independent) + F5 support
    # ------------------------------------------------------------------------
    gamma_baseline = zeros(D)
    for d in 1:D
        s = 0.0
        @inbounds for o in 1:D, w in 1:W
            profit, _, price = firm(o, d, z_draws[w, o])
            if profit > 0
                s += weights[w] * price^(1 - sigma)
            end
        end
        gamma_baseline[d] = s
    end
    max_F2 = maximum(abs.(gamma_baseline .- 1.0))
    gamma_j_factual = gamma_baseline[j]  # needed for the general GT formula

    q_check = zeros(D, D)
    for o in 1:D, d in 1:D
        C_od = expenditure_[d] * (markup * w_[o] * tau_[o, d] / A[o, d])^(1 - sigma)
        q_check[o, d] = log((sigma * w_[o] * f[o, d] / C_od)^(1 / (sigma - 1)))
    end
    max_F5 = maximum(abs.(q_check .- q))

    # ------------------------------------------------------------------------
    # F3: factual free entry, ALL D origins -- f_Eo table
    # ------------------------------------------------------------------------
    Pi_F = zeros(D); f_Eo = zeros(D); fe_resid = zeros(D)
    for o in 1:D
        profit_sums = Vector{Float64}(undef, W)
        @inbounds for w in 1:W
            z = z_draws[w, o]
            s = 0.0
            for d in 1:D
                profit, _, _ = firm(o, d, z)
                s += profit > 0 ? profit : 0.0
            end
            profit_sums[w] = s
        end
        Pi_F[o] = kahan_sum(weights .* profit_sums)
        f_Eo[o] = Pi_F[o] / w_[o]
        fe_resid[o] = Pi_F[o] - w_[o] * f_Eo[o]   # identically 0 by construction; reported explicitly
    end
    all_nonneg = all(f_Eo .>= 0.0)

    # ------------------------------------------------------------------------
    # F4: factual resource/labor equation. This repo's maintained implementation uses a
    # PURE ICEBERG convention (docs/melitz_delta_star.md Sec.13.6: population_X's market
    # clearing is the plain w_o*L_o = sum_d X_od, "NO 1/tau_od tariff-revenue term
    # anywhere" -- confirmed by direct code reading of melitz_solve_wages_ge/population_X).
    # The alternative "(sigma-1)/(sigma*(1+t_od)) + 1/sigma" weighted formula in the audit
    # prompt corresponds to an AD-VALOREM-tariff-with-revenue-rebate convention this repo
    # does NOT implement -- using the model's actual (iceberg) formula per the "use the
    # exact production definitions" instruction.
    # ------------------------------------------------------------------------
    income_data = vec(sum(ctx.X_data, dims=2))
    resid_F4_data = income_data .- (w_ .* ctx.L)
    income_pred = vec(sum(pred_share .* expenditure_', dims=2))
    resid_F4_pred = income_pred .- (w_ .* ctx.L)
    max_F4 = maximum(abs.(resid_F4_pred))

    # ------------------------------------------------------------------------
    # PART II: focal autarky equilibrium
    # ------------------------------------------------------------------------
    # C1: zero-profit at z=1 (BigFloat)
    profit1_bf, rev1_bf, price1_bf = bf_firm(w_prime, 1.0, A[j, j], f_jj, sigma, expenditure_prime, gamma_prime_j, 1.0)

    # C2: same f_Ej in factual and autarky
    f_Ej_F = f_Eo[j]
    autarky_profit_sums = Vector{Float64}(undef, W)
    autarky_rev = Vector{Float64}(undef, W)
    autarky_price_pow = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        z = z_draws[w, j]
        profit, rev, price = firm(j, j, z, gamma_prime_j, w_prime, 1.0, A[j, j], f_jj, expenditure_prime)
        autarky_profit_sums[w] = profit > 0 ? profit : 0.0
        autarky_rev[w] = profit > 0 ? rev : 0.0          # active=1 a.s. given zhat'_jj=1, Pareto(1,.) support
        autarky_price_pow[w] = profit > 0 ? price^(1 - sigma) : 0.0
    end
    Pi_A = kahan_sum(weights .* autarky_profit_sums)
    f_Ej_A = Pi_A / w_prime
    fe_link_resid = Pi_F[j] / w_[j] - Pi_A / w_prime   # exactly f_Ej_F - f_Ej_A by construction

    # C3: autarky price-index equation -> N'_j via price index
    Q_prime_j = kahan_sum(weights .* autarky_price_pow)
    N_prime_gamma = gamma_prime_j / Q_prime_j
    resid_C3 = gamma_prime_j - N_prime_gamma * Q_prime_j   # identically 0 by construction; reported

    # C4: autarky resource equation, DIRECT independent formula (not via the entry-cost
    # detour): N'_{j,resource} = w'_j*L_j / E_LFD[x'_jj(z)*1{z>=1}]
    E_rev_prime = kahan_sum(weights .* autarky_rev)
    N_prime_resource = expenditure_prime / E_rev_prime
    N_diff_rel = abs(N_prime_resource - N_prime_gamma) / abs(N_prime_gamma)
    resid_C4_shared_N = N_prime_gamma * E_rev_prime - expenditure_prime   # using the price-index N'_j

    # C6: GT via the general factual/autarky price-index formula
    GT_general = 100 * (1 - (w_prime / w_[j]) * (gamma_j_factual / gamma_prime_j)^(1 / (1 - sigma)))
    wage_ratio = w_prime / w_[j]
    GT_simplified = 100 * (1 - wage_ratio * gamma_prime_j^(1 / (sigma - 1)))  # valid iff gamma_j_factual==1

    # ACR, reported descriptively only (non-binding away from Pareto)
    lambda_jj_data = ctx.X_data[j, j] / expenditure_[j]
    GT_ACR = 100 * (1 - lambda_jj_data^(1 / theta_star))

    @printf("Checkpointed GT=%.10f%%  Delta*=%.10f\n", best.GT, best.Delta)
    @printf("F1 max|trade-share resid|=%.3e   F2 max|gamma_d-1|=%.3e   F5 max|cutoff resid|=%.3e\n", max_F1, max_F2, max_F5)
    @printf("F4 max|income resid| (data)=%.3e  (model-predicted)=%.3e\n", maximum(abs.(resid_F4_data)), max_F4)
    @printf("All factual f_Eo >= 0: %s   range=[%.6g, %.6g]\n", all_nonneg, minimum(f_Eo), maximum(f_Eo))
    @printf("C1 (BigFloat) profit'(z=1)=%.6e\n", Float64(profit1_bf))
    @printf("C2: f_Ej_F=%.6g  f_Ej_A=%.6g  diff=%.3e  FE-link residual=%.3e\n", f_Ej_F, f_Ej_A, f_Ej_A - f_Ej_F, fe_link_resid)
    @printf("C3: gamma'_j=%.6g  Q'_j=%.6g  N'_(gamma)=%.6g  price-index residual=%.3e\n", gamma_prime_j, Q_prime_j, N_prime_gamma, resid_C3)
    @printf("C4: N'_(resource,direct)=%.6g  N'_(gamma)=%.6g  rel diff=%.3e  resource residual (shared N')=%.3e\n",
        N_prime_resource, N_prime_gamma, N_diff_rel, resid_C4_shared_N)
    @printf("gamma_j_factual (baseline price index at j, should be ~1)=%.10f\n", gamma_j_factual)
    @printf("GT_general (factual/autarky price-index formula)=%.6f%%   GT_simplified(assumes gamma_j=1)=%.6f%%   checkpoint=%.6f%%\n",
        GT_general, GT_simplified, best.GT)
    @printf("GT_ACR (descriptive only, non-binding away from Pareto)=%.6f%%   |GT_general-GT_ACR|=%.6f pp\n",
        GT_ACR, abs(GT_general - GT_ACR))

    return (label=label, direction=direction, best=best, w_j=w_[j], countries=countries_d20,
        Pi_F=Pi_F, f_Eo=f_Eo, fe_resid=fe_resid, all_nonneg=all_nonneg,
        max_F1=max_F1, max_F2=max_F2, max_F5=max_F5,
        resid_F4_data=resid_F4_data, resid_F4_pred=resid_F4_pred,
        profit1_bf=Float64(profit1_bf), f_Ej_F=f_Ej_F, f_Ej_A=f_Ej_A, fe_link_resid=fe_link_resid,
        gamma_prime_j=gamma_prime_j, Q_prime_j=Q_prime_j, N_prime_gamma=N_prime_gamma, resid_C3=resid_C3,
        N_prime_resource=N_prime_resource, N_diff_rel=N_diff_rel, resid_C4_shared_N=resid_C4_shared_N,
        gamma_j_factual=gamma_j_factual, GT_general=GT_general, GT_simplified=GT_simplified,
        GT_ACR=GT_ACR)
end

results = Dict{String,Any}()
results["upper"] = audit_incumbent("phase6_upper_m1", "upper")
results["lower"] = audit_incumbent("phase6_lower_m-1", "lower")

for dirn in ["upper", "lower"]
    r = results[dirn]
    open(joinpath(OUTDIR, "factual_entry_table_$(dirn).csv"), "w") do io
        println(io, "origin_idx,country,w_o,Pi_o_F,f_Eo,FE_residual,nonneg_ok")
        for o in 1:D
            println(io, join([o, r.countries[o], ctx.w[o], r.Pi_F[o], r.f_Eo[o], r.fe_resid[o], r.f_Eo[o] >= 0], ","))
        end
    end
    open(joinpath(OUTDIR, "focal_autarky_table_$(dirn).csv"), "w") do io
        println(io, "field,value")
        println(io, "f_Ej_factual,$(r.f_Ej_F)")
        println(io, "f_Ej_autarky,$(r.f_Ej_A)")
        println(io, "f_Ej_diff,$(r.f_Ej_A - r.f_Ej_F)")
        println(io, "FE_link_residual,$(r.fe_link_resid)")
        println(io, "gamma_prime_j,$(r.gamma_prime_j)")
        println(io, "Q_prime_j,$(r.Q_prime_j)")
        println(io, "N_prime_gamma,$(r.N_prime_gamma)")
        println(io, "N_prime_resource_direct,$(r.N_prime_resource)")
        println(io, "N_prime_diff_rel,$(r.N_diff_rel)")
        println(io, "price_index_residual,$(r.resid_C3)")
        println(io, "resource_equation_residual,$(r.resid_C4_shared_N)")
        println(io, "zero_profit_residual_bigfloat,$(r.profit1_bf)")
        println(io, "gamma_j_factual_check,$(r.gamma_j_factual)")
        println(io, "GT_general_reconstructed,$(r.GT_general)")
        println(io, "GT_simplified_reconstructed,$(r.GT_simplified)")
        println(io, "GT_checkpointed,$(r.best.GT)")
        println(io, "GT_ACR_descriptive_nonbinding,$(r.GT_ACR)")
        println(io, "abs_diff_GT_general_vs_ACR,$(abs(r.GT_general - r.GT_ACR))")
    end
end
println("\nWrote CSVs to $OUTDIR")
println("\nDONE CORRECTED ENDOGENOUS-ENTRY AUDIT")
