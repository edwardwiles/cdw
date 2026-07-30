# Follow-up (user question): exactly how can a raw callback evaluation be NaN, given the
# functor is "just multiplying dual multipliers by positive CES kernels and zeros"?
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

st = deserialize(joinpath(SCRATCH, "phase0_state.jls"))
theta0 = st.theta0; calib = st.calib
D = st.D; nA = st.nA; b_q = st.b_q

function build_bundle()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end
obj = build_bundle()
ctx = obj.γ
theta_m = copy(theta0); theta_m[1+nA+1:end] .-= 0.5 .* b_q

melitz_objective_trace_reset!()
MELITZ_OBJECTIVE_TRACE_ENABLED[] = true
MELITZ_OBJECTIVE_TRACE_X_ENABLED[] = true
obj.use_cached_x = false; obj.x .= NaN
lfd_m = melitz_recover_lfd(obj, theta_m)
MELITZ_OBJECTIVE_TRACE_ENABLED[] = false
MELITZ_OBJECTIVE_TRACE_X_ENABLED[] = false

trace = copy(MELITZ_OBJECTIVE_TRACE)
xs = copy(MELITZ_OBJECTIVE_TRACE_X)
println("n_calls=", length(trace))
for i in eachindex(trace)
    @printf("call %2d: f=%s  crossed=%s  |x|_max=%.6e  zeta=%.6e\n", i, string(trace[i][1]), trace[i][2],
        maximum(abs.(xs[i])), xs[i][1])
end

# Pick the FIRST NaN call.
nan_idx = findfirst(t -> isnan(t[1]), trace)
println("\nFirst NaN call = ", nan_idx)
x_nan = xs[nan_idx]
zeta = x_nan[1]
mu = @view x_nan[2:end]
@printf("zeta = %.6e\n", zeta)
println("max|mu| = ", maximum(abs.(mu)), "   argmax = ", argmax(abs.(mu)))
println("mu[argmax] = ", mu[argmax(abs.(mu))])

# Reproduce mul_G! by hand, origin by origin, tracking overflow onset for EVERY row, and
# report which rows end up NaN/Inf and via which origin's own contribution.
op = obj.op
Dd = op.D; W = op.W
trade_index = op.layout.trade_index
coef = op.coef; lambda = op.lambda; order = op.order; bin = op.bin
z_power = op.sorted_ctx.z_power_original

u = fill(-Float64(zeta), W)
n_nan_after = zeros(Int, Dd)   # cumulative NaN row count after processing origin o
n_inf_after = zeros(Int, Dd)
first_nan_origin = Dict{Int,Int}()   # row -> origin index at which it FIRST became NaN
for o in 1:Dd
    const_o = 0.0
    cum = zeros(Dd+1)
    for m in 1:Dd
        d = order[m, o]
        mu_od = mu[trade_index[o, d]]
        cum[m+1] = cum[m] + mu_od*coef[o, d]
        const_o += mu_od*lambda[o, d]
    end
    @printf("origin %2d: const_o=%.6e   cum (range) = [%.6e, %.6e]   max|mu_od*coef|=%.6e\n",
        o, const_o, minimum(cum), maximum(cum),
        maximum(abs.(mu[trade_index[o,d]]*coef[o,d] for d in 1:Dd)))
    for s in 1:W
        was_nan = isnan(u[s])
        u[s] += const_o
        b = bin[s, o]
        if b != 0
            u[s] -= z_power[s, o]*cum[b+1]
        end
        if isnan(u[s]) && !was_nan && !haskey(first_nan_origin, s)
            first_nan_origin[s] = o
        end
    end
    n_nan_after[o] = count(isnan, u)
    n_inf_after[o] = count(x -> isinf(x), u)
    @printf("   after origin %2d: n_NaN_rows=%d  n_Inf_rows=%d  max|u| (finite)=%.6e\n",
        o, n_nan_after[o], n_inf_after[o],
        (any(isfinite,u) ? maximum(abs.(filter(isfinite,u))) : NaN))
end

println("\nTotal NaN rows at end (before focal link term) = ", count(isnan, u))
println("Total Inf rows at end = ", count(isinf, u))
if !isempty(first_nan_origin)
    row_example = first(keys(first_nan_origin))
    println("Example row s=", row_example, " first went NaN while processing origin o=", first_nan_origin[row_example])
end

# For ONE concrete example row, print the exact partial sums origin by origin (only up to
# +-5 origins around the culprit) to show the Inf-then-cancel-to-NaN mechanism explicitly.
if !isempty(first_nan_origin)
    s_ex = first(keys(first_nan_origin))
    println("\n=== Detailed per-origin trace for row s=", s_ex, " (z=", op.sorted_ctx.z_original[s_ex,1:3], "...) ===")
    u2 = -Float64(zeta)
    for o in 1:Dd
        const_o = 0.0
        cum = zeros(Dd+1)
        for m in 1:Dd
            d = order[m, o]
            mu_od = mu[trade_index[o, d]]
            cum[m+1] = cum[m] + mu_od*coef[o, d]
            const_o += mu_od*lambda[o, d]
        end
        b = bin[s_ex, o]
        term2 = b != 0 ? z_power[s_ex, o]*cum[b+1] : 0.0
        u_before = u2
        u2 += const_o
        u2 -= term2
        @printf("  o=%2d  const_o=%12.5e  bin=%3d  z_power*cum=%12.5e  u: %12.5e -> %12.5e\n",
            o, const_o, b, term2, u_before, u2)
        isnan(u2) && println("    *** became NaN here ***")
    end
end
println("\nDONE follow-up 1")
