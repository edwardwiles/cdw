# Phase 2/3: exact negative- (and positive-) side participation-switch thresholds along
# q(t) = q0 -+ t*b_q, computed ANALYTICALLY (not bisected) by exploiting that, with (g,A_free)
# held fixed, the full D x D log-cutoff matrix q_full(t) is an EXACT LINEAR function of t
# (log_cutoff_param.jl: pivot_expand is linear, A/g fixed => g0_q fixed => q-pivot offset
# fixed => q_free_full(t) = pivot_expand(q_anchor_free +- t*b_q, pivot) is affine in t for
# every cell, including the analytically-reconstructed gravity-pivot cell).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

st = deserialize(joinpath(SCRATCH, "phase0_state.jls"))
theta0 = st.theta0; calib = st.calib; focal = st.focal
D = st.D; nA = st.nA; nq = st.nq; b_q = st.b_q; stage = st.stage
println("Loaded phase0 state. D=", D, "  |b_q|=", norm(b_q))

function build_bundle()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end
obj = build_bundle()
ctx = obj.γ
sorted_ctx = ctx.sorted_tail_ctx
j = ctx.target_country

theta_plain0 = melitz_unpower_theta_free(theta0, ctx)

function q_full_at(sign::Int, t::Real)
    th = copy(theta_plain0)
    th[1+nA+1:end] .+= sign .* t .* b_q
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx)
    return q
end

q0 = q_full_at(+1, 0.0)   # baseline (sign irrelevant at t=0)

# ---- verify exact linearity of q_full(t) along both signs (not merely assumed) ----
println("\n" * "="^100); println("Linearity verification"); println("="^100); flush(stdout)
for sign in (+1, -1)
    qa = q_full_at(sign, 0.2)
    qb = q_full_at(sign, 0.6)
    qc = q_full_at(sign, 1.0)
    slope_from_02_06 = (qb .- qa) ./ 0.4
    slope_from_06_10 = (qc .- qb) ./ 0.4
    maxdiff = maximum(abs.(slope_from_02_06 .- slope_from_06_10))
    println("sign=$sign: max|slope(0.2->0.6) - slope(0.6->1.0)| = ", maxdiff,
            "  (must be ~0 for exact affine reconstruction)")
    @assert maxdiff < 1e-9 "q_full(t) is NOT exactly affine in t for sign=$sign -- switch-threshold analytic formula invalid"
end

# Exact slope matrices (per sign; note Slope(-1) should equal -Slope(+1) since both are the
# SAME linear map applied with the opposite argument sign -- verified below too).
Slope_plus = q_full_at(+1, 1.0) .- q0
Slope_minus_direct = q_full_at(-1, 1.0) .- q0
println("max|Slope_minus_direct - (-Slope_plus)| = ", maximum(abs.(Slope_minus_direct .- (.-Slope_plus))))
@assert maximum(abs.(Slope_minus_direct .- (.-Slope_plus))) < 1e-9

# ================================================================================================
# Exact switch-threshold enumeration for one sign (+1 = "plus" direction q0+t*b_q,
# -1 = "minus" direction q0-t*b_q). Returns a sorted Vector of NamedTuples (t, o, d, k, row_s,
# on_or_off, z_value, q0_od, slope_od).
# ================================================================================================
function enumerate_switches(sign::Int, n_switches::Int; t_max::Real=2.0)
    Slope = sign == 1 ? Slope_plus : Slope_minus_direct
    D_ = size(q0, 1)
    # current active-tail-start per (o,d) cell (only cells in ctx.f_free_lin participate; the
    # domestic (j,j) cell is excluded -- q[j,j] is not part of the free-q/pivot domain at all).
    kcur = Dict{Tuple{Int,Int},Int}()
    slope = Dict{Tuple{Int,Int},Float64}()
    q0v = Dict{Tuple{Int,Int},Float64}()
    for lin in ctx.f_free_lin
        o, d = lin2od(lin, D_)
        sorted_z_o = @view sorted_ctx.sorted_z[:, o]
        cutoff0 = exp(q0[o, d])
        kcur[(o, d)] = melitz_active_tail_start(sorted_z_o, cutoff0)
        slope[(o, d)] = Slope[o, d]
        q0v[(o, d)] = q0[o, d]
    end

    events = NamedTuple[]
    for step in 1:n_switches
        best_t = Inf
        best_cell = nothing
        best_k = 0
        best_dir = :none
        for (cell, sl) in slope
            abs(sl) < 1e-300 && continue
            o, d = cell
            sorted_z_o = @view sorted_ctx.sorted_z[:, o]
            W_ = length(sorted_z_o)
            k0 = kcur[cell]
            # q_od(t) = q0v[cell] + t*sl (exact, linear). Solving q_od(t)=log(z_target) gives
            # t = (log(z_target) - q0v[cell]) / sl for EITHER sign of sl -- only which k is the
            # relevant next target differs by sign.
            if sl > 0
                # cutoff = exp(q_od(t)) increasing with t -> next switch: the smallest
                # currently-active draw (sorted_z_o[k0]) becomes inactive (if k0<=W).
                k0 > W_ && continue
                t_cand = (log(sorted_z_o[k0]) - q0v[cell]) / sl
                cand_k = k0; cand_dir = :off
            else
                # cutoff decreasing with t -> next switch: the largest currently-inactive draw
                # (sorted_z_o[k0-1]) becomes active (if k0-1>=1).
                k0 - 1 < 1 && continue
                t_cand = (log(sorted_z_o[k0-1]) - q0v[cell]) / sl
                cand_k = k0 - 1; cand_dir = :on
            end
            if isfinite(t_cand) && t_cand > 1e-13 && t_cand < best_t && t_cand <= t_max
                best_t = t_cand; best_cell = cell; best_k = cand_k; best_dir = cand_dir
            end
        end
        best_cell === nothing && break
        o, d = best_cell
        row_s = sorted_ctx.permutation[best_k, o]
        z_val = sorted_ctx.sorted_z[best_k, o]
        push!(events, (t=best_t, o=o, d=d, k=best_k, row_s=row_s, dir=best_dir, z=z_val,
                        q0_od=q0v[best_cell], slope_od=slope[best_cell], step=step))
        kcur[best_cell] = best_dir == :off ? best_k + 1 : best_k
    end
    return events
end

println("\n" * "="^100); println("MINUS-direction first switches (q0 - t*b_q)"); println("="^100); flush(stdout)
minus_events = enumerate_switches(-1, 15)
for e in minus_events
    @printf("step=%2d  t=%.10e  (o=%d,d=%d)  k=%d  row_s=%d  dir=%s  z=%.6f  q0_od=%.6f  slope_od=%.6e\n",
            e.step, e.t, e.o, e.d, e.k, e.row_s, e.dir, e.z, e.q0_od, e.slope_od)
end

println("\n" * "="^100); println("PLUS-direction first switches (q0 + t*b_q)"); println("="^100); flush(stdout)
plus_events = enumerate_switches(+1, 15)
for e in plus_events
    @printf("step=%2d  t=%.10e  (o=%d,d=%d)  k=%d  row_s=%d  dir=%s  z=%.6f  q0_od=%.6f  slope_od=%.6e\n",
            e.step, e.t, e.o, e.d, e.k, e.row_s, e.dir, e.z, e.q0_od, e.slope_od)
end

# Cross-check against the EXISTING melitz_q_direction_two_sided_crossings infrastructure:
# total crossing COUNT at t=minus_events[k].t + eps should equal k (assuming no ties), and at
# t=minus_events[k].t - eps should equal k-1.
println("\n" * "="^100); println("Cross-check vs melitz_q_direction_two_sided_crossings"); println("="^100); flush(stdout)
for k in 1:min(10, length(minus_events))
    t_here = minus_events[k].t
    eps = max(1e-12, t_here * 1e-9)
    _, tm_below = melitz_q_direction_two_sided_crossings(theta0, b_q, t_here - eps, ctx, sorted_ctx)
    _, tm_above = melitz_q_direction_two_sided_crossings(theta0, b_q, t_here + eps, ctx, sorted_ctx)
    @printf("k=%2d  t=%.10e  crossings(t-eps)=%d  crossings(t+eps)=%d  (expect %d,%d)\n",
            k, t_here, tm_below, tm_above, k-1, k)
end

# CSV outputs.
function write_events_csv(path, events)
    open(path, "w") do io
        println(io, "step,t,o,d,k,row_s,dir,z,q0_od,slope_od")
        for e in events
            println(io, "$(e.step),$(e.t),$(e.o),$(e.d),$(e.k),$(e.row_s),$(e.dir),$(e.z),$(e.q0_od),$(e.slope_od)")
        end
    end
end
write_events_csv(joinpath(OUTDIR, "melitz_negswitch_phase2_minus_switches_2026-07-30.csv"), minus_events)
write_events_csv(joinpath(OUTDIR, "melitz_negswitch_phase2_plus_switches_2026-07-30.csv"), plus_events)

serialize(joinpath(SCRATCH, "phase2_events.jls"), (minus_events=minus_events, plus_events=plus_events,
    Slope_plus=Slope_plus, q0=q0))
println("\nDONE PHASE 2/3-setup")
