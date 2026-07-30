# Phases 3(bracketing)/4(cross-backend)/5(affected-row)/6(feasibility LP)/8(plus-vs-minus) of
# docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md.
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
ev = deserialize(joinpath(SCRATCH, "phase2_events.jls"))
theta0 = st.theta0; x0 = st.x0; calib = st.calib; focal = st.focal
D = st.D; nA = st.nA; nq = st.nq; b_q = st.b_q; stage = st.stage
minus_events = ev.minus_events; plus_events = ev.plus_events
println("Loaded state+events. minus[1:5] t=", [e.t for e in minus_events[1:5]])
println("plus[1:5] t=", [e.t for e in plus_events[1:5]])

function build_bundle_logcutoff()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end
obj = build_bundle_logcutoff()
ctx = obj.γ
sorted_ctx = ctx.sorted_tail_ctx
theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
ctx_logf = merge(ctx, (outer_parameterization=:logf,))
obj_logf = build_bundle_logcutoff()   # placeholder, replaced below with :logf build

function build_bundle_logf()
    obj2, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logf)
    return obj2
end
obj_logf = build_bundle_logf()
ctx_logf2 = obj_logf.γ

function theta_at(sign::Int, t::Real)
    th = copy(theta_plain0)
    th[1+nA+1:end] .+= sign .* t .* b_q
    return th   # in :logcutoff plain-A units (technology_coordinate=:logA => plain==powered here)
end

function classify_reduced(theta_lc::AbstractVector)
    session = MelitzInnerSession(obj, ctx, policy_cap)
    return solve_melitz_delta!(session, theta_lc, policy_cap)
end

function classify_production_equiv(theta_lc::AbstractVector)
    A, f, gpj, fjj = melitz_expand_theta(theta_lc, ctx)
    p = MelitzPrimitives(D, ctx.sigma, ctx.theta_star, ctx.target_country, ctx.tau, ctx.w, A, f, gpj)
    theta_logf = melitz_reduce_theta(p, ctx_logf2)
    session = MelitzInnerSession(obj_logf, ctx_logf2, policy_cap)
    return solve_melitz_delta!(session, theta_logf, policy_cap), theta_logf
end

function state_fingerprint(theta_lc::AbstractVector)
    A, f, gpj, fjj = melitz_expand_theta(theta_lc, ctx)
    return hash((round.(A; digits=12), round.(f; digits=12), round(gpj; digits=12)))
end

function classify_summary(r)
    if r isa FiniteSolved
        return (kind="FiniteSolved", Delta=r.Delta, nStatus=r.nStatus, cert=NaN, src=Symbol(""))
    elseif r isa AboveEvaluationCap
        return (kind="AboveEvaluationCap", Delta=NaN, nStatus=-1, cert=r.certified_lower_bound, src=r.source)
    elseif r isa InfiniteDeltaCertified
        return (kind="InfiniteDeltaCertified", Delta=NaN, nStatus=-1, cert=NaN, src=Symbol(r.kind))
    else
        return (kind="NumericalFailure", Delta=NaN, nStatus=r.nStatus, cert=NaN, src=Symbol(""))
    end
end

# ================================================================================================
# Phase 3/4: bracket first 5 distinct minus-side switches, cross-backend replay at each bracket.
# ================================================================================================
println("\n" * "="^100); println("Phase 3/4: MINUS-side bracketing + cross-backend replay"); println("="^100); flush(stdout)

bracket_rows = NamedTuple[]
distinct_ts = Float64[]
for e in minus_events
    if isempty(distinct_ts) || e.t > distinct_ts[end] * (1 + 1e-6)
        push!(distinct_ts, e.t)
    end
    length(distinct_ts) >= 6 && break
end
println("distinct switch t values (first 6): ", distinct_ts)

for (ki, tk) in enumerate(distinct_ts[1:5])
    gap_lo = ki == 1 ? tk : tk - distinct_ts[ki-1]
    gap_hi = ki < length(distinct_ts) ? distinct_ts[ki+1] - tk : tk
    eps = 0.15 * min(gap_lo, gap_hi)
    eps = max(eps, tk * 1e-8)
    for (label, tt) in (("minus", tk), ("plus_side_of_boundary", tk))
        nothing
    end
    for (label, tt) in (("below", tk - eps), ("at", tk), ("above", tk + eps))
        th = theta_at(-1, tt)
        r_reduced = classify_reduced(th)
        r_prod, theta_logf = classify_production_equiv(th)
        fp = state_fingerprint(th)
        sr = classify_summary(r_reduced); sp = classify_summary(r_prod)
        @printf("k=%d label=%-6s t=%.10e  reduced=%-22s Delta/cert=%.6g  production=%-22s Delta/cert=%.6g  match=%s  fp=%d\n",
                ki, label, tt, sr.kind, isnan(sr.Delta) ? sr.cert : sr.Delta,
                sp.kind, isnan(sp.Delta) ? sp.cert : sp.Delta, sr.kind == sp.kind, fp)
        push!(bracket_rows, (k=ki, label=label, t=tt, reduced_kind=sr.kind, reduced_Delta=sr.Delta,
            reduced_cert=sr.cert, reduced_src=String(sr.src), prod_kind=sp.kind, prod_Delta=sp.Delta,
            prod_cert=sp.cert, prod_src=String(sp.src), match=(sr.kind==sp.kind), fingerprint=fp,
            switch_o=minus_events[ki].o, switch_d=minus_events[ki].d, switch_dir=String(minus_events[ki].dir),
            switch_row_s=minus_events[ki].row_s))
    end
    flush(stdout)
end

open(joinpath(OUTDIR, "melitz_negswitch_phase34_minus_bracket_2026-07-30.csv"), "w") do io
    println(io, "k,label,t,reduced_kind,reduced_Delta,reduced_cert,reduced_src,prod_kind,prod_Delta,prod_cert,prod_src,match,fingerprint,switch_o,switch_d,switch_dir,switch_row_s")
    for r in bracket_rows
        println(io, join([r.k,r.label,r.t,r.reduced_kind,r.reduced_Delta,r.reduced_cert,r.reduced_src,
            r.prod_kind,r.prod_Delta,r.prod_cert,r.prod_src,r.match,r.fingerprint,r.switch_o,r.switch_d,
            r.switch_dir,r.switch_row_s], ","))
    end
end

# ================================================================================================
# Phase 5: independent affected-row verification for the FIRST switch (targeted, raw melitz_firm
# primitives, no shared low-level moment-operator helper).
# ================================================================================================
println("\n" * "="^100); println("Phase 5: independent affected-row check (switch 1)"); println("="^100); flush(stdout)
e1 = minus_events[1]
o1, d1, row1 = e1.o, e1.d, e1.row_s
for (label, tt) in (("below", distinct_ts[1] - 0.15*distinct_ts[1]), ("above", distinct_ts[1] + 0.15*(distinct_ts[2]-distinct_ts[1])))
    th = theta_at(-1, tt)
    A, f, gpj, fjj = melitz_expand_theta(th, ctx)
    z_row = ctx.sorted_tail_ctx.z_original[row1, o1]
    firm = melitz_firm(ctx.w[o1], ctx.tau[o1, d1], A[o1, d1], f[o1, d1], ctx.sigma, ctx.expenditure[d1], 1.0, z_row)
    raw_active = firm.realized_operating_profit > 0
    # production matrix-free operator's own bin[]: rebuild op at th, read bin[row1,o1] and
    # rank[d1,o1] -- active iff bin[row1,o1] >= rank[d1,o1] (destinations ranked ascending by
    # cutoff; bin counts how many of the D destinations are "cleared" for this draw).
    melitz_update_operator_at_theta!(obj.op, th, ctx)
    b = obj.op.bin[row1, o1]
    rk = obj.op.rank[d1, o1]
    op_active = b >= rk
    @printf("label=%-6s t=%.10e  raw melitz_firm active=%s  op.bin>=rank active=%s  match=%s  (bin=%d,rank=%d,profit=%.6g)\n",
            label, tt, raw_active, op_active, raw_active==op_active, b, rk, firm.realized_operating_profit)
end
melitz_update_operator_at_theta!(obj.op, theta_plain0, ctx)   # restore

# ================================================================================================
# Phase 6: exact finite-support feasibility (origin-block LP, compressed AND full-W reference)
# before/after the first switch, for the affected origin. Also the FULL screen (every origin).
# ================================================================================================
println("\n" * "="^100); println("Phase 6: origin-block feasibility LP, switch 1"); println("="^100); flush(stdout)
feas_rows = NamedTuple[]
for (label, tt) in (("below", distinct_ts[1] - 0.15*distinct_ts[1]), ("above", distinct_ts[1] + 0.15*(distinct_ts[2]-distinct_ts[1])))
    th = theta_at(-1, tt)
    feas_compressed = melitz_origin_block_lp(o1, th, ctx, obj)
    feas_reference = melitz_origin_block_lp_reference(o1, th, ctx, obj)
    mono = melitz_origin_block_monotonicity_check(o1, th, ctx, obj)
    screen_all = melitz_origin_block_screen(th, ctx, obj)
    @printf("label=%-6s t=%.10e  origin=%d  compressed_feasible=%s  reference_feasible=%s  monotonicity_ok=%s  full_screen=%s\n",
            label, tt, o1, feas_compressed, feas_reference, mono, screen_all === nothing ? "PASS(no cert)" : "INFEASIBLE($(screen_all.column))")
    push!(feas_rows, (label=label, t=tt, origin=o1, compressed_feasible=feas_compressed,
        reference_feasible=feas_reference, monotonicity_ok=mono,
        full_screen_infeasible_origin=(screen_all===nothing ? -1 : screen_all.column)))
end
open(joinpath(OUTDIR, "melitz_negswitch_phase6_feasibility_2026-07-30.csv"), "w") do io
    println(io, "label,t,origin,compressed_feasible,reference_feasible,monotonicity_ok,full_screen_infeasible_origin")
    for r in feas_rows
        println(io, join([r.label,r.t,r.origin,r.compressed_feasible,r.reference_feasible,r.monotonicity_ok,
            r.full_screen_infeasible_origin], ","))
    end
end

# ================================================================================================
# Phase 8: PLUS-side first switches -- same bracket + feasibility protocol, for comparison.
# ================================================================================================
println("\n" * "="^100); println("Phase 8: PLUS-side bracketing (comparison)"); println("="^100); flush(stdout)
distinct_ts_p = Float64[]
for e in plus_events
    if isempty(distinct_ts_p) || e.t > distinct_ts_p[end] * (1 + 1e-6)
        push!(distinct_ts_p, e.t)
    end
    length(distinct_ts_p) >= 6 && break
end
println("plus distinct switch t values (first 6): ", distinct_ts_p)
plus_bracket_rows = NamedTuple[]
for (ki, tk) in enumerate(distinct_ts_p[1:5])
    gap_lo = ki == 1 ? tk : tk - distinct_ts_p[ki-1]
    gap_hi = ki < length(distinct_ts_p) ? distinct_ts_p[ki+1] - tk : tk
    eps = max(0.15 * min(gap_lo, gap_hi), tk * 1e-8)
    for (label, tt) in (("below", tk - eps), ("above", tk + eps))
        th = theta_at(+1, tt)
        r_reduced = classify_reduced(th)
        sr = classify_summary(r_reduced)
        @printf("k=%d label=%-6s t=%.10e  reduced=%-22s Delta/cert=%.6g\n",
                ki, label, tt, sr.kind, isnan(sr.Delta) ? sr.cert : sr.Delta)
        push!(plus_bracket_rows, (k=ki, label=label, t=tt, kind=sr.kind, Delta=sr.Delta, cert=sr.cert,
            switch_o=plus_events[ki].o, switch_d=plus_events[ki].d, switch_dir=String(plus_events[ki].dir)))
    end
    flush(stdout)
end
open(joinpath(OUTDIR, "melitz_negswitch_phase8_plus_bracket_2026-07-30.csv"), "w") do io
    println(io, "k,label,t,kind,Delta,cert,switch_o,switch_d,switch_dir")
    for r in plus_bracket_rows
        println(io, join([r.k,r.label,r.t,r.kind,r.Delta,r.cert,r.switch_o,r.switch_d,r.switch_dir], ","))
    end
end

# LFD weight comparison for the switched row on both sides (base anchor dual x0 as reference,
# via the one-sided-safe weak-duality lower bound at x0 -- cheap, no new solve).
println("\n" * "="^100); println("LFD/weight context for switched rows (minus[1] vs plus[1])"); println("="^100)
println("minus[1]: o=$(minus_events[1].o) d=$(minus_events[1].d) row_s=$(minus_events[1].row_s) dir=$(minus_events[1].dir) z=$(minus_events[1].z)")
println("plus[1]: o=$(plus_events[1].o) d=$(plus_events[1].d) row_s=$(plus_events[1].row_s) dir=$(plus_events[1].dir) z=$(plus_events[1].z)")

serialize(joinpath(SCRATCH, "phase4_8_results.jls"), (bracket_rows=bracket_rows, feas_rows=feas_rows,
    plus_bracket_rows=plus_bracket_rows, distinct_ts=distinct_ts, distinct_ts_p=distinct_ts_p))
println("\nDONE PHASE 3/4/5/6/8")
