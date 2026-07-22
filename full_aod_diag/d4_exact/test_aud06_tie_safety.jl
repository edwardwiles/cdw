# ============================================================================
# AUD-06 regression test: the fused winning-range screen (fast_range_screen.jl's
# screen_hard_winners_ranged) must never issue a false :winning_range exact
# infeasibility certificate against a genuinely feasible origin that only ties
# (does not strictly win) the minimum price in every draw where it wins.
#
# Reuses the EXACT tie-construction fixture test_infeasibility_screen.jl already
# validated `screen_hard_winners` (the dense/original screen) against -- this
# file adds the equivalent check for `screen_hard_winners_ranged` (the newer,
# fused screen that was NOT tie-safe before this fix; see
# docs/fullA_independent_audit_remediation.md AUD-06).
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_aud06_tie_safety.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

ctx = d4_exact_setup(find_smallest = true)
D = ctx.D
pe = build_pivot_elimination(ctx)
zfree0 = pivot_reduce(zeros(D, D), pe)
gp0 = ctx.θ0_up[3+D]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Pmat = target_shares(ctx)
rsc = build_ranged_screen_context(ctx)
rsc.envelope !== nothing || error("test_aud06_tie_safety: envelope screen not supported for this ctx -- cannot exercise screen_hard_winners_ranged")

println("== AUD-06: deliberate exact price tie must not produce a false :winning_range certificate ==")

# ---- construct the SAME bit-identical tie test_infeasibility_screen.jl uses:
# force Aod_theta[1,1] == Aod_theta[2,1] exactly (origins 1 and 2 tie at destination 1) -- the
# SAME construction test_infeasibility_screen.jl already validated screen_hard_winners's
# win-count tie-safety against. NOTE: `compute_a_od`'s own `a[1,1]==a[2,1]` bit-identity is
# reported for information only (not asserted) -- it is not necessarily the same quantity
# screen_hard_winners_ranged's price formula (constCons[o,d]/UPow[s,o], built from
# wHat/tau/AodPow, a DIFFERENT derived representation) compares, so it can legitimately diverge
# even when a genuine downstream price tie exists; the win_counts/certificate agreement checks
# below are the real test of THIS function's tie-safety, not `a` bit-identity. ----
w_tie = copy(vcat(gp0, zfree0))
xf_tie = x_free_from_w(w_tie)
θ_full_tie = CS.reconstruct_full(xf_tie, ctx.m)
Aoff = ctx.Aod_offset
idx(o, d) = Aoff + o + (d - 1) * D
θ_full_tie[idx(2, 1)] = θ_full_tie[idx(1, 1)]
a_tie = compute_a_od(θ_full_tie, ctx)
println("  INFO: a[1,1]=", a_tie[1, 1], "  a[2,1]=", a_tie[2, 1],
        "  bit-identical=", a_tie[1, 1] == a_tie[2, 1], " (informational, see note above)")

# ---- dense reference (already tie-safe, per infeasibility_screen.jl's own fix) ----
wres_dense = screen_hard_winners(θ_full_tie, ctx, Pmat; order = 1:D, full_scan = true)
check("dense reference: origin 1 has >0 wins at destination 1", wres_dense.win_counts[1, 1] > 0)
check("dense reference: origin 2 has >0 wins at destination 1", wres_dense.win_counts[2, 1] > 0)

# ---- ranged/fused screen, BOTH full_scan modes -- this is the code path this fix touches ----
for full_scan in (false, true)
    wres = screen_hard_winners_ranged(θ_full_tie, ctx, Pmat, rsc.envelope; order = 1:D, full_scan = full_scan)
    check("ranged screen (full_scan=$full_scan): win_counts[1,1] matches dense reference",
          wres.win_counts[1, 1] == wres_dense.win_counts[1, 1])
    check("ranged screen (full_scan=$full_scan): win_counts[2,1] matches dense reference (tie-safe win credit)",
          wres.win_counts[2, 1] == wres_dense.win_counts[2, 1])
    # The actual AUD-06 regression: this tie construction sets Pmat[2,1]'s target so low (it is
    # whatever target_shares(ctx) produced originally) that only a TRUE bug would reject it -- the
    # real assertion is structural: no certificate may be issued that blames the tied origin
    # (failing_o==2, failing_d==1) with reject_kind==:winning_range, since origin 2's Hmax_d entry
    # was never populated by the single-pass fusion (it only tracks the strict-first winner).
    false_cert = !wres.feasible && wres.reject_kind === :winning_range && wres.failing_o == 2 && wres.failing_d == 1
    check("ranged screen (full_scan=$full_scan): no false :winning_range certificate against the tied origin",
          !false_cert)
    # And the two screens must agree on overall feasibility for this point (no regression vs the
    # already-validated dense path).
    check("ranged screen (full_scan=$full_scan): feasibility agrees with dense reference",
          wres.feasible == wres_dense.feasible)
end

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
