# ============================================================================
# Remediation task Part F (sequential branch): unsuccessful-solve regression test for
# recover_lfd's active production gate.
#
# Traced live (not assumed): recover_lfd (run_profiled_production.jl, the ACTIVE production
# entry point on production/sequential-linearized -- confirmed no other recover_lfd copy in this
# tree is reachable from seq_gravcol/the outer search) already gates correctly:
#   nStatus in (0,-100,-101,-103) || return fill(1/W,W), false        (KNITRO status check)
#   all(isfinite, x)              || return fill(1/W,W), false        (finite dual state)
#   isfinite(s) && s>0 && all(isfinite,LFD) && all(>=(0),LFD)
#                                  || return fill(1/W,W), false        (residual/positivity check)
# This is the 2026-07-16 fix referenced in this repo's own memory/handoff docs
# (head_to_head/HANDOFF_2026-07-16_recover_lfd_bug.md); confirmed still present by direct code
# reading in this remediation session. The independent assessment's F2 finding (10 OTHER
# recover_lfd copies in the tree bind nStatus without checking it) is real but does NOT touch
# this active production path -- those copies live in phase-5/diagnostic/head-to-head scripts,
# none of which seq_gravcol's own outer search calls.
#
# This test replaces the historical smoke_test_recover_lfd_fix.jl/check_recover_lfd_status.jl
# pair (both depend on a saved out_gc/gc_T2_warm.jld2 fixture from a prior investigation that is
# not present in this worktree) with a SELF-CONTAINED pathological point: the real calibration
# theta with its A_od block scaled by an extreme factor, chosen to force a genuinely
# infeasible/unbounded inner dual solve rather than depending on a historical artifact.
# ============================================================================
using Test
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf

@testset "recover_lfd rejects a genuinely unsuccessful inner solve (active production path)" begin
    # Sanity anchor: the real calibration theta itself must be a GOOD point (recover_lfd should
    # accept it) -- otherwise this test can't distinguish "the gate works" from "everything gets
    # rejected regardless".
    col0, R0, Rcol0, umat0, p0, ok0 = seq_gravcol(θr0; δ = 1.0, maxit = 20, tol = 5e-4)
    @printf("calibration theta: ok=%s\n", ok0)
    @test ok0

    # Pathological point: scale the A_od block by a large factor (1e6) -- forces the implied
    # trade-cost/price structure far enough from anything gravity-consistent that the inner dual
    # solve should genuinely fail (infeasible/unbounded), not just converge slowly.
    θ_bad = copy(θr0)
    θ_bad[4:end] .*= 1.0e6
    col1, R1, Rcol1, umat1, p1, ok1 = seq_gravcol(θ_bad; δ = 1.0, maxit = 20, tol = 5e-4)
    @printf("pathological theta (A_od x 1e6): ok=%s\n", ok1)
    @test !ok1   # must be REJECTED, not silently accepted with a bogus LFD

    # Direct unit check of recover_lfd itself (not just seq_gravcol, which has other gates too):
    # confirms specifically that recover_lfd's own three-layer check (nStatus / isfinite(x) /
    # LFD positivity-finiteness) is what catches this, by calling it directly at the same
    # pathological theta with the same (moments_fn, d) convention seq_gravcol itself uses.
    oci = D + 1 + 1
    p_direct, ok_direct = recover_lfd(θ_bad, EK_moments_focal_norm_directgp!, D + 1)
    @printf("recover_lfd direct call at pathological theta: ok=%s\n", ok_direct)
    @test !ok_direct
    @test p_direct == fill(1.0 / W, W)   # the documented safe fallback, not a derived (bogus) LFD
end
println("All recover_lfd unsuccessful-solve tests passed.")
