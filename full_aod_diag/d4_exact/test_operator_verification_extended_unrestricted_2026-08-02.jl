# Phase 10 verifier audit (2026-08-02): extends test_operator_verification_unrestricted.jl's D=4
# gate with the two genuine gaps found in the checklist audit -- (1) kkt_resid_E/france_ratio_resid
# (now returned by verify_inner_solution_operator_unrestricted!, trivially equal to kkt_resid since
# G=E only for this family) and (2) recovered full factual shares
# (verify_recovered_full_factual_shares, operator_verification.jl). Mirrors
# test_operator_verification_unrestricted.jl's own setup exactly -- purely additive, does not
# modify that file.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "compressed_moments.jl", "compressed_cc_inner.jl", "compressed_factual_buffer_reuse.jl", "operator_verification.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

all_pass = true
println("=== D=4 unrestricted EXTENDED operator verification gate (Phase 10 audit) ===")
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

base = solve_base_state(x_free_calib, ctx)
obj = ctx.obj
W = size(obj.U, 1)
cf = cf_build(base.θ_full0, ctx; check_ties = true)
ov = verify_inner_solution_operator_unrestricted!(base.ζstar, base.λstar, cf, obj, W)

# --- (1) kkt_resid_E / france_ratio_resid ---
ok_E = abs(ov.kkt_resid_E - ov.kkt_resid) < 1e-12
ok_france = isfinite(ov.france_ratio_resid) && ov.france_ratio_resid < 1e-6
global all_pass &= ok_E && ok_france
@printf("  kkt_resid_E=%.3e (==kkt_resid=%.3e: %s)  france_ratio_resid=%.3e\n", ov.kkt_resid_E, ov.kkt_resid, ok_E, ov.france_ratio_resid)

# --- (2) recovered full factual shares ---
m_weights, _ = verify_namedtuple_from_operator(ov, obj, W, base.inner_status)
rec = verify_recovered_full_factual_shares(base.θ_full0, ctx, cf, m_weights)
ok_winner = rec.max_winner_mismatch == 0
ok_ratio = rec.max_share_ratio_diff < 1e-6
global all_pass &= ok_winner && ok_ratio
@printf("  winner_mismatch=%d  share_ratio_diff=%.3e\n", rec.max_winner_mismatch, rec.max_share_ratio_diff)

println()
println(all_pass ? "ALL PASS" : "SOME FAILURES")
all_pass || error("unrestricted EXTENDED operator verification d4 gate FAILED")
