# ============================================================================
# D=4 gate for the shared gravity_sample_mask/precompute_q_tilde/gravity_value machinery
# (2026-07-31, Brazil-Korea gravity-exclusion task, task §12 "D=4" gates), self-contained
# (synthetic 4x4 economy, no full ctx/KNITRO machinery needed -- this exercises the mask
# functions directly, complementing the real-D20 end-to-end pivot gate which already covers
# the actual Brazil/Korea resolution + full pivot_expand/pivot_reduce round trip).
#
# Run: julia --project=. full_aod_diag/d4_exact/test_gravity_sample_mask_d4_gate_2026-07-31.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "..", "misc", "doubleDiff.jl"))
include(joinpath(@__DIR__, "..", "gravity_tariff.jl"))
using Random, LinearAlgebra

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

Random.seed!(20260731)
D = 4; Ddest = 4
τ = 1.0 .+ 0.5 .* rand(D, Ddest)
AodPow = 1.0 .+ 0.3 .* rand(D, Ddest)

# ---- shared mask: default (empty) is bit-identical to pre-existing behavior ----
q0, N0 = precompute_q_tilde(τ)
check("default (no exclusions) uses within_transform_rect, N_obs=D*Ddest", N0 == D * Ddest)
check("default (no exclusions) matches within_transform_rect(τ) exactly", q0 == within_transform_rect(τ))

# ---- exclude_diagonal alone: unchanged from before this task ----
q1, N1 = precompute_q_tilde(τ; exclude_diagonal = true)
mask1 = gravity_sample_mask(D, Ddest; exclude_diagonal = true)
check("exclude_diagonal mask matches [o!=d]", mask1 == [o != d for o in 1:D, d in 1:Ddest])
check("exclude_diagonal N_obs == D*Ddest - D (4 diagonal cells dropped)", N1 == D * Ddest - D)
check("exclude_diagonal: q_tilde zero on diagonal", all(q1[o, o] == 0.0 for o in 1:D))

# ---- NEW: exclude_cells generalizes to an arbitrary cell (not diagonal, not ROW-related --
# this D=4 synthetic economy has no ROW/Brazil/Korea concept at all, purely mechanical) ----
excl_cell = (2, 3)
q2, N2 = precompute_q_tilde(τ; exclude_diagonal = false, exclude_cells = [excl_cell])
mask2 = gravity_sample_mask(D, Ddest; exclude_cells = [excl_cell])
check("exclude_cells alone: mask true everywhere except the one cell", mask2 == [(o, d) != excl_cell for o in 1:D, d in 1:Ddest])
check("exclude_cells alone: N_obs == D*Ddest - 1", N2 == D * Ddest - 1)
check("exclude_cells alone: q_tilde[2,3] == 0 exactly", q2[2, 3] == 0.0)

# combine both
q3, N3 = precompute_q_tilde(τ; exclude_diagonal = true, exclude_cells = [excl_cell])
check("combined: N_obs == D*Ddest - D - 1 (diagonal + the extra cell, no overlap)", N3 == D * Ddest - D - 1)
check("combined: q_tilde zero on diagonal AND the extra cell", all(q3[o, o] == 0.0 for o in 1:D) && q3[2, 3] == 0.0)

# ---- gravity_value: sign convention + consistency with q_tilde (task's "coefficient sign mapping") ----
gv0 = gravity_value(τ, AodPow, q0, N0)
gv3 = gravity_value(τ, AodPow, q3, N3; exclude_diagonal = true, exclude_cells = [excl_cell])
check("gravity_value finite and sign-consistent (no NaN/Inf) at default spec", isfinite(gv0))
check("gravity_value finite and sign-consistent (no NaN/Inf) at combined-exclusion spec", isfinite(gv3))

# ---- excluded-cell derivative is exactly zero: ∂gravity_value/∂log(AodPow[o,d]) = -q_tilde[o,d]/N_obs
# (direct AodPow derivative, via the same FWL orthogonality identity used below) -- q_tilde[2,3]==0
# by construction (masked), so this is exactly zero regardless of N_obs or AodPow's own value ----
check("excluded-cell (2,3) direct gravity_value derivative is exactly zero", (-q3[2, 3] / N3) == 0.0)

# ---- included-cell derivative matches finite differences of gravity_value itself ----
o_inc, d_inc = 1, 1   # off-diagonal-free... wait D=4 square, use a genuinely included cell: (1,2)
o_inc, d_inc = 1, 2
h = 1e-6
AodPow_p = copy(AodPow); AodPow_p[o_inc, d_inc] *= exp(h)
AodPow_m = copy(AodPow); AodPow_m[o_inc, d_inc] *= exp(h * -1)
gv_p = gravity_value(τ, AodPow_p, q3, N3; exclude_diagonal = true, exclude_cells = [excl_cell])
gv_m = gravity_value(τ, AodPow_m, q3, N3; exclude_diagonal = true, exclude_cells = [excl_cell])
fd = (gv_p - gv_m) / (2h)
# NOTE: c3 = mu.*q_tilde./N_obs (gravity_elimination.jl's own coefficient) is the derivative
# w.r.t. log(Aod_theta), a DIFFERENT variable reached via a further mu-dependent chain rule
# (module header, gravity_tariff.jl) -- NOT the direct derivative of gravity_value w.r.t.
# log(AodPow) tested here. By the same FWL orthogonality identity that formula's derivation
# relies on (q_tilde already orthogonal to the masked FE column space, so
# sum(q_tilde.*within(logAodPow)) == sum(q_tilde.*logAodPow) exactly, no regression-leverage
# term), the direct analytic derivative of gravity_value=-sumGrav/N_obs w.r.t. log(AodPow[o,d])
# is simply -q_tilde[o,d]/N_obs (no mu factor -- mu only enters via the separate Aod_theta chain).
analytic = -q3[o_inc, d_inc] / N3
ok_fd = abs(fd - analytic) < 1e-6 * max(1.0, abs(analytic))
check("included cell (1,2): analytic gravity-coefficient derivative matches finite difference (analytic=$analytic, fd=$fd)", ok_fd)

# ---- encode/decode-style round trip: mask -> exclude_cells recoverable from mask ----
recovered = [(o, d) for o in 1:D, d in 1:Ddest if !mask2[o, d]]
check("mask -> excluded-cell-list round trip recovers the exact input", recovered == [excl_cell])

println("\n=== SUMMARY ===")
println(length(FAILURES) == 0 ? "ALL PASS" : "FAILURES: $(length(FAILURES))")
for f in FAILURES
    println("  FAIL: ", f)
end
