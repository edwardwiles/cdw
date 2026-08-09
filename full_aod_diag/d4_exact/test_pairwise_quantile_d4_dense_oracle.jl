# ================================================================================================
# D=4 dense truth oracle for the pairwise-quantile-independence restriction (draft eq. 32),
# prototype/pairwise-quantile-independence-2026-08-09.
#
# Cross-checks the production lookup-based code (pairwise_quantile_{cutoff_transform,bin_context,
# operator,hessian,cutoff_gradient}.jl) against a NAIVE DENSE reference built directly from raw
# W x n_rows indicator columns (no lookup tricks, no reused table infrastructure) for: the ordered-
# cutoff transform's Jacobian, forward contraction, transpose/gradient, every Hessian raw-table
# block (H_MM same-origin/cross-origin, H_MP same-origin/disjoint, H_PP same-pair/shared-origin/
# disjoint), the centered+packed Hessian, and the fixed-dual boundary-crossing cutoff-gradient
# shortcut (checked against a SLOW full O(W) recompute at the SAME explicit cutoff bump -- an EXACT
# identity, not an approximate finite-difference comparison, since the fixed-dual objective is a
# genuine step function of any single cutoff between consecutive draws, so an "infinitesimal-h"
# comparison would be meaningless here; see this file's own CHECK 8/9 comments).
#
# SCOPE NOTE: this oracle validates the restriction's OWN numerics (forward/transpose/Hessian/
# cutoff-gradient) in isolation, using synthetic draws -- it does NOT exercise the real economic
# H_EE/H_E,restriction cross-block or a live KNITRO inner solve (that requires the full production
# context, `pairwise_quantile_cross_hessian.jl` + `pairwise_quantile_checkpoint.jl`, which follow
# the researched WinnerPairHessCtx/run_originzc_upper_checkpointed patterns but have not been
# exercised end-to-end against a live D20/W=100k campaign in this session -- see the session status
# doc). Production code must NEVER include this file.
#
# Uses this repo's own check()/PASS-FAIL convention (see e.g. test_zc_centered_cache_d4.jl).
# ================================================================================================

function packed_pair_index(D::Int)
    pairs = Vector{Tuple{Int,Int}}(undef, div(D * (D - 1), 2))
    k = 0
    for o in 1:D-1, p in o+1:D
        k += 1
        pairs[k] = (o, p)
    end
    return pairs
end
function pair_oi_to_lin(o::Int, p::Int, D::Int)
    o, p = o < p ? (o, p) : (p, o)
    return div((o - 1) * (2D - o), 2) + (p - o)
end
# NOTE: production code gets packed_pair_index/pair_oi_to_lin from cm_meanzc_moments.jl (verbatim
# reuse, per the plan) -- duplicated here ONLY because that file pulls in the full ~90-file
# production include chain (KNITRO, operator_psi_bundle.jl, etc.) that this standalone oracle
# deliberately does not depend on. The two definitions above are copied byte-for-byte from
# cm_meanzc_moments.jl:68-82.

const D4X = @__DIR__
for f in ["pairwise_quantile_cutoff_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl", "pairwise_quantile_cutoff_gradient.jl"]
    include(joinpath(D4X, f))
end
using Random, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end
function check_tol(name::AbstractString, err::Real, tol::Real)
    cond = err < tol
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name, "  (err=", err, ", tol=", tol, ")")
end

Random.seed!(20260809)
const D = 4
const W = 6000
U = rand(W, D) .* 3.0 .+ 0.3   # positive support, matches ctx.U's own convention

op = PairwiseQuantileOperator(U)
layout = PairwiseQuantileCutoffLayout(D)
raw = randn(n_raw(layout)) .* 0.3
state = PairwiseQuantileBinState(W, D)
refresh_pairwise_quantile_bins!(state, op, U, raw, layout)
npair = op.npair
nrow = n_total_rows(D)

println("D=$D W=$W npair=$npair nrow=$nrow (marginal=$(n_marginal_rows(D)) pair=$(n_pair_rows(D)))")
check("D=4 counts: npair==6", npair == 6)
check("D=4 counts: nrow==112 (16 marginal + 96 pair)", nrow == 112 && n_marginal_rows(D) == 16 && n_pair_rows(D) == 96)

# ==== CHECK 1: ordering + Jacobian vs finite differences ====
check("cutoffs strictly ordered at every origin", all(o -> all(diff(state.Q[:, o]) .> 0), 1:D))
begin
    o = 2
    base = raw_index(layout, o, 1)
    raw4 = (raw[base], raw[base+1], raw[base+2], raw[base+3])
    Qcol = state.Q[:, o]
    J = zeros(4, 4); cutoff_jacobian_block!(J, raw4, Qcol)
    h = 1e-6; Jfd = zeros(4, 4)
    for k in 1:4
        rp = collect(raw4); rp[k] += h
        rm = collect(raw4); rm[k] -= h
        qp = exp.(collect(decode_origin_logcutoffs(Tuple(rp))))
        qm = exp.(collect(decode_origin_logcutoffs(Tuple(rm))))
        Jfd[:, k] .= (qp .- qm) ./ (2h)
    end
    check_tol("cutoff Jacobian vs finite differences", maximum(abs.(J .- Jfd)), 1e-6)
end

# ==== dense reference builder (shared by checks 2-7) ====
lambda_M = randn(D, 4)
lambda_P = randn(4, 4, npair)
Gdense = zeros(W, nrow)
tvec = zeros(nrow)
for o in 1:D, a in 1:4
    i = marginal_row(o, a)
    @views Gdense[:, i] .= (state.bin[:, o] .== a) .- 0.2
    tvec[i] = 0.2
end
for pidx in 1:npair
    (p, q) = op.pairs[pidx]
    for b in 1:4, a in 1:4
        j = pair_row(D, pidx, a, b)
        @views Gdense[:, j] .= ((state.bin[:, p] .== a) .& (state.bin[:, q] .== b)) .- 0.04
        tvec[j] = 0.04
    end
end
lam_flat = zeros(nrow)
for o in 1:D, a in 1:4
    lam_flat[marginal_row(o, a)] = lambda_M[o, a]
end
for pidx in 1:npair, b in 1:4, a in 1:4
    lam_flat[pair_row(D, pidx, a, b)] = lambda_P[a, b, pidx]
end

# ==== CHECK 2: forward ====
arg0 = zeros(W)
pairwise_quantile_forward!(arg0, lambda_M, lambda_P, op, state)
R_lookup = -arg0
R_dense = Gdense * lam_flat
check_tol("forward! vs dense G*lambda", maximum(abs.(R_lookup .- R_dense)), 1e-9)

# ==== CHECK 3: transpose ====
draw_weights = rand(W) .+ 0.1
g_M = zeros(D, 4); g_P = zeros(4, 4, npair)
tls = build_pairwise_quantile_thread_scratch(D, npair)
scratch_tr = PairwiseQuantileTransposeScratch(D, npair)
pairwise_quantile_transpose!(g_M, g_P, draw_weights, op, state, tls, scratch_tr)
g_dense_flat = -(1.0 / W) .* (Gdense' * draw_weights)
g_lookup_flat = zeros(nrow)
for o in 1:D, a in 1:4
    g_lookup_flat[marginal_row(o, a)] = g_M[o, a]
end
for pidx in 1:npair, b in 1:4, a in 1:4
    g_lookup_flat[pair_row(D, pidx, a, b)] = g_P[a, b, pidx]
end
check_tol("transpose! vs dense adjoint -(1/W)G'w", maximum(abs.(g_lookup_flat .- g_dense_flat)), 1e-9)

# ==== CHECK 4-7: Hessian (every raw block family + centering + packing) ====
h = rand(W) .+ 0.05
Ghw = Gdense .* h
Xraw = Gdense' * Ghw
r_dense = Gdense' * h
S_dense = sum(h)
tvec2 = tvec   # same target vector as above (already includes the -t offset baked into Gdense... )
# NOTE: Gdense here is the CENTERED feature matrix (columns already have -0.2/-0.04 subtracted),
# so Xraw = G'WG is ALREADY the doubly-centered Gram matrix directly (no separate r t'+t r' term
# needed) -- this is an INDEPENDENT reference construction from the lookup code's own centering
# path (which centers the RAW indicator via the X'WX-rt'-tr'+Stt' identity on UNCENTERED features),
# so agreement between the two is a genuine cross-check of the centering algebra, not a tautology.
Hdense = Xraw ./ W

tabs = PairwiseQuantileHessianTables(op)
build_pairwise_quantile_hessian_tables!(tabs, op, state, h, tls)
check_tol("Hessian: sum(h) matches S", abs(tabs.S - S_dense), 1e-9)

HfullR = zeros(nrow, nrow)
fill_pairwise_quantile_hessian_raw!(HfullR, op, tabs)
center_and_scale_pairwise_quantile_hessian!(HfullR, op, tabs)
check_tol("Hessian: full centered block vs INDEPENDENT centered-dense reference", maximum(abs.(HfullR .- Hdense)), 1e-8)
check_tol("Hessian: HfullR symmetric", maximum(abs.(HfullR .- HfullR')), 1e-10)

hvec = zeros(div(nrow * (nrow + 1), 2))
pack_upper_pairwise_quantile_hessian!(hvec, HfullR, nrow)
Hrecon = zeros(nrow, nrow)
k = 0
for i in 1:nrow, j in i:nrow
    global k += 1
    Hrecon[i, j] = hvec[k]; Hrecon[j, i] = hvec[k]
end
check_tol("Hessian: packed round-trip", maximum(abs.(Hrecon .- HfullR)), 1e-12)

# ==== CHECK 8-9: cutoff gradient (fixed-dual boundary-crossing shortcut, exact vs slow O(W)) ====
r0 = zeros(W)
pairwise_quantile_forward!(r0, lambda_M, lambda_P, op, state)
r_current = -r0
f0 = sum(psi_scalar.(r_current)) / W

function slow_full_delta(U, op, Qmod, lambda_M, lambda_P, f0)
    D = op.D; W = op.W
    st = PairwiseQuantileBinState(W, D)
    st.Q .= Qmod
    for w in 1:W, oo in 1:D
        st.bin[w, oo] = UInt8(searchsortedfirst(view(Qmod, :, oo), U[w, oo]))
    end
    a0 = zeros(W)
    pairwise_quantile_forward!(a0, lambda_M, lambda_P, op, st)
    fnew = sum(psi_scalar.(-a0)) / W
    return fnew - f0, st
end

o8 = 3; r8 = 2
q_old8 = state.Q[r8, o8]
q_up8 = q_old8 + 0.06
(df_fast_up, k_up8) = fixed_dual_delta_f(op, state, o8, r8, q_up8, lambda_M, lambda_P, r_current)
Qmod_up = copy(state.Q); Qmod_up[r8, o8] = q_up8
(df_slow_up, st_up) = slow_full_delta(U, op, Qmod_up, lambda_M, lambda_P, f0)
check_tol("cutoff gradient: fast Delta_f (up) vs slow O(W) recompute", abs(df_fast_up - df_slow_up), 1e-10)
check("cutoff gradient: crossed count matches brute-force diff (up)",
      k_up8 == count(w -> st_up.bin[w, o8] != state.bin[w, o8], 1:W))

q_down8 = q_old8 - 0.05
(df_fast_dn, k_dn8) = fixed_dual_delta_f(op, state, o8, r8, q_down8, lambda_M, lambda_P, r_current)
Qmod_dn = copy(state.Q); Qmod_dn[r8, o8] = q_down8
(df_slow_dn, st_dn) = slow_full_delta(U, op, Qmod_dn, lambda_M, lambda_P, f0)
check_tol("cutoff gradient: fast Delta_f (down) vs slow O(W) recompute", abs(df_fast_dn - df_slow_dn), 1e-10)
check("cutoff gradient: crossed count matches brute-force diff (down)",
      k_dn8 == count(w -> st_dn.bin[w, o8] != state.bin[w, o8], 1:W))

grad_raw = zeros(n_raw(layout))
cutoff_secant_gradient!(grad_raw, op, state, lambda_M, lambda_P, r_current, layout, raw; min_crossed=25)
check("cutoff gradient: full orchestration finite and non-degenerate", all(isfinite, grad_raw) && any(!=(0.0), grad_raw))

println()
println(ALL_PASS[] ? "ALL PAIRWISE-QUANTILE D4 ORACLE CHECKS PASSED" : "SOME CHECKS FAILED -- see above")
ALL_PASS[] || error("test_pairwise_quantile_d4_dense_oracle.jl: one or more checks FAILED")
