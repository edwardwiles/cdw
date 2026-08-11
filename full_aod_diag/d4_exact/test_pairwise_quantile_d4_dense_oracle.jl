# ================================================================================================
# D=4 dense truth oracle for the pairwise-quantile-independence restriction (draft eq. 32),
# VERSION B: fixed cutoffs + FREE bin masses (2026-08-10).
#
# Cross-checks the production lookup-based code (pairwise_quantile_{mass_transform,bin_context,
# operator,hessian,mass_gradient}.jl) against a NAIVE DENSE reference built directly from raw
# W x n_rows indicator columns (no lookup tricks, no reused table infrastructure) for:
#   1. the stick-breaking mass transform's Jacobian (vs finite differences of the decode itself)
#   2. the forward contraction, with mu-centering
#   3. the transpose/gradient, with mu-centering
#   4-7. every Hessian raw-block family + the centered block + the packed round-trip
#   8-9. the EXACT closed-form mass gradient `d_delta_dual_d_mu` and its chain rule to raw
#        coordinates, against finite differences of the FIXED-DUAL objective.
#
# WHY CHECK 8-9 IS A REAL GATE WITHOUT A SOLVER. `d_delta_dual_d_mu` is the envelope derivative
# `-mean_m * A_{o,a}` with `A_{o,a} = dC_lambda/dmu_{o,a}`. `A` is exact for the FIXED-dual
# objective at ANY `(zeta,lambda)`, not only at the optimum -- the envelope theorem is what lets the
# same expression stand in for the REOPTIMIZED derivative, and that half is gated separately by the
# reoptimized-FD gate (test_pairwise_quantile_outer_gradient_fd.jl) against a real KNITRO solve.
# So here, at an arbitrary lambda and with no solver at all, `d(fixed-dual f)/dmu` must equal
# `+mean_m * A` to FD accuracy, and its negation is what the production function returns.
# `mu` is deliberately set NON-UNIFORM for this check: at `mu = 1/L` every bin's mass is equal, so
# the natural transcription error in the pair term (`mu[p,a]` instead of `mu[p,b]` -- partner's
# mass at the PARTNER's bin index) is invisible.
#
# VERSION A's CHECK 8-9 -- the fixed-dual boundary-crossing cutoff-gradient shortcut against a slow
# O(W) recompute -- is GONE, along with the machinery it gated (pairwise_quantile_cutoff_gradient.jl,
# deleted). Its `r_current`/`psi_scalar` convention was corrected on 2026-08-10 to match production
# (`r_current = r0`, `psi_scalar.(a0)`, no negation) after a self-cancelling double negation hid a
# real sign bug; the same corrected convention is used in CHECK 8-9 below, and must NOT be
# "restored" to the negated form (see memory
# `feedback-self-cancelling-test-convention-cannot-gate-a-sign`).
#
# `L`-generic: `L` (number of quantile bins) is a test parameter, `ARGS[1]` if given else 5 -- this
# file is the fast, standalone, real-scale-independent way to prove the whole restriction is
# genuinely `L`-generic: run once at `L=5` (regression) and once at a DIFFERENT `L` (e.g. `L=7`).
# The `ARGS`-with-fallback here is a TEST-file convenience only, not a production default (this file
# is never included by production code -- see the SCOPE NOTE below).
#
# SCOPE NOTE: this oracle validates the restriction's OWN numerics in isolation, using synthetic
# draws -- it does NOT exercise the real economic H_EE/H_E,restriction cross-block or a live KNITRO
# inner solve. Production code must NEVER include this file.
#
# Usage:  julia --project=. full_aod_diag/d4_exact/test_pairwise_quantile_d4_dense_oracle.jl [L]
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

"Psi(x) at a single point, matching cc_algo/Psi.jl's Psi! piecewise formula exactly. Test-local
(version A kept it in pairwise_quantile_cutoff_gradient.jl, which no longer exists); production
never evaluates Psi this way, it goes through obj.Psi!."
psi_scalar(x::Float64) = x <= 1.0 ? (exp(x) - 1.0) : (0.5 * exp(1) * (x^2 + 1.0) - 1.0)
"Psi'(x) at a single point, matching cc_algo/Psi.jl's dPsi!."
dpsi_scalar(x::Float64) = x <= 1.0 ? exp(x) : (exp(1) * x)

const D4X = @__DIR__
for f in ["pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_mass_gradient.jl"]
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
const L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5   # test-file convenience only, see header note
const NC = L - 1
println("=== D4 dense oracle (VERSION B: fixed cutoffs, free masses), L=$L (n_free_bins=$NC) ===")
U = rand(W, D) .* 3.0 .+ 0.3   # positive support, matches ctx.U's own convention

layout = PairwiseQuantileMassLayout(D, L)
Q = pairwise_quantile_fixed_cutoffs(U, L; cutoff_source = :empirical_quantile)
op = PairwiseQuantileOperator(U, L, Q)
npair = op.npair
nrow = n_total_rows(D, L)

println("D=$D W=$W L=$L npair=$npair nrow=$nrow (marginal=$(n_marginal_rows(D,L)) pair=$(n_pair_rows(D,L)))")
check("D=4 counts: npair==6", npair == 6)
check("D=4 counts: nrow == NC*D + NC^2*6 (marginal+pair)", nrow == NC * D + NC^2 * 6 &&
      n_marginal_rows(D, L) == NC * D && n_pair_rows(D, L) == NC^2 * 6)
check("fixed cutoffs strictly ordered at every origin", all(o -> all(diff(Q[:, o]) .> 0), 1:D))

# Non-uniform masses on purpose (see the header note on CHECK 8-9): a random draw on the simplex,
# via random raw coordinates, so no two bins share a mass and no origin shares another's.
raw = randn(n_raw(layout)) .* 0.5
state = PairwiseQuantileMassState(D, L)
set_pairwise_quantile_masses!(state, raw, layout)
MU = state.mu
check("decoded masses are strictly positive", all(MU .> 0.0))
check("decoded free masses sum to < 1 at every origin", all(o -> sum(@view MU[o, :]) < 1.0, 1:D))
check_tol("mu_last == 1 - sum(free masses)",
          maximum(abs, state.mu_last .- (1.0 .- vec(sum(MU, dims = 2)))), 1e-14)
check_tol("Pcum is the running sum of mu",
          maximum(abs, state.Pcum .- cumsum(MU, dims = 2)), 1e-14)
check("mu is NOT uniform (this test would be blind to a pair-index bug if it were)",
      maximum(MU) - minimum(MU) > 1e-3)

# uniform_mass_raw must decode back to exactly 1/L everywhere -- the equivalence anchor's own
# starting point, checked here where it is cheap rather than only at real D=20.
begin
    st_u = PairwiseQuantileMassState(D, L)
    set_pairwise_quantile_masses!(st_u, uniform_mass_raw(layout), layout)
    check_tol("uniform_mass_raw decodes to mu == 1/L exactly",
              max(maximum(abs, st_u.mu .- 1.0 / L), maximum(abs, st_u.mu_last .- 1.0 / L)), 1e-14)
end

# empirical_mass_raw must decode back to the draws' own bin frequencies, and under
# :empirical_quantile cutoffs those are 1/L up to the rounding of W/L to whole draws.
begin
    counts = pairwise_quantile_bin_counts(op)
    check("bin counts sum to W at every origin", all(o -> sum(@view counts[o, :]) == W, 1:D))
    st_e = PairwiseQuantileMassState(D, L)
    set_pairwise_quantile_masses!(st_e, empirical_mass_raw(counts, layout), layout)
    check_tol("empirical_mass_raw decodes to the draws' own bin frequencies",
              maximum(abs, st_e.mu .- counts[:, 1:NC] ./ W), 1e-13)
    check_tol("under :empirical_quantile cutoffs those frequencies are 1/L up to W/L rounding",
              maximum(abs, st_e.mu .- 1.0 / L), 2.0 / W)
end

# ==== CHECK 1: stick-breaking Jacobian vs finite differences ====
begin
    o = 2
    base = raw_index(layout, o, 1)
    raw_o = raw[base:base+NC-1]
    mu_row = zeros(NC); decode_origin_masses!(mu_row, raw_o)
    J = zeros(NC, NC); mass_jacobian_block!(J, raw_o, mu_row)
    h = 1e-6; Jfd = zeros(NC, NC)
    buf = zeros(NC)
    for k in 1:NC
        rp = copy(raw_o); rp[k] += h
        rm = copy(raw_o); rm[k] -= h
        decode_origin_masses!(buf, rp); mp = copy(buf)
        decode_origin_masses!(buf, rm); mm = copy(buf)
        Jfd[:, k] .= (mp .- mm) ./ (2h)
    end
    check_tol("mass Jacobian vs finite differences", maximum(abs.(J .- Jfd)), 1e-8)
    check("mass Jacobian is lower-triangular by construction",
          all(k -> all(a -> J[a, k] == 0.0, 1:k-1), 1:NC))
    # Structural identity: masses sum to 1 - mu_L, so a raw step reallocates mass ACROSS bins and
    # the column sums of J are exactly -d(mu_L)/d(raw_k). Checked against an independent FD of the
    # remainder itself, so it is a cross-check, not a restatement.
    dlast = zeros(NC)
    for k in 1:NC
        rp = copy(raw_o); rp[k] += h
        rm = copy(raw_o); rm[k] -= h
        dlast[k] = (decode_origin_masses!(buf, rp) - decode_origin_masses!(buf, rm)) / (2h)
    end
    check_tol("sum_a dmu_a/draw_k == -dmu_L/draw_k (mass is conserved)",
              maximum(abs, vec(sum(J, dims = 1)) .+ dlast), 1e-8)
end

# ==== dense reference builder (shared by checks 2-9) ====
lambda_M = randn(D, NC)
lambda_P = randn(NC, NC, npair)
Gdense = zeros(W, nrow)
tvec_ref = zeros(nrow)
for o in 1:D, a in 1:NC
    i = marginal_row(o, a, L)
    @views Gdense[:, i] .= (op.bin[:, o] .== a) .- MU[o, a]
    tvec_ref[i] = MU[o, a]
end
for pidx in 1:npair
    (p, q) = op.pairs[pidx]
    for b in 1:NC, a in 1:NC
        j = pair_row(D, pidx, a, b, L)
        @views Gdense[:, j] .= ((op.bin[:, p] .== a) .& (op.bin[:, q] .== b)) .- MU[p, a] * MU[q, b]
        tvec_ref[j] = MU[p, a] * MU[q, b]
    end
end
lam_flat = zeros(nrow)
for o in 1:D, a in 1:NC
    lam_flat[marginal_row(o, a, L)] = lambda_M[o, a]
end
for pidx in 1:npair, b in 1:NC, a in 1:NC
    lam_flat[pair_row(D, pidx, a, b, L)] = lambda_P[a, b, pidx]
end

# The shared target-vector builder must agree with the dense reference's own per-column constants.
begin
    tvec = zeros(nrow)
    pairwise_quantile_target_vector!(tvec, op, state)
    check_tol("pairwise_quantile_target_vector! vs the dense reference's own column constants",
              maximum(abs.(tvec .- tvec_ref)), 1e-15)
end

# ==== CHECK 2: forward ====
arg0 = zeros(W)
pairwise_quantile_forward!(arg0, lambda_M, lambda_P, op, state)
R_lookup = -arg0
R_dense = Gdense * lam_flat
check_tol("forward! vs dense G*lambda (mu-centered)", maximum(abs.(R_lookup .- R_dense)), 1e-9)

# ==== CHECK 3: transpose ====
draw_weights = rand(W) .+ 0.1
g_M = zeros(D, NC); g_P = zeros(NC, NC, npair)
tls = build_pairwise_quantile_thread_scratch(D, npair, L)
scratch_tr = PairwiseQuantileTransposeScratch(D, npair, L)
pairwise_quantile_transpose!(g_M, g_P, draw_weights, op, state, tls, scratch_tr)
g_dense_flat = -(1.0 / W) .* (Gdense' * draw_weights)
g_lookup_flat = zeros(nrow)
for o in 1:D, a in 1:NC
    g_lookup_flat[marginal_row(o, a, L)] = g_M[o, a]
end
for pidx in 1:npair, b in 1:NC, a in 1:NC
    g_lookup_flat[pair_row(D, pidx, a, b, L)] = g_P[a, b, pidx]
end
check_tol("transpose! vs dense adjoint -(1/W)G'w (mu-centered)", maximum(abs.(g_lookup_flat .- g_dense_flat)), 1e-9)

# ==== CHECK 4-7: Hessian (every raw block family + centering + packing) ====
h = rand(W) .+ 0.05
Ghw = Gdense .* h
Xraw = Gdense' * Ghw
S_dense = sum(h)
# NOTE: Gdense here is the CENTERED feature matrix (columns already have their own mu-target
# subtracted), so Xraw = G'WG is ALREADY the doubly-centered Gram matrix directly (no separate
# r t'+t r' term needed) -- an INDEPENDENT reference construction from the lookup code's own
# centering path (which centers the RAW indicator via the X'WX-rt'-tr'+Stt' identity on UNCENTERED
# features), so agreement between the two is a genuine cross-check of the centering algebra, not a
# tautology. It is also where a wrong per-row target `c_I` would show up.
Hdense = Xraw ./ W

tabs = PairwiseQuantileHessianTables(op)
build_pairwise_quantile_hessian_tables!(tabs, op, h, tls)
check_tol("Hessian: sum(h) matches S", abs(tabs.S - S_dense), 1e-9)

HfullR = zeros(nrow, nrow)
fill_pairwise_quantile_hessian_raw!(HfullR, op, tabs)
center_and_scale_pairwise_quantile_hessian!(HfullR, op, state, tabs)
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

# ==== CHECK 8: closed-form d(Delta_dual)/dmu vs FD of the FIXED-DUAL objective ====
# SIGN CONVENTION (corrected 2026-08-10, carried into version B): `r` must be in the SAME convention
# the production solve uses -- `pairwise_quantile_forward!` SUBTRACTS the restriction contribution
# into its accumulator (`arg0[w] -= Rw`), exactly as the economic block does, so starting from zeros
# gives `r = -G_R*lambda_R` directly and NOTHING is negated here. Version A's oracle briefly read
# `r_current = -r0` and `psi_scalar.(-a0)`, negating BOTH sides; the two errors cancelled and the
# check passed to 1e-16 under a convention no real solve ever uses, which is exactly how a genuine
# sign bug survived it. Do not reintroduce those negations.
function fixed_dual_f(raw_vec::Vector{Float64})
    st = PairwiseQuantileMassState(D, L)
    set_pairwise_quantile_masses!(st, raw_vec, layout)
    a0 = zeros(W)
    pairwise_quantile_forward!(a0, lambda_M, lambda_P, op, st)
    return sum(psi_scalar.(a0)) / W
end
r0 = zeros(W)
pairwise_quantile_forward!(r0, lambda_M, lambda_P, op, state)
mean_m = sum(dpsi_scalar.(r0)) / W

d_mu = d_delta_dual_d_mu(lambda_M, lambda_P, MU, op; mean_m = mean_m)
# FD in MASS space directly: perturb one mu_{o,a} and renormalize nothing -- the masses are free
# coordinates of the moment TARGETS, so a lone mu can move on its own. Done by rebuilding the raw
# vector from perturbed masses through the transform's own exact inverse, which keeps this an FD of
# the same object the production code differentiates.
d_mu_fd = zeros(D, NC)
hmu = 1e-7
for oo in 1:D, aa in 1:NC
    bidx = raw_index(layout, oo, 1)
    mp = collect(MU[oo, :]); mp[aa] += hmu
    mm = collect(MU[oo, :]); mm[aa] -= hmu
    rp = copy(raw); rm = copy(raw)
    raw_from_origin_masses!(@view(rp[bidx:bidx+NC-1]), mp)
    raw_from_origin_masses!(@view(rm[bidx:bidx+NC-1]), mm)
    d_mu_fd[oo, aa] = -(fixed_dual_f(rp) - fixed_dual_f(rm)) / (2hmu)   # Delta = -f
end
rel8 = maximum(abs, d_mu .- d_mu_fd) / max(maximum(abs, d_mu_fd), eps())
check_tol("closed-form d(Delta)/dmu vs fixed-dual FD (relative)", rel8, 1e-6)

# NEGATIVE CONTROL: the pair term must use the PARTNER's mass at the PARTNER's bin index. Rebuilding
# it with `mu[p,a]` (same bin index for both) is the natural transcription error; at non-uniform mu
# it must disagree decisively. A gate nobody has watched fail is not known to be a gate.
d_mu_wrong = zeros(D, NC)
for aa in 1:NC, oo in 1:D
    d_mu_wrong[oo, aa] -= lambda_M[oo, aa]
end
for pidx in 1:npair
    (oo, pp) = op.pairs[pidx]
    for bb in 1:NC, aa in 1:NC
        lp = lambda_P[aa, bb, pidx]
        d_mu_wrong[oo, aa] -= MU[pp, aa] * lp   # WRONG on purpose: partner's mass at bin `aa`, not `bb`
        d_mu_wrong[pp, bb] -= MU[oo, bb] * lp   # WRONG on purpose
    end
end
d_mu_wrong .*= mean_m
rel8_wrong = maximum(abs, d_mu_wrong .- d_mu_fd) / max(maximum(abs, d_mu_fd), eps())
check("NEGATIVE CONTROL: the mu[p,a]-instead-of-mu[p,b] pair index disagrees decisively",
      rel8_wrong > 1e-3)
println("  (control relative error = ", rel8_wrong, " vs correct ", rel8, ")")

# ==== CHECK 9: chain rule to raw coordinates vs FD of the fixed-dual objective in RAW space ====
g_raw = chain_mass_gradient_to_raw(d_mu, raw, MU, layout)
g_raw_fd = zeros(n_raw(layout))
hraw = 1e-6
for j in 1:n_raw(layout)
    rp = copy(raw); rp[j] += hraw
    rm = copy(raw); rm[j] -= hraw
    g_raw_fd[j] = -(fixed_dual_f(rp) - fixed_dual_f(rm)) / (2hraw)   # Delta = -f
end
rel9 = norm(g_raw .- g_raw_fd) / max(norm(g_raw_fd), eps())
check_tol("chain rule to raw coords vs fixed-dual FD (relative L2)", rel9, 1e-6)
check("raw-space gradient is finite and non-degenerate", all(isfinite, g_raw) && any(!=(0.0), g_raw))

println()
println(ALL_PASS[] ? "ALL PAIRWISE-QUANTILE D4 ORACLE CHECKS PASSED" : "SOME CHECKS FAILED -- see above")
ALL_PASS[] || error("test_pairwise_quantile_d4_dense_oracle.jl: one or more checks FAILED")
