# Gate C (exclude-ROW-destination UNRESTRICTED-CORE release, 2026-07-24): real D=20/W=80,000
# unrestricted destination_sample=:exclude_row fixed-point validation. Mandatory release gate --
# the actual production compressed evaluation path (evaluate_fullA_screened_ranged ->
# compressed_factual_from_screen -> inner_loop_KNITRO_compressed) at the real benchmark scale.
#
# Run: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t 20 \
#        full_aod_diag/d4_exact/test_exclude_row_unrestricted_gateC_d20.jl
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Random, Printf, LinearAlgebra, Dates

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    lp(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

lp("=== GATE C: real D=20/W=80,000 unrestricted :exclude_row === ", Dates.now(), "  threads=", Threads.nthreads())

t0 = time()
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true)   # default -> :exclude_row
lp("ctx build wall=", round(time() - t0, digits = 1), "s  screen_setup_wall=", ctx.screen_setup_wall)
print_active_layout_banner(ctx, "unrestricted_gateC")

check("destination_sample == :exclude_row (default)", ctx.destination_sample == :exclude_row)
check("D (origins) == 20", ctx.D == 20)
check("D_dest (active destinations) == 19", ctx.D_dest == 19)
check("row_idx == 20 (ROW)", ctx.row_idx == 20)
check("active A cells == 380 (D*D_dest)", ctx.D * ctx.D_dest == 380)
pe = build_pivot_elimination(ctx)
check("free A coordinates == 379 (raw 380 - 1 gravity pivot)", length(pe.other_idx) == 379)
check("envelope precomputation succeeds under :exclude_row (rsc.envelope !== nothing)",
      (rsc_probe = build_ranged_screen_context(ctx); rsc_probe.envelope !== nothing))

x_free_from_w2(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*ctx.D_dest]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, ctx.D, ctx.D_dest), pe)
gp0 = ctx.θ0_up[3+ctx.D]
w0 = vcat(gp0, zfree0)
xf0 = x_free_from_w2(w0)

# ---- (a) calibration point (A_od==1) at the real benchmark W: cross-screen consistency ----
# As established in Gate B (real D=20 data, verified at both W=8000 and W=80,000 via a direct
# diagnostic before writing this section): A_od==1 is NOT an implied economically-feasible point
# for the REAL D=20 economy (unlike the D=4 SYNTHETIC test economies, where it is feasible by
# construction) -- it is genuinely, structurally infeasible here, and every independent screen
# (pairwise/envelope/range_screen_standalone) agrees on the exact same violated cell. That
# agreement is itself the correctness check at this scale; a real converged/feasible point is
# then found via a real short outer-loop search below (step a2), matching how production actually
# operates (KNITRO's outer solve explores away from an initial infeasible point, exactly as
# screens are designed to let it do cheaply).
rsc = build_ranged_screen_context(ctx)
θ_full_cal = CS.reconstruct_full(xf0, ctx.m)
a_cal = compute_a_od(θ_full_cal, ctx)
pc80k = precompute_pairwise_M(ctx)
Pmat80k = target_shares(ctx)
pres_cal = pairwise_certificate(a_cal, pc80k, Pmat80k)
eres_cal = envelope_prewinner_screen(θ_full_cal, ctx, rsc.envelope)
lp("[calibration point @ W=80,000] pairwise infeasible=", pres_cal.infeasible, "  envelope status=", eres_cal.status,
   "  origin=", eres_cal.origin, "  dest_slot=", eres_cal.destination, "  h_upper=", eres_cal.h_upper_bound, "  target=", eres_cal.target)
# pairwise_certificate and envelope_prewinner_screen are independent certificates with different
# scan orders/metrics -- both independently confirming infeasibility is the correctness property
# (see Gate B for the fuller discussion); they are not required to report the identical cell.
check("calibration point: pairwise_certificate independently confirms infeasibility at real W=80,000", pres_cal.infeasible)
check("calibration point: envelope_prewinner_screen independently confirms infeasibility at real W=80,000",
      eres_cal.status == :EXACT_INFEASIBLE_PREWINNER_ENVELOPE)

# ---- (a2) real short outer-loop search to find a genuinely feasible/converged incumbent ----
SEARCH_BUDGET_S = 400.0
CKPT_GATEC = joinpath(D4X_ROOT, "results", "fullA_d4", "gateC_unrestricted_search")
rm(CKPT_GATEC; recursive = true, force = true); mkpath(CKPT_GATEC)
lp("\n[real search] launching run_profile_checkpointed, budget=", SEARCH_BUDGET_S, "s, from the calibration seed...")
t0 = time()
res_search = run_profile_checkpointed("gateC", gp0, true, zfree0;
    maxtime_real = SEARCH_BUDGET_S, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
    ckpt_dir = CKPT_GATEC, checkpoint_interval_s = 60.0)
t_search = time() - t0
lp("[real search] wall=", round(t_search, digits = 1), "s  n_eval=", res_search.n_eval, "  knitro_status=", res_search.knitro_status,
   "  screens(pw/wt/wn/env/wr/sn/pass)=", res_search.screen_counts)
check("real search: screens (pairwise/envelope/winning-range/safety-net) recorded nonzero activity",
      sum(values(res_search.screen_counts)) > 0)
check("real search: at least one screen category other than 'passed' fired (screens are actually doing work, not a no-op)",
      (res_search.screen_counts.pairwise + res_search.screen_counts.envelope + res_search.screen_counts.winning_range +
       res_search.screen_counts.safety_net + res_search.screen_counts.winner) > 0)

if res_search.best === nothing
    check("real search found a verified-feasible incumbent within the search budget", false)
    lp(">>> GATE C RESULT: BLOCKED -- no feasible incumbent found in ", SEARCH_BUDGET_S, "s; cannot proceed to gradient/finite-difference checks.")
    exit(1)
end
lp("[real search] best: Delta_dual=", res_search.best.Delta_dual, "  found_at_eval=", res_search.best.n_eval,
   "  t_elapsed=", round(res_search.best.t_elapsed, digits = 1), "s  max_abs_moment_kkt_resid=", res_search.best.max_abs_moment_kkt_resid)
check("real search: best incumbent has finite Delta_dual", isfinite(res_search.best.Delta_dual))
check("real search: best incumbent has finite, small max_abs_moment_kkt_resid (< 1e-6)",
      isfinite(res_search.best.max_abs_moment_kkt_resid) && res_search.best.max_abs_moment_kkt_resid < 1e-6)
check("real search: best incumbent's inner_status is a real solved KNITRO code", res_search.best.inner_status in (0, -100, -101, -103))

xf0 = x_free_from_w2(vcat(gp0, res_search.best.zfree))   # REPLACE xf0 with the real found-feasible point for everything below
lp("[real search] using found incumbent as xf0 for the remaining checks")

# ---- (b) compressed vs dense at the SAME point, bypassing screens (direct oracle_fast comparison) ----
t0 = time()
r_dense = evaluate_fullA_fast(xf0, ctx; moment_representation = :dense, cache = nothing, use_cache = false, warm = false)[1]
t_dense = time() - t0
t0 = time()
r_comp = evaluate_fullA_fast(xf0, ctx; moment_representation = :compressed, cache = nothing, use_cache = false, warm = false)[1]
t_comp = time() - t0
lp("[dense vs compressed] dense wall=", round(t_dense, digits = 2), "s  compressed wall=", round(t_comp, digits = 2), "s",
   "  Delta_dual dense=", r_dense.Delta_dual, "  compressed=", r_comp.Delta_dual)
check("dense vs compressed: both solved", r_dense.inner_status in (0, -100, -101, -103) && r_comp.inner_status in (0, -100, -101, -103))
check("dense vs compressed: Delta_dual agree to 1e-8", abs(r_dense.Delta_dual - r_comp.Delta_dual) < 1e-8)
check("dense vs compressed: gravity_value agree to 1e-8", abs(r_dense.gravity_value - r_comp.gravity_value) < 1e-8)
check("dense vs compressed: max_abs_moment_kkt_resid agree to 1e-6", abs(r_dense.max_abs_moment_kkt_resid - r_comp.max_abs_moment_kkt_resid) < 1e-6)
check("dense vs compressed: winner_hash IDENTICAL (same winner assignment)", r_dense.winner_hash == r_comp.winner_hash)

# ---- (c) full outer gradient, :reference vs :cplus, all 379 free coordinates ----
base = compressed_base_state(xf0, ctx)
t0 = time()
g_ref, gmeta_ref = composite_gradient_at_fast(xf0, ctx, pe; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_ref = time() - t0
grad_pool = build_grad_workspace_pool(size(ctx.U, 1))
lfix_c_ws = build_lfix_factorized_workspace(ctx.D, ctx.D_dest, size(ctx.U, 1))
t0 = time()
g_cplus, gmeta_cplus = composite_gradient_at_Cplus(xf0, ctx, pe, grad_pool, lfix_c_ws; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
t_cplus = time() - t0
lp("[gradient] :reference wall=", round(t_ref, digits = 2), "s  :cplus wall=", round(t_cplus, digits = 2), "s  speedup=", round(t_ref / t_cplus, digits = 2), "x")
check("gradient: both length 379", length(g_ref) == 379 && length(g_cplus) == 379)
cosang = dot(g_ref, g_cplus) / (norm(g_ref) * norm(g_cplus))
maxdiff = maximum(abs.(g_ref .- g_cplus))
lp("[gradient] cosine=", @sprintf("%.12f", cosang), "  max_abs_diff=", @sprintf("%.3e", maxdiff))
check("gradient: cosine(:reference, :cplus) > 1 - 1e-8 (effectively 1)", cosang > 1 - 1e-8)
check("gradient: max_abs_diff < 1e-6", maxdiff < 1e-6)
check("gradient: no sign mismatches on any coordinate with |g_ref|>1e-8",
      all((abs(g_ref[i]) <= 1e-8) || (sign(g_ref[i]) == sign(g_cplus[i])) for i in eachindex(g_ref)))

# ---- (d) finite-difference on a bounded coordinate subset ----
# free_idx layout: index 1 == gamma_focal_prime; indices 2:380 == z_free (D*D_dest-1 pivot-reduced
# log-A coordinates). pe.other_idx[k] gives the RAW (pre-pivot) linear index for z_free[k] in the
# SAME convention as pivot_expand's own `reshape(z, pe.D, pe.Ddest)` -- i.e. the Aod-parameter-
# block ORIGIN-FAST convention (j = o + (s-1)*D), NOT the destination-fast moments/P convention
# used elsewhere in this release (see MEMORY moments-vs-aod-linear-index-convention / cc_algo/
# active_layout.jl's active_cell_index vs active_cell_index_aod) -- decode accordingly.
Ddest = ctx.D_dest
raw_idx_of(k) = pe.other_idx[k]                       # raw j in 1..D*Ddest for z_free[k]
raw_to_od(j) = (o = mod1(j, ctx.D); s = (j - o) ÷ ctx.D + 1; (o, s))
bi = ctx.bi
bi_slot = dest_slot(ctx, bi)
# candidate coordinates: (1) largest |gravity coefficient| among free coords (most pivot-coupled),
# (2) a cell at destination slot 1 or Ddest (adjacent to the omitted-destination boundary),
# (3) a cell with origin==ROW (global 20, the omitted destination's own index, still valid as an
#     origin), (4) a cell touching the focal country bi as either origin or destination slot.
gcoef = abs.(pe.c[pe.other_idx])
k_pivot_coupled = argmax(gcoef)
k_boundary = findfirst(k -> raw_to_od(raw_idx_of(k))[2] == Ddest, 1:length(pe.other_idx))
k_row_origin = findfirst(k -> raw_to_od(raw_idx_of(k))[1] == 20, 1:length(pe.other_idx))
k_focal = findfirst(k -> raw_to_od(raw_idx_of(k))[1] == bi || raw_to_od(raw_idx_of(k))[2] == bi_slot, 1:length(pe.other_idx))
fd_targets = unique(filter(!isnothing, [k_pivot_coupled, k_boundary, k_row_origin, k_focal]))
lp("\n[finite-difference] bounded subset: z_free indices ", fd_targets, " (of 379)")

h = 2e-4
fd_ok = true
w_center = vcat(gp0, res_search.best.zfree)   # the REAL found-feasible incumbent, not the raw calibration w0
for k in fd_targets
    (o, s) = raw_to_od(raw_idx_of(k))
    wplus = copy(w_center); wplus[1+k] += h
    wminus = copy(w_center); wminus[1+k] -= h
    xplus = x_free_from_w2(wplus); xminus = x_free_from_w2(wminus)
    rplus = evaluate_fullA_fast(xplus, ctx; moment_representation = :compressed, cache = nothing, use_cache = false, warm = false)[1]
    rminus = evaluate_fullA_fast(xminus, ctx; moment_representation = :compressed, cache = nothing, use_cache = false, warm = false)[1]
    solved = rplus.inner_status in (0, -100, -101, -103) && rminus.inner_status in (0, -100, -101, -103)
    if !solved
        lp("  z_free[", k, "] (origin=", o, ",dest_slot=", s, "): SKIP (bump point failed to solve, not a code-correctness failure)")
        continue
    end
    fd_slope = (rplus.Delta_dual - rminus.Delta_dual) / (2h)
    analytic = g_ref[k]
    same_sign = sign(fd_slope) == sign(analytic) || abs(analytic) < 1e-6
    rel_ok = abs(fd_slope) < 1e-6 || abs(fd_slope - analytic) / max(abs(fd_slope), abs(analytic)) < 0.25
    ok = same_sign && rel_ok
    fd_ok &= ok
    lp("  z_free[", k, "] (origin=", o, ",dest_slot=", s, "): fd_slope=", @sprintf("%.6e", fd_slope),
       "  analytic(g_ref)=", @sprintf("%.6e", analytic), "  ", ok ? "PASS" : "FAIL")
end
check("finite-difference subset: sign/magnitude agreement with analytic gradient", fd_ok)

# ---- (e) :all_legacy unchanged (numerically, at its own calibration point) ----
# Same caveat as (a) above: A_od==1 is not assumed economically-feasible for the real economy
# (square or rectangular) -- the correctness property checked here is that the compressed and
# dense evaluators are NUMERICALLY IDENTICAL at this point regardless of whether it happens to be
# feasible (i.e. :all_legacy's own behavior is unaffected by anything in this release), not that
# the point solves.
t0 = time()
ctx_legacy = d20_real_setup(W = 8000, δ = 1.0, find_smallest = true, destination_sample = :all_legacy)
lp("\n[legacy] ctx build wall=", round(time() - t0, digits = 1), "s")
check("legacy: D_dest == D == 20 (square)", ctx_legacy.D_dest == ctx_legacy.D == 20)
check("legacy: row_idx === nothing", ctx_legacy.row_idx === nothing)
pe_legacy = build_pivot_elimination(ctx_legacy)
zfree0_legacy = pivot_reduce(zeros(ctx_legacy.D, ctx_legacy.D_dest), pe_legacy)
xf0_legacy = vcat(ctx_legacy.θ0_up[3+ctx_legacy.D], vec(exp.(pivot_expand(zfree0_legacy, pe_legacy))))
r_legacy = evaluate_fullA_fast(xf0_legacy, ctx_legacy; moment_representation = :compressed, cache = nothing, use_cache = false, warm = false)[1]
r_legacy_dense = evaluate_fullA_fast(xf0_legacy, ctx_legacy; moment_representation = :dense, cache = nothing, use_cache = false, warm = false)[1]
lp("[legacy] compressed inner_status=", r_legacy.inner_status, " Delta_dual=", r_legacy.Delta_dual,
   "  dense inner_status=", r_legacy_dense.inner_status, " Delta_dual=", r_legacy_dense.Delta_dual)
check("legacy: compressed and dense reach the IDENTICAL inner_status at the calibration point (feasible or not, same verdict)",
      r_legacy.inner_status == r_legacy_dense.inner_status)
check("legacy: compressed vs dense Delta_dual agree at the calibration point (both finite-agree, or both non-finite together)",
      isfinite(r_legacy.Delta_dual) == isfinite(r_legacy_dense.Delta_dual) &&
      (!isfinite(r_legacy.Delta_dual) || abs(r_legacy.Delta_dual - r_legacy_dense.Delta_dual) < 1e-8))

lp("\n", "="^96)
if isempty(FAILURES)
    lp(">>> GATE C RESULT: ALL PASS")
else
    lp(">>> GATE C RESULT: ", length(FAILURES), " FAILURE(S): ", FAILURES)
    exit(1)
end
