println("=== :cplus backend under destination_sample=:exclude_row validation starting ===")
flush(stdout)

const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)
using Printf, Random

W = 80000
ALL_PASS = Ref(true)

function build_calib(destination_sample)
    ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :pseudorandom,
                                  draw_seed = 20260719, destination_sample = destination_sample)
    pe0 = build_pivot_elimination(ctx0)
    D = ctx0.D; Ddest = ctx0.D_dest
    Aod_theta_natural = ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D*Ddest]
    zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, Ddest), pe0)
    gp0 = ctx0.θ0_up[3+D]
    w0 = vcat(gp0 * 1.01, zfree0)
    x_free_calib = vcat(w0[1], vec(exp.(pivot_expand(w0[2:end], pe0))))
    return ctx0, pe0, w0, x_free_calib
end

println("--- :exclude_row (NEW true-shrink) ---"); flush(stdout)
ctx_excl, pe_excl, w0_excl, xfc_excl = build_calib(:exclude_row)
println("ctx_excl.D=", ctx_excl.D, " D_dest=", ctx_excl.D_dest, " pe.D=", pe_excl.D, " pe.Ddest=", pe_excl.Ddest)
flush(stdout)

println("--- :all_legacy (reference) ---"); flush(stdout)
ctx_leg, pe_leg, w0_leg, xfc_leg = build_calib(:all_legacy)
println("ctx_leg.D=", ctx_leg.D, " D_dest=", ctx_leg.D_dest, " pe.D=", pe_leg.D, " pe.Ddest=", pe_leg.Ddest)
flush(stdout)

snaps10 = nested_grid_sequence([10])[10]
cfg_oz = OriginZCConfig(distribution_restriction = :origin_specific_moments, K_mean = 1, K_pair = 0,
                        power_target_layout = :origin_by_power, meanzc_basis = :direct)

println("\n=== Building CM (L=10, :anchored) + origin-ZC (K_mean=1, origin_by_power) production contexts on BOTH arms ===")
flush(stdout)
pcx_cm_excl = build_cm_production_context(ctx_excl, CS; L = 10, contrasts = :anchored, probs = snaps10)
pcx_cm_leg  = build_cm_production_context(ctx_leg,  CS; L = 10, contrasts = :anchored, probs = snaps10)
layout_excl = originzc_make_layout(cfg_oz, ctx_excl.D)
layout_leg  = originzc_make_layout(cfg_oz, ctx_leg.D)
pcx_oz_excl = build_originzc_production_context(ctx_excl, CS, layout_excl)
pcx_oz_leg  = build_originzc_production_context(ctx_leg,  CS, layout_leg)
nu0_excl = ones(n_eta(layout_excl)); nu0_leg = ones(n_eta(layout_leg))
println("CM       :exclude_row ncore=", pcx_cm_excl.aug.ncore, " ncm=", pcx_cm_excl.aug.ncm, " d_new=", pcx_cm_excl.ctx_cm.obj.d)
println("CM       :all_legacy  ncore=", pcx_cm_leg.aug.ncore,  " ncm=", pcx_cm_leg.aug.ncm,  " d_new=", pcx_cm_leg.ctx_cm.obj.d)
println("originZC :exclude_row n_eta=", n_eta(layout_excl), " d_new=", pcx_oz_excl.ctx_cm.obj.d)
println("originZC :all_legacy  n_eta=", n_eta(layout_leg),  " d_new=", pcx_oz_leg.ctx_cm.obj.d)
flush(stdout)

# ============================================================================
# PART 1: base dual solves (real KNITRO) -- backend-independent (Architecture C
# inner-solve path is orthogonal to the outer gradient backend), REUSED from
# what :reference needs -- built once per arm/family, shared by both backends.
# ============================================================================
println("\n" * "="^78); println("PART 1: base dual solves (archC_verified_state / archOZ_verified_state)"); println("="^78)
flush(stdout)

base_cm_excl, verify_cm_excl = archC_verified_state(xfc_excl, pcx_cm_excl.ctx_cm, pcx_cm_excl.cctx)
base_cm_leg,  verify_cm_leg  = archC_verified_state(xfc_leg,  pcx_cm_leg.ctx_cm,  pcx_cm_leg.cctx)
base_oz_excl, verify_oz_excl = archOZ_verified_state(xfc_excl, nu0_excl, pcx_oz_excl.ctx_cm)
base_oz_leg,  verify_oz_leg  = archOZ_verified_state(xfc_leg,  nu0_leg,  pcx_oz_leg.ctx_cm)

for (lbl, v) in [("CM       :exclude_row", verify_cm_excl), ("CM       :all_legacy ", verify_cm_leg),
                 ("originZC :exclude_row", verify_oz_excl), ("originZC :all_legacy ", verify_oz_leg)]
    cls = classify_inner_result(v)
    ok = v.inner_status in (0,-100,-101,-103) && is_verified_success(v)
    global ALL_PASS[] &= ok
    @printf("%s  inner_status=%d  Delta_dual=%.10g  primal_dual_gap=%.3g  kkt_resid=%.3g  class=%s  %s\n",
            lbl, v.inner_status, v.Delta_dual, v.primal_dual_gap, v.max_abs_moment_kkt_resid, cls, ok ? "PASS" : "FAIL")
end
flush(stdout)

# ============================================================================
# PART 2: EXACT tier-check on Backend C+'s own incremental path --
# lfix_incremental_at_Cplus! (the function a live :cplus KNITRO cb_G! actually
# calls, via a_block_fd_component_Cplus!) vs the trusted full-rebuild
# fixed_dual_L(_originzc), same methodology as lfix_gradient_layer_validation.jl's
# own Part 2 (which validated :reference's lfix_incremental_at). Requires
# GradWorkspace/LFixFactorizedWorkspace + the CM/originZC-aware C+ cache
# builders (build_lfix_base_cache_cm_C!/build_lfix_base_cache_originzc_C!).
# ============================================================================
println("\n" * "="^78); println("PART 2: EXACT tier-check -- lfix_incremental_at_Cplus! vs fixed_dual_L(_originzc)"); println("="^78)
flush(stdout)

function exact_tier_check_cplus(label, ctx_cm, pe, base, cache_C, gws; n_sample::Int = 24, hs = (0.02, 0.005), nu0 = nothing)
    D = ctx_cm.D; Ddest = ctx_cm.D_dest
    n_free = D * Ddest
    rng = MersenneTwister(1234)
    coords = unique(vcat(1, rand(rng, 2:n_free, min(n_sample, n_free - 1))))
    w0 = vcat(base.x_free0[1], pivot_reduce(log.(reshape(base.x_free0[2:end], D, Ddest)), pe))
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    L_true_fn(xf) = nu0 === nothing ? fixed_dual_L(xf, ctx_cm, base) : fixed_dual_L_originzc(xf, nu0, ctx_cm, base)
    maxdiff = 0.0; n_checks = 0
    for coord in coords, sign in (+1, -1), h in hs
        new_val = w0[coord] + sign * h
        w_true = copy(w0); w_true[coord] = new_val
        L_true = L_true_fn(x_free_from_w(w_true))
        L_Cp = lfix_incremental_at_Cplus!(gws, cache_C, ctx_cm, pe, w0, coord, new_val)
        d = abs(L_Cp - L_true)
        maxdiff = max(maxdiff, d); n_checks += 1
        d > 1e-8 && println("    MISMATCH coord=$coord sign=$sign h=$h L_true=$L_true L_Cp=$L_Cp diff=$d")
    end
    ok = maxdiff < 1e-8
    global ALL_PASS[] &= ok
    println("  $label: n_checks=$n_checks (n_free=$n_free)  max|lfix_incremental_at_Cplus! - fixed_dual_L|=$maxdiff  $(ok ? "PASS" : "FAIL")")
    flush(stdout)
    return nothing
end

nT = Threads.maxthreadid()
gws1 = GradWorkspace(W)   # single-slot scratch, matches a_block_fd_component_Cplus!'s per-call contract

cache_cm_excl_C = build_lfix_base_cache_cm_C!(build_lfix_factorized_workspace(ctx_excl.D, ctx_excl.D_dest, W),
    xfc_excl, pcx_cm_excl.ctx_cm, base_cm_excl, ctx_excl, pcx_cm_excl.aug, pcx_cm_excl.bins)
cache_cm_leg_C = build_lfix_base_cache_cm_C!(build_lfix_factorized_workspace(ctx_leg.D, ctx_leg.D_dest, W),
    xfc_leg, pcx_cm_leg.ctx_cm, base_cm_leg, ctx_leg, pcx_cm_leg.aug, pcx_cm_leg.bins)
cache_oz_excl_C = build_lfix_base_cache_originzc_C!(build_lfix_factorized_workspace(ctx_excl.D, ctx_excl.D_dest, W),
    xfc_excl, pcx_oz_excl.ctx_cm, base_oz_excl, pcx_oz_excl.aug, nu0_excl)
cache_oz_leg_C = build_lfix_base_cache_originzc_C!(build_lfix_factorized_workspace(ctx_leg.D, ctx_leg.D_dest, W),
    xfc_leg, pcx_oz_leg.ctx_cm, base_oz_leg, pcx_oz_leg.aug, nu0_leg)
println("  Backend C+ caches built (CM/originZC-aware, both arms)")
flush(stdout)

exact_tier_check_cplus("CM       :exclude_row (C+)", pcx_cm_excl.ctx_cm, pe_excl, base_cm_excl, cache_cm_excl_C, gws1)
exact_tier_check_cplus("CM       :all_legacy  (C+)", pcx_cm_leg.ctx_cm,  pe_leg,  base_cm_leg,  cache_cm_leg_C,  gws1)
exact_tier_check_cplus("originZC :exclude_row (C+)", pcx_oz_excl.ctx_cm, pe_excl, base_oz_excl, cache_oz_excl_C, gws1; nu0 = nu0_excl)
exact_tier_check_cplus("originZC :all_legacy  (C+)", pcx_oz_leg.ctx_cm,  pe_leg,  base_oz_leg,  cache_oz_leg_C,  gws1; nu0 = nu0_leg)

# ============================================================================
# PART 3: full production analytic gradient, :cplus backend, via the SAME
# entry points a live KNITRO cb_G! calls (cm_production_gradient_cplus /
# cm_originzc_production_gradient_cplus), cross-checked against :reference
# (cm_production_gradient / cm_originzc_production_gradient) to
# machine-precision agreement -- the SAME bar test_cm_meanzc_cplus_equivalence.jl
# / test_cm_originzc_cplus_equivalence.jl already establish at D=4 square, now
# at real D=20/W=80000 scale AND under the genuinely rectangular :exclude_row
# context (never previously exercised for :cplus).
# ============================================================================
println("\n" * "="^78); println("PART 3: full production analytic gradient, :cplus vs :reference (real D=20/W=80000)"); println("="^78)
flush(stdout)

cplus_pool_excl = build_grad_workspace_pool(W)
cplus_ws_excl = build_lfix_factorized_workspace(ctx_excl.D, ctx_excl.D_dest, W)
cplus_pool_leg = build_grad_workspace_pool(W)
cplus_ws_leg = build_lfix_factorized_workspace(ctx_leg.D, ctx_leg.D_dest, W)

t0 = time()
g_cm_excl_ref, meta_cm_excl_ref = cm_production_gradient(xfc_excl, pcx_cm_excl, ctx_excl, pe_excl; base = base_cm_excl)
t_cm_excl_ref = time() - t0
t0 = time()
g_cm_excl_cp, meta_cm_excl_cp = cm_production_gradient_cplus(xfc_excl, pcx_cm_excl, ctx_excl, pe_excl, cplus_pool_excl, cplus_ws_excl; base = base_cm_excl, bandwidth_cache = Dict{Int,Float64}())
t_cm_excl_cp = time() - t0

t0 = time()
g_cm_leg_ref, meta_cm_leg_ref = cm_production_gradient(xfc_leg, pcx_cm_leg, ctx_leg, pe_leg; base = base_cm_leg)
t_cm_leg_ref = time() - t0
t0 = time()
g_cm_leg_cp, meta_cm_leg_cp = cm_production_gradient_cplus(xfc_leg, pcx_cm_leg, ctx_leg, pe_leg, cplus_pool_leg, cplus_ws_leg; base = base_cm_leg, bandwidth_cache = Dict{Int,Float64}())
t_cm_leg_cp = time() - t0

t0 = time()
g_oz_excl_ref, meta_oz_excl_ref = cm_originzc_production_gradient(xfc_excl, nu0_excl, pcx_oz_excl, ctx_excl, pe_excl; base = base_oz_excl, verify = verify_oz_excl)
t_oz_excl_ref = time() - t0
t0 = time()
g_oz_excl_cp, meta_oz_excl_cp = cm_originzc_production_gradient_cplus(xfc_excl, nu0_excl, pcx_oz_excl, ctx_excl, pe_excl, cplus_pool_excl, cplus_ws_excl; base = base_oz_excl, verify = verify_oz_excl, bandwidth_cache = Dict{Int,Float64}())
t_oz_excl_cp = time() - t0

t0 = time()
g_oz_leg_ref, meta_oz_leg_ref = cm_originzc_production_gradient(xfc_leg, nu0_leg, pcx_oz_leg, ctx_leg, pe_leg; base = base_oz_leg, verify = verify_oz_leg)
t_oz_leg_ref = time() - t0
t0 = time()
g_oz_leg_cp, meta_oz_leg_cp = cm_originzc_production_gradient_cplus(xfc_leg, nu0_leg, pcx_oz_leg, ctx_leg, pe_leg, cplus_pool_leg, cplus_ws_leg; base = base_oz_leg, verify = verify_oz_leg, bandwidth_cache = Dict{Int,Float64}())
t_oz_leg_cp = time() - t0

function report_cross(label, g_ref, g_cp, t_ref, t_cp)
    d = abs.(g_ref .- g_cp)
    maxabs = maximum(d)
    denom = max(maximum(abs.(g_ref)), 1e-12)
    maxrel = maxabs / denom
    cosang = dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
    sign_mismatches = sum((sign.(g_ref) .!= sign.(g_cp)) .& (abs.(g_ref) .> 1e-8))
    ok = maxabs < 1e-9
    global ALL_PASS[] &= ok
    @printf("%s  len=%d  max|diff|=%.3e  max_rel=%.3e  cosine=%.12f  sign_mismatches=%d  t_ref=%.2fs  t_cplus=%.2fs  speedup=%.2fx  %s\n",
            label, length(g_ref), maxabs, maxrel, cosang, sign_mismatches, t_ref, t_cp, t_ref / t_cp, ok ? "PASS" : "FAIL")
    flush(stdout)
end
using LinearAlgebra: dot, norm
report_cross("CM       :exclude_row", g_cm_excl_ref, g_cm_excl_cp, t_cm_excl_ref, t_cm_excl_cp)
report_cross("CM       :all_legacy ", g_cm_leg_ref,  g_cm_leg_cp,  t_cm_leg_ref,  t_cm_leg_cp)
report_cross("originZC :exclude_row", g_oz_excl_ref, g_oz_excl_cp, t_oz_excl_ref, t_oz_excl_cp)
report_cross("originZC :all_legacy ", g_oz_leg_ref,  g_oz_leg_cp,  t_oz_leg_ref,  t_oz_leg_cp)

len_ok = length(g_cm_excl_cp) == ctx_excl.D * ctx_excl.D_dest &&
         length(g_cm_leg_cp)  == ctx_leg.D  * ctx_leg.D_dest  &&
         length(g_oz_excl_cp) == ctx_excl.D * ctx_excl.D_dest + n_eta(layout_excl) &&
         length(g_oz_leg_cp)  == ctx_leg.D  * ctx_leg.D_dest  + n_eta(layout_leg)
global ALL_PASS[] &= len_ok
println("gradient dimensions match D*Ddest(+n_eta) expectation: $len_ok")
flush(stdout)

# ============================================================================
# PART 4: threaded == serial check for :cplus (Backend C+'s own threading
# discipline, matching test_lfix_factorized_workspace.jl's own established
# check, now at real D=20/W=80000 under :exclude_row).
# ============================================================================
println("\n" * "="^78); println("PART 4: :cplus threaded vs serial (must be exact)"); println("="^78)
flush(stdout)

bwc1 = Dict{Int,Float64}(); bwc2 = Dict{Int,Float64}()
g_cm_excl_cp_thr, _ = cm_production_gradient_cplus(xfc_excl, pcx_cm_excl, ctx_excl, pe_excl, cplus_pool_excl, cplus_ws_excl; base = base_cm_excl, threaded = true, bandwidth_cache = bwc1)
g_cm_excl_cp_ser, _ = cm_production_gradient_cplus(xfc_excl, pcx_cm_excl, ctx_excl, pe_excl, cplus_pool_excl, cplus_ws_excl; base = base_cm_excl, threaded = false, bandwidth_cache = bwc2)
d_thr = maximum(abs.(g_cm_excl_cp_thr .- g_cm_excl_cp_ser))
ok_thr = d_thr == 0.0
global ALL_PASS[] &= ok_thr
println("  CM :exclude_row threaded vs serial maxabsdiff=$d_thr (must be exact)  $(ok_thr ? "PASS" : "FAIL")")
flush(stdout)

# ============================================================================
# PART 5: genuine finite-difference check -- :cplus analytic gradient vs
# Delta_FD (central difference from a FULL re-solved inner problem), same
# discipline as lfix_gradient_layer_validation.jl's own Part 4.
# ============================================================================
println("\n" * "="^78); println("PART 5: :cplus analytic gradient vs Delta_FD (real re-solved inner problem), matched-h"); println("="^78)
flush(stdout)

function delta_fd_gradient_cm(coords, w0, pe, ctx_cm, pcx, hs_by_coord; kind::Symbol)
    g = Dict{Int,Float64}()
    for c in coords
        h = hs_by_coord[c]
        wp = copy(w0); wp[c] += h
        wm = copy(w0); wm[c] -= h
        xp = vcat(wp[1], vec(exp.(pivot_expand(wp[2:end], pe))))
        xm = vcat(wm[1], vec(exp.(pivot_expand(wm[2:end], pe))))
        if kind == :cm
            _, vp = archC_verified_state(xp, ctx_cm, pcx.cctx)
            _, vm = archC_verified_state(xm, ctx_cm, pcx.cctx)
        else
            _, vp = archOZ_verified_state(xp, pcx.nu0, ctx_cm)
            _, vm = archOZ_verified_state(xm, pcx.nu0, ctx_cm)
        end
        g[c] = (vp.Delta_dual - vm.Delta_dual) / (2h)
    end
    return g
end

function fd_check(label, w0, pe, ctx_cm, pcx, meta, g_analytic; kind::Symbol, nu0 = nothing, n_ablock::Int = 3, rng_seed::Int = 7)
    rng = MersenneTwister(rng_seed)
    D2 = length(w0)
    a_coords = unique(rand(rng, 2:D2, min(n_ablock, D2 - 1)))
    coords = vcat(1, a_coords)
    hs_by_coord = Dict{Int,Float64}(1 => 1e-4)
    for c in a_coords
        hs_by_coord[c] = meta.h_used[c] > 0 ? meta.h_used[c] : 0.01
    end
    pcx_aug = kind == :oz ? merge(pcx, (nu0 = nu0,)) : pcx
    g_fd = delta_fd_gradient_cm(coords, w0, pe, ctx_cm, pcx_aug, hs_by_coord; kind = kind == :oz ? :oz : :cm)
    println("  $label:")
    ok = true
    for c in coords
        ga = g_analytic[c]; gf = g_fd[c]
        rel = abs(ga - gf) / max(abs(gf), 1e-8)
        same_sign = sign(ga) == sign(gf) || abs(gf) < 1e-6
        @printf("    coord=%-4d h=%.5g  analytic=%.6g  Delta_FD=%.6g  rel_diff=%.3g  same_sign=%s\n", c, hs_by_coord[c], ga, gf, rel, same_sign)
        ok &= same_sign
    end
    global ALL_PASS[] &= ok
    println("  $label sign-agreement: $(ok ? "PASS" : "FAIL")")
    flush(stdout)
    return ok
end

fd_check("CM       :exclude_row (C+)", meta_cm_excl_cp.w0, pe_excl, pcx_cm_excl.ctx_cm, pcx_cm_excl, meta_cm_excl_cp, g_cm_excl_cp; kind = :cm)
fd_check("CM       :all_legacy  (C+)", meta_cm_leg_cp.w0,  pe_leg,  pcx_cm_leg.ctx_cm,  pcx_cm_leg,  meta_cm_leg_cp,  g_cm_leg_cp;  kind = :cm)
fd_check("originZC :exclude_row (C+)", meta_oz_excl_cp.w0, pe_excl, pcx_oz_excl.ctx_cm, pcx_oz_excl, meta_oz_excl_cp, g_oz_excl_cp; kind = :oz, nu0 = nu0_excl)
fd_check("originZC :all_legacy  (C+)", meta_oz_leg_cp.w0,  pe_leg,  pcx_oz_leg.ctx_cm,  pcx_oz_leg,  meta_oz_leg_cp,  g_oz_leg_cp;  kind = :oz, nu0 = nu0_leg)

println("\n" * "="^78)
println(ALL_PASS[] ? "ALL :cplus / :exclude_row VALIDATION CHECKS PASSED" : "SOME CHECKS FAILED -- see MISMATCH/FAIL lines above")
println("="^78)
println("=== :cplus backend under destination_sample=:exclude_row validation COMPLETE ===")
ALL_PASS[] || error("lfix_cplus_exclude_row_validation.jl: one or more checks failed")
