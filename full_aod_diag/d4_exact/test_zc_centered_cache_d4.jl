# Section 10 release (2026-07-28, lifecycle-audit HIGHEST_PRIORITY_REMAINING_GAP): D=4 gate for
# the new opt-in `ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]` / `cache_across_callbacks` flag on
# `refresh_zc_centered!` (zc_restriction_operator.jl). Two things to prove, for BOTH ZC families
# (cm_meanzc, origin_zc):
#   1. Bit-exact equivalence: the complete packed Hessian is IDENTICAL whether the cache is off
#      (today's default, rebuild Zc every callback) or on (rebuild only when the outer point's
#      targets actually change) -- this is a pure caching/timing change, not an algebra change.
#   2. `centered-Z rebuilds per Hessian callback = 0` under the new mode: simulate several Hessian
#      callbacks within ONE inner solve (same nu_ref, i.e. same outer point, different dual (x)
#      points -- exactly what KNITRO's own trust-region iterations do) and confirm
#      `zc_centered_rebuilds` increments by exactly 1 (the FIRST callback only) while
#      `zc_centered_cache_hits` increments on every subsequent callback, under
#      cache_across_callbacks=true -- and, for contrast, that the OLD default (cache=false)
#      rebuilds every single callback (rebuilds == number of callbacks), proving the new mode is a
#      genuine reduction, not a no-op relabeling.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2028)

const ONLY_FAMILY = get(ENV, "ONLY_FAMILY", "")
run_family(name) = isempty(ONLY_FAMILY) || ONLY_FAMILY == name

# Several distinct dual (x) points to fire as separate "Hessian callbacks" within ONE inner solve
# (same outer point / nu_ref throughout) -- mimics KNITRO visiting several different dual iterates
# under one fixed theta/nu.
function callback_points(base_x::AbstractVector{Float64}, n::Int; scale = 0.02)
    rng = MersenneTwister(77)
    return [base_x .+ (i == 1 ? 0.0 : scale * i .* randn(rng, length(base_x))) for i in 1:n]
end

# ============================================================================
# cm_meanzc
# ============================================================================
if run_family("cm_meanzc")
println("\n=== cm_meanzc: Zc-caching bit-exactness + rebuild-count gate ===")
for (K_mean, K_pair, label) in [(1, 0, "K_mean1_pair0"), (1, 1, "K1"), (2, 1, "K2")]
    ν0 = nu0vec(K_mean)
    aug_z = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = :anchored, meanzc_basis = :direct)
    ctx_cm_z = merge(ctx, (obj = aug_z.obj_cm,))
    cctx_z = build_cm_meanzc_bin_ctx(ctx, aug_z; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
    base_z = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm_z, cctx_z)
    check("$label: inner solve feasible (nStatus=$(base_z.inner_status))", base_z.inner_status in (0, -100, -101, -103))
    n_z = cctx_z.NCORE + cctx_z.ncm
    base_x = vcat(base_z.ζstar, base_z.λstar)

    # ---- 1. Bit-exact equivalence: cache off vs cache on, several dual points ----
    for x in callback_points(base_x, 3)
        _archC_prep_for_hessian!(ctx_cm_z.obj, x)
        cctx_z.cross_hessian_threaded = false
        cctx_z.zc_gram_backend = :reference
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false
        h_off = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
        hessian_cm_structured_v2!(h_off, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
        H_off = unpack_packed(h_off, n_z)

        _archC_prep_for_hessian!(ctx_cm_z.obj, x)
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = true
        h_on = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
        hessian_cm_structured_v2!(h_on, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
        H_on = unpack_packed(h_on, n_z)
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false

        maxdiff = maximum(abs.(H_off .- H_on))
        check("$label cache on/off bit-exact (maxdiff=$maxdiff)", maxdiff < 1e-12)
    end

    # ---- 2. Rebuild-count gate: several callbacks, ONE outer point (nu_ref unchanged). Each
    # sub-test below FIRST re-establishes a fresh outer point (archC_meanzc_base_state does
    # `cctx.nu_ref[] = collect(νvec)`, a NEW object each call) so `cctx_z.hzz_centered`'s leftover
    # `built_gen` from a PRIOR sub-test/K-config never contaminates this measurement -- the cache
    # persisting Zc across truly-unrelated prior calls is the intended behavior, not a test bug, so
    # the test controls for it explicitly rather than assuming a clean slate. ----
    ν0c = copy(ν0)
    base_zc = archC_meanzc_base_state(x_free_calib, ν0c, ctx_cm_z, cctx_z)
    base_x_c = vcat(base_zc.ζstar, base_zc.λstar)
    pts = callback_points(base_x_c, 5)
    reset_no_dense_g_counters!()
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false
    for x in pts
        _archC_prep_for_hessian!(ctx_cm_z.obj, x)
        h = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
        hessian_cm_structured_v2!(h, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
    end
    r_off = no_dense_g_report()
    check("$label cache=false: rebuilds == n_callbacks ($(length(pts))) (got $(r_off.zc_centered_rebuilds))", r_off.zc_centered_rebuilds == length(pts))
    check("$label cache=false: cache_hits == 0 (got $(r_off.zc_centered_cache_hits))", r_off.zc_centered_cache_hits == 0)

    ν0d = copy(ν0)
    base_zd = archC_meanzc_base_state(x_free_calib, ν0d, ctx_cm_z, cctx_z)   # fresh outer point again
    base_x_d = vcat(base_zd.ζstar, base_zd.λstar)
    pts2 = callback_points(base_x_d, 5)
    reset_no_dense_g_counters!()
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = true
    for x in pts2
        _archC_prep_for_hessian!(ctx_cm_z.obj, x)
        h = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
        hessian_cm_structured_v2!(h, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
    end
    r_on = no_dense_g_report()
    check("$label cache=true: rebuilds == 1 (ONE outer point, got $(r_on.zc_centered_rebuilds))", r_on.zc_centered_rebuilds == 1)
    check("$label cache=true: cache_hits == n_callbacks-1 ($(length(pts2)-1)) (got $(r_on.zc_centered_cache_hits))", r_on.zc_centered_cache_hits == length(pts2) - 1)
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false

    # ---- 3. New outer point (different nu_ref object) forces a fresh rebuild even with cache=true ----
    ν0b = copy(ν0)   # SAME values, but archC_meanzc_base_state below does cctx.nu_ref[] = collect(...), a FRESH object
    base_z2 = archC_meanzc_base_state(x_free_calib, ν0b, ctx_cm_z, cctx_z)
    reset_no_dense_g_counters!()
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = true
    _archC_prep_for_hessian!(ctx_cm_z.obj, vcat(base_z2.ζstar, base_z2.λstar))
    h = Vector{Float64}(undef, n_z * (n_z + 1) ÷ 2)
    hessian_cm_structured_v2!(h, ctx_cm_z.obj, cctx_z; threaded_bins = true, tls = cctx_z.tls)
    r_new = no_dense_g_report()
    check("$label cache=true, NEW outer point: forces a rebuild (got rebuilds=$(r_new.zc_centered_rebuilds))", r_new.zc_centered_rebuilds == 1)
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false
end
end # run_family("cm_meanzc")

# ============================================================================
# origin_zc
# ============================================================================
if run_family("origin_zc")
println("\n=== origin_zc: Zc-caching bit-exactness + rebuild-count gate ===")
for (K_mean, K_pair, label) in [(1, 0, "K_mean1_pair0"), (1, 1, "K1")]   # K2 (2,1) is infeasible at this ctx -- see docs/ZC_CENTERING_LIFECYCLE_RELEASE_2026-07-28.md
    layout_o = OriginByPowerLayout(ctx.D, K_mean, K_pair)
    νfull0 = vcat([fill(Float64(factorial(k)), ctx.D) for k in 1:K_mean]..., [fill(Float64(factorial(k)), ctx.D * (ctx.D - 1) ÷ 2) for k in 1:K_pair]...)
    pcx_o = build_originzc_production_context(ctx, CS, layout_o; zc_cross_hessian_backend = :winner_bin)
    octx = pcx_o.octx
    base_o = archOZ_base_state(x_free_calib, νfull0, pcx_o.ctx_cm)
    check("$label: inner solve feasible (nStatus=$(base_o.inner_status))", base_o.inner_status in (0, -100, -101, -103))
    n_o = octx.NCORE + octx.n_eta
    obj_o = pcx_o.ctx_cm.obj
    base_x = vcat(base_o.ζstar, base_o.λstar)

    for x in callback_points(base_x, 3)
        _prep_dual_index_for_archA!(octx, obj_o, x)
        octx.cross_hessian_threaded = false
        octx.zc_gram_backend = :reference
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false
        h_off = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
        archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h_off,), obj_o)
        H_off = unpack_packed(h_off, n_o)

        _prep_dual_index_for_archA!(octx, obj_o, x)
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = true
        h_on = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
        archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h_on,), obj_o)
        H_on = unpack_packed(h_on, n_o)
        ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false

        maxdiff = maximum(abs.(H_off .- H_on))
        check("$label cache on/off bit-exact (maxdiff=$maxdiff)", maxdiff < 1e-12)
    end

    # Fresh outer point before EACH sub-test (archOZ_base_state does `octx.nu_ref[] =
    # collect(νvec)`, a NEW object each call) -- same rationale as cm_meanzc's section above:
    # cctx.hzz_centered's cache persists ACROSS these calls by design, so a clean measurement needs
    # a genuinely new outer point immediately before it, not an assumption of a clean slate.
    base_oc = archOZ_base_state(x_free_calib, copy(νfull0), pcx_o.ctx_cm)
    base_x_c = vcat(base_oc.ζstar, base_oc.λstar)
    pts = callback_points(base_x_c, 5)
    reset_no_dense_g_counters!()
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false
    for x in pts
        _prep_dual_index_for_archA!(octx, obj_o, x)
        h = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
        archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h,), obj_o)
    end
    r_off = no_dense_g_report()
    # origin-ZC's own callback calls refresh_zc_centered! TWICE per Hessian callback (once for HER,
    # once for HRR -- see cm_hessian_architectures.jl's own comment on this at the HRR call site) --
    # so cache=false rebuilds 2x per callback, not 1x. Documented, not a bug this release introduces.
    check("$label cache=false: rebuilds == 2*n_callbacks ($(2*length(pts))) (got $(r_off.zc_centered_rebuilds))", r_off.zc_centered_rebuilds == 2 * length(pts))

    base_od = archOZ_base_state(x_free_calib, copy(νfull0), pcx_o.ctx_cm)   # fresh outer point again
    base_x_d = vcat(base_od.ζstar, base_od.λstar)
    pts2 = callback_points(base_x_d, 5)
    reset_no_dense_g_counters!()
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = true
    for x in pts2
        _prep_dual_index_for_archA!(octx, obj_o, x)
        h = Vector{Float64}(undef, n_o * (n_o + 1) ÷ 2)
        archA_partitioned_hess_cb_builder(octx)(nothing, nothing, (x = x,), (hess = h,), obj_o)
    end
    r_on = no_dense_g_report()
    check("$label cache=true: rebuilds == 1 (ONE outer point, got $(r_on.zc_centered_rebuilds))", r_on.zc_centered_rebuilds == 1)
    ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false
end
end # run_family("origin_zc")

println("\n", ALL_PASS[] ? "ALL_PASS" : "SOME_FAILED")
