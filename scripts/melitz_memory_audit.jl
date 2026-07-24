# Continuation session (2026-07-23, "make the optimized architecture scalable in memory and
# D") Section 2: exact byte-level memory audit of the CURRENT full-matrix parallel gradient
# (:B_localized_parallel) vs. the NEW argument-localized backend, at D=4 (live), and
# projected to D=10/D=20 at W=80,000 by the same formulas (both backends' buffer shapes are
# closed-form in D/K/W/n/nt, so no live D=10/D=20 construction is needed for THIS audit --
# Section 9 separately diagnoses why a live D=20 fixture itself is slow to build).
#
# Usage: JULIA_NUM_THREADS=<n> julia --project=. scripts/melitz_memory_audit.jl

using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

fmt_bytes(b) = b < 1024^2 ? @sprintf("%.2f KB", b / 1024) :
               b < 1024^3 ? @sprintf("%.2f MB", b / 1024^2) :
               @sprintf("%.2f GB", b / 1024^3)

function audit_dims(D, W, nt)
    K = D^2 + 1
    n = 2 * D^2 - 2
    Gbase_bytes = W * K * 8
    per_thread_buf_bytes = 2 * Gbase_bytes   # Gp + Gm, each (W,K)
    total_thread_scratch_bytes = nt * per_thread_buf_bytes
    copyto_calls_per_gradient = 2 * n
    total_copy_bytes_per_gradient = copyto_calls_per_gradient * Gbase_bytes
    return (; D, K, n, W, nt, Gbase_bytes, per_thread_buf_bytes, total_thread_scratch_bytes,
        copyto_calls_per_gradient, total_copy_bytes_per_gradient)
end

function audit_dims_argument_localized(D, W, nt, maxcols)
    K = D^2 + 1
    n = 2 * D^2 - 2
    per_thread_buf_bytes = 2 * W * maxcols * 8   # Gp_local + Gm_local, each (W, maxcols)
    total_thread_scratch_bytes = nt * per_thread_buf_bytes
    zero_fill_bytes_per_gradient = W * K * n * 8   # single fill!(G_jac, 0.0) -- OUTPUT array, memset only
    return (; D, K, n, W, nt, maxcols, per_thread_buf_bytes, total_thread_scratch_bytes,
        zero_fill_bytes_per_gradient)
end

function audit_dims_direct(D, W, nt, maxcols)
    K = D^2 + 1
    n = 2 * D^2 - 2
    # Continuation4, Section 4/7: direct_gradient.jl's per-thread scratch -- arg0_base is
    # SHARED (built once, O(W), not per-thread), Gp/Gm are (W,maxcols) per thread (same sizing
    # as the argument-localized backend's own probe buffers), plus 5 more O(W) vectors per
    # thread (linkp, linkm, profit, u_plus, u_minus, psi_buf -- 6 total). NO jac_h tensor is
    # ever allocated, read, or zeroed -- the only "K"-scale reads are single-column views
    # (`obj.H[:, 2+gcol]`), not a materialized (W,K) or (W,K,n) array.
    shared_arg0_bytes = W * 8
    per_thread_buf_bytes = 2 * W * maxcols * 8 + 6 * W * 8   # Gp+Gm (W,maxcols) + 6x O(W) vectors
    total_thread_scratch_bytes = nt * per_thread_buf_bytes
    output_bytes = n * 8   # the ONLY per-gradient persistent output: a length-n vector
    jac_h_tensor_bytes = 0   # never allocated for this backend (needs_outer_moment_jacobian=false)
    return (; D, K, n, W, nt, maxcols, shared_arg0_bytes, per_thread_buf_bytes,
        total_thread_scratch_bytes, output_bytes, jac_h_tensor_bytes)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== LIVE D=4 measurement ===")
    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
    obj, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt)
    ctx = obj.γ
    nt_live = Threads.maxthreadid()
    compact = melitz_compact_columns_map(ctx)
    maxcols_live = maximum(length(c.direct_cols) for c in compact)
    mincols_live = minimum(length(c.direct_cols) for c in compact)
    println("Threads.nthreads()=", Threads.nthreads(), "  Threads.maxthreadid()=", nt_live)
    @printf("D=4: K=%d moment columns, n=%d free coordinates, W=%d\n", ctx.moment_layout.num_moments,
        length(theta0), size(obj.U, 1))
    @printf("compact-column count per coordinate: min=%d max=%d (vs K=%d full columns)\n",
        mincols_live, maxcols_live, ctx.moment_layout.num_moments)

    println()
    println("=== :B_localized_parallel (current full-matrix backend) ===")
    for (D, W, nt) in [(4, 20_000, nt_live), (4, 20_000, 16), (10, 80_000, 16), (20, 80_000, 16)]
        a = audit_dims(D, W, nt)
        @printf("D=%2d W=%6d nt=%2d | K=%4d n=%4d | Gbase=%s | per-thread(Gp+Gm)=%s | total thread scratch=%s | copyto! calls/grad=%d | total copy bytes/grad=%s\n",
            a.D, a.W, a.nt, a.K, a.n, fmt_bytes(a.Gbase_bytes), fmt_bytes(a.per_thread_buf_bytes),
            fmt_bytes(a.total_thread_scratch_bytes), a.copyto_calls_per_gradient,
            fmt_bytes(a.total_copy_bytes_per_gradient))
    end

    println()
    println("=== :B_argument_localized_parallel (new backend) ===")
    # maxcols at D=4 measured live above; for D=10/D=20 projected as D+4 (dependency-map's
    # own bound: |dep.cells| <= 4 direct cells + D origin-j destinations when touches_link,
    # the worst case -- see argument_localized_gradient.jl's header) -- a conservative
    # (slightly high) projection, not a live measurement, since building a full D=10/D=20
    # ctx is what Section 9 is diagnosing as currently slow.
    for (D, W, nt, maxcols) in [(4, 20_000, nt_live, maxcols_live), (4, 20_000, 16, maxcols_live),
                                  (10, 80_000, 16, 10 + 4), (20, 80_000, 16, 20 + 4)]
        a = audit_dims_argument_localized(D, W, nt, maxcols)
        @printf("D=%2d W=%6d nt=%2d maxcols=%3d | K=%4d n=%4d | per-thread(Gp+Gm local)=%s | total thread scratch=%s | zero-fill bytes/grad (memset, not economics)=%s\n",
            a.D, a.W, a.nt, a.maxcols, a.K, a.n, fmt_bytes(a.per_thread_buf_bytes),
            fmt_bytes(a.total_thread_scratch_bytes), fmt_bytes(a.zero_fill_bytes_per_gradient))
    end

    println()
    println("=== Ratio: full-matrix total copy bytes/grad vs argument-localized total thread scratch (standing) ===")
    for (D, W, nt, maxcols) in [(4, 20_000, 16, maxcols_live), (10, 80_000, 16, 14), (20, 80_000, 16, 24)]
        full = audit_dims(D, W, nt)
        argl = audit_dims_argument_localized(D, W, nt, maxcols)
        @printf("D=%2d: full-matrix copy traffic/grad=%s  vs  argument-localized standing scratch=%s  (reduction in per-gradient copy traffic: %.1fx)\n",
            D, fmt_bytes(full.total_copy_bytes_per_gradient), fmt_bytes(argl.total_thread_scratch_bytes),
            full.total_copy_bytes_per_gradient / argl.total_thread_scratch_bytes)
    end

    println()
    println("=== :B_direct_argument_parallel (continuation4 NEW backend): no jac_h tensor at all ===")
    for (D, W, nt, maxcols) in [(4, 20_000, nt_live, maxcols_live), (4, 20_000, 16, maxcols_live),
                                  (10, 80_000, 16, 10 + 4), (20, 80_000, 16, 20 + 4)]
        d = audit_dims_direct(D, W, nt, maxcols)
        argl = audit_dims_argument_localized(D, W, nt, maxcols)
        @printf("D=%2d W=%6d nt=%2d maxcols=%3d | per-thread scratch=%s | total thread scratch=%s | output=%s | jac_h tensor=NEVER ALLOCATED (argument-localized backend needs %s)\n",
            d.D, d.W, d.nt, d.maxcols, fmt_bytes(d.per_thread_buf_bytes), fmt_bytes(d.total_thread_scratch_bytes),
            fmt_bytes(d.output_bytes), fmt_bytes(argl.zero_fill_bytes_per_gradient))
    end

    println()
    println("=== Live @allocated check, D=4/W=20,000: direct backend allocates O(W), not O(W*K*n) ===")
    obj_impl = build_melitz_implicit_bundle(ctx, obj.U, theta0; delta=1.0, find_smallest=true,
        gradient_backend=:B_direct_argument_serial, h=1e-4, inner_loop_opt=inner_opt,
        outer_loop_opt=joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))
    @printf("obj_impl.jac_h size = %s (needs_outer_moment_jacobian=%s)\n",
        size(obj_impl.jac_h), obj_impl.needs_outer_moment_jacobian)
    obj_impl.use_cached_x = false; obj_impl.x .= NaN
    _, x_live, nStatus_live = CounterfactualSensitivity.inner_loop_internal(obj_impl, theta0)
    @printf("live inner solve nStatus=%d\n", nStatus_live)
    direct_fn = make_melitz_gradient_delta_direct_serial(1e-4)
    g_live = zeros(length(theta0))
    direct_fn(g_live, theta0, ctx, obj_impl, x_live)   # warm-up (JIT + buffer allocation)
    bytes_warm = @allocated direct_fn(g_live, theta0, ctx, obj_impl, x_live)
    @printf("post-warm-up @allocated for one full direct-backend gradient call: %s (expected ~0, all scratch reused)\n",
        fmt_bytes(bytes_warm))
end
