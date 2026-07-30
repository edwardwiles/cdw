# Gate 3A (2026-07-29 reduced-q validation session): threaded D20 direction construction.
#
# `melitz_reduced_q_propose_direction`'s own coordinatewise sweep (`reduced_q_subspace.jl`,
# `for m in 1:nq ... melitz_q_coordinate_probe(...)`) took ~105s at real D=20/W=80,000 (398
# coordinates, Phase 13 of `docs/melitz_reduced_q_subspace_search_2026-07-29.md`) -- the
# dominant cost of reduced-q direction construction at D20 scale. `melitz_q_coordinate_probe`
# MUTATES `obj.op` in place (`melitz_update_operator_at_theta!`) every call, so a shared `obj`
# CANNOT be probed concurrently across threads without a race on `obj.op`'s own buffers. This
# file threads the sweep the SAME way every other threaded gradient backend in this codebase
# does (`localized_gradient.jl`'s `make_melitz_moments_jacobian_b_localized_parallel`,
# attributed, not reinvented): `Threads.@threads :static`, one independent, PRE-BUILT
# `MelitzCCBundle` per thread (bounded per-thread workspace, Gate 3A's own requirement), each
# thread writing only its own disjoint set of `g_q[m]` entries (no shared mutable state
# touched by more than one thread, no crossing-counter or cache race).
#
# `melitz_q_coordinate_probe` itself (`q_bandwidth_policy.jl`) is REUSED UNMODIFIED -- Rule 10,
# "no second, script-only implementation" -- every thread just calls it on its OWN bundle.

using LinearAlgebra: BLAS, norm

"""
    melitz_build_thread_bundle_pool(bundle_factory::Function, n::Integer=Threads.maxthreadid()) -> Vector

Builds `n` independent bundles by calling `bundle_factory()` `n` times (each call must return
a fresh, independently-mutable `MelitzCCBundle` describing the IDENTICAL economic point --
deterministic given identical construction arguments, verified by the caller/tests, not by
this function). Intended to be built ONCE per session/anchor and reused across every
subsequent threaded direction construction at that anchor (rebuilding per call would dominate
wall time at D20 scale and defeats the point of threading).

Sized by `Threads.maxthreadid()` by default, NOT `Threads.nthreads()` -- see
[`melitz_q_coordinatewise_sweep_threaded!`](@ref)'s own docstring for why.
"""
function melitz_build_thread_bundle_pool(bundle_factory::Function, n::Integer=Threads.maxthreadid())
    n >= 1 || throw(ArgumentError("melitz_build_thread_bundle_pool: n must be >= 1, got $n"))
    return [bundle_factory() for _ in 1:n]
end

"""
    melitz_q_coordinatewise_sweep_threaded!(g_q, theta0, nq, policy, bundles, ctx; x0) -> g_q

Fills `g_q[m]` for `m in 1:nq` via `melitz_q_coordinate_probe(theta0, m, policy, bundles[tid],
ctx; x0=x0, mode=:fixed_dual)`, `tid = Threads.threadid()`, under `Threads.@threads :static`
(guarantees a stable, contiguous thread-to-iteration-range assignment so `bundles[tid]` is
never touched by two threads at once -- the SAME scheduling discipline
`make_melitz_moments_jacobian_b_localized_parallel` already uses in this codebase).

`length(bundles) >= Threads.maxthreadid()` is required -- checked, not assumed. `Threads.threadid()`
ranges over `1:Threads.maxthreadid()`, NOT `1:Threads.nthreads()`: Julia's `:default`/
`:interactive` threadpool split (1.9+) means the global thread id a task lands on can exceed
the `:default`-pool count `Threads.nthreads()` returns -- sizing/checking against
`Threads.nthreads()` alone throws a live `BoundsError` the moment a task is scheduled onto a
thread outside that range (confirmed live this session: `nthreads()=20` but a task landed on
`threadid()=21`, exactly the failure mode already documented, and attributed, in
`localized_gradient.jl`'s own `make_melitz_moments_jacobian_b_localized_parallel` -- the SAME
established fix, `Threads.maxthreadid()`, is reused here, not rediscovered independently).
"""
function melitz_q_coordinatewise_sweep_threaded!(g_q::AbstractVector{Float64}, theta0::AbstractVector,
                                                   nq::Integer, policy::MelitzQBandwidthPolicy,
                                                   bundles::AbstractVector, ctx; x0::AbstractVector)
    length(g_q) == nq || throw(ArgumentError("melitz_q_coordinatewise_sweep_threaded!: length(g_q)=$(length(g_q)) != nq=$nq"))
    nt = Threads.maxthreadid()
    length(bundles) >= nt || throw(ArgumentError(
        "melitz_q_coordinatewise_sweep_threaded!: need >= $nt bundles (Threads.maxthreadid()), got $(length(bundles))"))
    prev_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads :static for m in 1:nq
            tid = Threads.threadid()
            r = melitz_q_coordinate_probe(theta0, m, policy, bundles[tid], ctx; x0=x0, mode=:fixed_dual)
            g_q[m] = r.secant
        end
    finally
        BLAS.set_num_threads(prev_blas)
    end
    return g_q
end

"""
    melitz_reduced_q_propose_direction_threaded(theta_anchor, x0, ctx, bundles;
        bandwidth_policy, var_scale_q=nothing, sign_check_target=10) -> (d, g_q)

Threaded counterpart to `melitz_reduced_q_propose_direction` (`reduced_q_subspace.jl`) --
IDENTICAL algorithm (steepest-descent proposal from the coordinatewise secant vector, unit
normalization, sign check via one direct block secant at a small crossing-calibrated probe),
the ONLY difference being the coordinatewise sweep itself runs threaded
(`melitz_q_coordinatewise_sweep_threaded!`) instead of a serial `for` loop.
`bundles[1]` is used for every single-call step downstream of the sweep (the sign-check probe,
`melitz_bisect_amplitude_for_target_crossings`, `melitz_q_direct_block_secant`) -- these are
single `O(W)` calls, not threaded (matching this codebase's own disclosed convention that only
genuinely `Threads.@threads`-eligible loops get threaded, not every downstream single call).
"""
function melitz_reduced_q_propose_direction_threaded(theta_anchor::AbstractVector, x0::AbstractVector, ctx,
                                                       bundles::AbstractVector;
                                                       bandwidth_policy::MelitzQBandwidthPolicy=PowerScaledQBandwidth(1e-3, 80_000, 0.5),
                                                       var_scale_q::Union{Nothing,AbstractVector}=nothing,
                                                       sign_check_target::Integer=10)
    melitz_reduced_q_check_ctx(ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    obj1 = bundles[1]
    sorted_ctx = ctx.sorted_tail_ctx
    g_q = zeros(nq)
    melitz_q_coordinatewise_sweep_threaded!(g_q, theta_anchor, nq, bandwidth_policy, bundles, ctx; x0=x0)

    Sq2inv = var_scale_q === nothing ? ones(nq) : 1.0 ./ (Vector{Float64}(var_scale_q) .^ 2)
    length(Sq2inv) == nq || throw(ArgumentError("var_scale_q must have length nq=$nq"))
    d_tilde = -Sq2inv .* g_q
    nrm = norm(d_tilde)
    (isfinite(nrm) && nrm > 1e-300) || return nothing, g_q
    d = d_tilde ./ nrm

    t_probe = melitz_bisect_amplitude_for_target_crossings(theta_anchor, d, sign_check_target, ctx, sorted_ctx)
    secant_s, _, _ = melitz_q_direct_block_secant(theta_anchor, d, t_probe, obj1, ctx, x0; mode=:fixed_dual)
    (isfinite(secant_s) && abs(secant_s) > 1e-300) || return nothing, g_q
    secant_s > 0 && (d = -d)
    return d, g_q
end
