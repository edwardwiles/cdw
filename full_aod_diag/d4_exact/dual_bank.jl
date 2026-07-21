# ============================================================================
# Successful-dual / KKT-scored small warm-start bank.
#
# Ported from diag/fullA-d20-warmstart-replay's replay_lib.jl (cheap_score,
# start_last_accepted, start_nearest_coord, start_best_of_cache) and its
# offline "Policy P3" (run_replay_full.jl: candidates {actual, last-accepted,
# nearest-successful-by-scaled-distance, neutral}, pick lowest KKT-proxy),
# which beat production's single-warm-slot baseline by 29.5% wall time on one
# live delta=5 trajectory (6.3% on a second, shorter one -- real but
# trajectory-dependent, see that worktree's handoff doc secs 12/18).
#
# WHY this exists: production's `ctx.obj.x` is a single warm-start slot,
# overwritten on every success and NaN-poisoned by `obj.x .= NaN` after an
# organic KNITRO -300 (oracle.jl/oracle_fast.jl's `!warm` cold-start reset,
# and inner_loop_initial_values's own `use_cached_x && norm(x)<1e6` guard
# falling back to zeros when NaN). A single -300 therefore forces the NEXT
# point to a neutral/cold start even if a perfectly good nearby successful
# dual is sitting one outer-iterate back. This bank keeps a small
# recency-bounded history of ONLY successfully-solved duals (never a -300 or
# screen-rejected point), so the next point can pick the best-scoring
# candidate instead of being forced to neutral.
#
# `cheap_score`'s kkt_proxy is NOT bit-identical to production's post-solve
# `max_abs_moment_kkt_resid` (that is computed from the SOLVED weights via a
# different, cross-checked formula) -- it is a genuinely cheap, pre-solve
# stand-in: the sup-norm of the unconstrained-Lagrangian gradient at the
# CANDIDATE start, computed via the SAME `compressed_cc_value_grad` the inner
# KNITRO objective/gradient callback itself evaluates, with NO KNITRO call.
# ============================================================================

"""
    DualBank(maxsize=8)

Recency-bounded bank of successfully-solved dual vectors, indexed by the
outer reduced coordinate (`zfree`) they were solved at. Never stores a -300
or screen-rejected result -- callers must only call `record_success!` after
confirming `inner_status in FEASIBLE_CODES`.
"""
mutable struct DualBank
    history::Vector{NamedTuple}   # (eval_id::Int, zfree::Vector{Float64}, x_solved::Vector{Float64})
    maxsize::Int
end
DualBank(maxsize::Int = 8) = DualBank(NamedTuple[], maxsize)

function record_success!(bank::DualBank, eval_id::Int, zfree::AbstractVector{Float64}, x_solved::AbstractVector{Float64})
    push!(bank.history, (eval_id = eval_id, zfree = collect(zfree), x_solved = collect(x_solved)))
    length(bank.history) > bank.maxsize && popfirst!(bank.history)
    return bank
end

"cheap_score: sup-norm Lagrangian-gradient KKT proxy at a candidate dual x0, no KNITRO call."
function cheap_score(obj, cf::CompressedFactual, x0::AbstractVector{Float64})
    ζ0 = x0[1]; λ0 = @view x0[2:end]
    _, g_ζ, g_λ, _, _ = compressed_cc_value_grad(ζ0, λ0, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    return max(abs(g_ζ), isempty(g_λ) ? 0.0 : maximum(abs, g_λ))
end

"""
    select_warm_start(bank, obj, cf, zfree_target) -> (x0::Vector{Float64}, label::Symbol)

Policy P3: score candidates {current production last-successful slot
(`obj.x`, if finite and not NaN-poisoned -- same guard as
`inner_loop_initial_values`), bank's last-accepted, bank's nearest-by-scaled-
reduced-coordinate-distance, neutral (zeros)} via `cheap_score`, return the
lowest-KKT-proxy candidate. Never returns a candidate that isn't either a
genuinely-solved prior dual or the neutral start.
"""
function select_warm_start(bank::DualBank, obj, cf::CompressedFactual, zfree_target::AbstractVector{Float64})
    cands = Tuple{Float64,Vector{Float64},Symbol}[]

    x_actual = (all(isfinite, obj.x) && norm(obj.x) < 1e6) ? obj.x : nothing
    x_actual !== nothing && push!(cands, (cheap_score(obj, cf, x_actual), collect(x_actual), :actual))

    if !isempty(bank.history)
        x_last = bank.history[end].x_solved
        push!(cands, (cheap_score(obj, cf, x_last), x_last, :last_accepted))

        n = length(zfree_target)
        if length(bank.history) >= 3
            M = reduce(hcat, (h.zfree for h in bank.history))'
            scl = vec(std(M, dims = 1)); scl .= max.(scl, 1e-8)
        else
            scl = ones(n)
        end
        best_d = Inf; best_x = nothing
        for h in bank.history
            d = norm((h.zfree .- zfree_target) ./ scl)
            if d < best_d
                best_d = d; best_x = h.x_solved
            end
        end
        best_x !== nothing && push!(cands, (cheap_score(obj, cf, best_x), best_x, :nearest))
    end

    x_neutral = zeros(obj.outer_constr_index)
    push!(cands, (cheap_score(obj, cf, x_neutral), x_neutral, :neutral))

    best_i = argmin([c[1] for c in cands])
    return cands[best_i][2], cands[best_i][3]
end
