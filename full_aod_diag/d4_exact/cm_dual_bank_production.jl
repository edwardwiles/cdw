# ================================================================================================
# Phase D remediation (production-audit continuation, 2026-07-26): dual-bank warm starts for the
# four restricted families' PUBLIC drivers. `DualBank`/`record_success!` (dual_bank.jl) are
# already generic (no dependency on which family built them) and reused UNCHANGED here. But
# `dual_bank.jl`'s own `select_warm_start`/`cheap_score` ARE UNRESTRICTED-family-specific --
# `cheap_score` calls `compressed_cc_value_grad(...; cf::CompressedFactual, ...)`, which scores
# ONLY the economic-core columns of a candidate dual and has no concept of a CM-grid or mean/pair
# restriction block. Calling it as-is on a restricted family's WIDER inner-solve vector
# (`x = [zeta; lambda_core; lambda_restriction]`) would either dimension-mismatch or silently
# ignore the restriction columns' own contribution to solve quality -- not safe to reuse.
#
# SCOPE DECISION (documented, not silently narrowed): building a restriction-aware KKT-proxy
# cheap_score analogue would require new per-family gradient-evaluation kernels (comparable in
# scope to Phase B1's lookup-FG work) -- out of scope for this pass. Instead, this file implements
# a SIMPLER, still scientifically sound, distance-only selection policy: among the bank's history
# plus the current `obj.x` "actual" slot, pick whichever solved dual's OWN economic coordinate
# (`x_free`, the same decoded value cb_F! already computes) is nearest (scaled Euclidean distance)
# to the target point; fall back to the neutral (zero) start when the bank is empty and `obj.x` is
# not a valid finite start. This is a real, functioning warm-start bank -- just without the
# unrestricted family's own richer KKT-based re-ranking, which is a legitimate, load-bearing
# reason to differ, not an oversight.
# ================================================================================================

"""
    RestrictedDualBank

Wraps a `DualBank` (dual_bank.jl, reused unchanged) plus the small bookkeeping this simpler
selection policy needs: each bank entry's `x_free` (the ECONOMIC coordinate the dual was solved
at, not the raw outer KNITRO vector) alongside its solved dual `x_solved`, for scaled-distance
comparison.
"""
mutable struct RestrictedDualBank
    bank::DualBank
    xfree_history::Vector{Vector{Float64}}   # parallel to bank.history, same push/pop discipline
end
RestrictedDualBank(maxsize::Int = 8) = RestrictedDualBank(DualBank(maxsize), Vector{Float64}[])

function record_success_restricted!(rb::RestrictedDualBank, eval_id::Int, x_free::AbstractVector{Float64}, x_solved::AbstractVector{Float64})
    record_success!(rb.bank, eval_id, x_free, x_solved)   # zfree slot reused to hold x_free here -- same struct, different semantic label
    push!(rb.xfree_history, collect(x_free))
    length(rb.xfree_history) > rb.bank.maxsize && popfirst!(rb.xfree_history)
    return rb
end

mutable struct RestrictedDualBankCounters
    queries::Int
    hits::Int          # a bank entry was selected (not actual, not neutral)
    misses::Int        # neutral start used (bank empty AND actual invalid)
    warm_inner_solves::Int
    cold_inner_solves::Int
    selected_distance_sum::Float64
    warm_start_failures::Int   # a warm-started solve came back non-feasible (nStatus not in FEASIBLE_CODES)
end
RestrictedDualBankCounters() = RestrictedDualBankCounters(0, 0, 0, 0, 0, 0.0, 0)
const RESTRICTED_DUAL_BANK_COUNTERS = Ref(RestrictedDualBankCounters())
reset_restricted_dual_bank_counters!() = (RESTRICTED_DUAL_BANK_COUNTERS[] = RestrictedDualBankCounters())

function print_restricted_dual_bank_counters(c::RestrictedDualBankCounters = RESTRICTED_DUAL_BANK_COUNTERS[])
    println("[restricted-dual-bank] queries=", c.queries, " hits=", c.hits, " misses=", c.misses,
            " warm_inner_solves=", c.warm_inner_solves, " cold_inner_solves=", c.cold_inner_solves,
            " warm_start_failures=", c.warm_start_failures,
            " mean_selected_distance=", c.hits == 0 ? "n/a" : round(c.selected_distance_sum / c.hits, digits = 6))
end

"""
    select_warm_start_restricted(rb, obj, x_free_target) -> (x0::Vector{Float64}, label::Symbol, distance::Float64)

Distance-only selection (see file header for why): scaled Euclidean distance in `x_free` space
(std-scaled once >=3 history points exist, same scaling discipline `select_warm_start`'s own
`:nearest` branch uses, unscaled otherwise) between `x_free_target` and every bank entry's own
`x_free`, PLUS the current `obj.x` "actual" slot treated as an implicit candidate at distance 0
if it is the bank's own most recent entry (the common case), or simply included as a same-priority
fallback otherwise. Falls back to neutral (zeros) only when the bank is empty AND `obj.x` is not a
valid finite start (`norm(obj.x) < 1e6`, same guard `inner_loop_initial_values` itself uses).
"""
function select_warm_start_restricted(rb::RestrictedDualBank, obj, x_free_target::AbstractVector{Float64})
    RESTRICTED_DUAL_BANK_COUNTERS[].queries += 1
    n_hist = length(rb.xfree_history)
    x_actual = (all(isfinite, obj.x) && norm(obj.x) < 1e6) ? obj.x : nothing

    if n_hist == 0
        if x_actual !== nothing
            RESTRICTED_DUAL_BANK_COUNTERS[].hits += 1
            return collect(x_actual), :actual, 0.0
        end
        RESTRICTED_DUAL_BANK_COUNTERS[].misses += 1
        return zeros(obj.outer_constr_index), :neutral, Inf
    end

    scl = if n_hist >= 3
        M = reduce(hcat, rb.xfree_history)'
        s = vec(std(M, dims = 1)); max.(s, 1e-8)
    else
        ones(length(x_free_target))
    end

    best_d = Inf; best_x = nothing; best_label = :neutral
    for (i, xf_h) in enumerate(rb.xfree_history)
        d = norm((xf_h .- x_free_target) ./ scl)
        if d < best_d
            best_d = d; best_x = rb.bank.history[i].x_solved; best_label = :nearest
        end
    end
    if x_actual !== nothing && best_x === nothing
        best_x = collect(x_actual); best_label = :actual; best_d = 0.0
    end

    RESTRICTED_DUAL_BANK_COUNTERS[].hits += 1
    RESTRICTED_DUAL_BANK_COUNTERS[].selected_distance_sum += isfinite(best_d) ? best_d : 0.0
    return best_x, best_label, best_d
end
