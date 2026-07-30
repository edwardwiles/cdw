# Typed evaluation-classification counters + standards-compliant CSV/Markdown export
# (2026-07-29 reduced-q validation session, Phase 0 report repair).
#
# MOTIVATION: `scripts/melitz_reducedq_phase12_d4_comparison_2026-07-29.jl`'s own hand-rolled
# `join([...], ",")` CSV writer produced a non-standards-compliant CSV (labels
# `production_(A,f)`/`full_experimental_(A,q)` contain a literal comma, unquoted, so a normal
# CSV parser splits those rows into 19 fields against an 18-column header) and the accompanying
# markdown doc's own prose ("the Phase 7 cap screen fired... 91/157/111 screened evaluations")
# misread the `n_numerical_failure` column as `n_screened` (the actual `n_screened` column is
# `0` in every one of the 12 CSV rows -- the cap screen never fired in that smoke test; the
# 91/157/111 values are genuine `NumericalFailure` counts). This file gives every
# classification-count consumer ONE typed struct and ONE pair of standards-compliant
# CSV/Markdown writers so the same class of bug (ad hoc string joins, numbers read from the
# wrong column by eye) cannot silently recur.
#
# Classification model (Rules 2-5 of the governing prompt, restated as a typed invariant):
#   - every trial point that reaches `solve_melitz_delta!` gets EXACTLY ONE of the four typed
#     `MelitzInnerResult` classifications: FiniteSolved / AboveEvaluationCap /
#     InfiniteDeltaCertified / NumericalFailure.
#   - a trial point may instead be SAFELY SCREENED (Phase 7 of
#     `docs/melitz_reduced_q_subspace_search_2026-07-29.md`: a one-sided weak-duality lower
#     bound on DeltaStar, computed from the anchor's own already-verified dual, certifies
#     AboveEvaluationCap WITHOUT calling `solve_melitz_delta!` at all). A screened point is
#     therefore a MEMBER of the AboveEvaluationCap superset (Rule: "screened AboveEvaluationCap
#     points are a subset of total AboveEvaluationCap"), tracked separately because it never
#     received a full inner-solve verification.
#   - a candidate point may be excluded before any evaluation is attempted because it violates
#     a registered affine (linear) constraint -- e.g. Gate 3B's one-sided `s` trust-interval
#     sweep only evaluates `s` values inside the closed-form feasible interval. These are
#     tracked as `n_affine_excluded` and are NOT part of "classified evaluations" at all (they
#     never reached KNITRO/the inner solver).

"""
    MelitzTypedEvalCounters

One typed record for every classification an outer-search trial point can receive. Every
field is a nonnegative `Int`. Construct via [`melitz_typed_counters_from_reduced_q_stage`](@ref),
[`melitz_typed_counters_from_direct_result`](@ref), or the raw keyword constructor; validate
via [`melitz_validate_typed_counters`](@ref) before reporting.
"""
struct MelitzTypedEvalCounters
    n_finite_solved::Int
    n_above_cap_evaluated::Int     # solve_melitz_delta! returned AboveEvaluationCap
    n_screened_above_cap::Int      # certified AboveEvaluationCap via the cheap screen, no solve
    n_infinite_delta::Int
    n_numerical_failure::Int
    n_affine_excluded::Int         # never attempted: outside a registered linear/affine bound
end

function MelitzTypedEvalCounters(; n_finite_solved::Integer=0, n_above_cap_evaluated::Integer=0,
                                  n_screened_above_cap::Integer=0, n_infinite_delta::Integer=0,
                                  n_numerical_failure::Integer=0, n_affine_excluded::Integer=0)
    return MelitzTypedEvalCounters(Int(n_finite_solved), Int(n_above_cap_evaluated),
        Int(n_screened_above_cap), Int(n_infinite_delta), Int(n_numerical_failure),
        Int(n_affine_excluded))
end

"Total AboveEvaluationCap, evaluated + screened (screened is always a subset of this total)."
melitz_total_above_cap(c::MelitzTypedEvalCounters) = c.n_above_cap_evaluated + c.n_screened_above_cap

"Every trial point that received one of the four typed `MelitzInnerResult` classifications, or was screened."
melitz_total_classified(c::MelitzTypedEvalCounters) =
    c.n_finite_solved + melitz_total_above_cap(c) + c.n_infinite_delta + c.n_numerical_failure

"All candidate points considered, including ones excluded before any evaluation was attempted."
melitz_total_candidates(c::MelitzTypedEvalCounters) = melitz_total_classified(c) + c.n_affine_excluded

"""
    melitz_validate_typed_counters(c; n_trials=nothing) -> Bool

Structural invariants (governing prompt Phase 0's own required test properties):
every field is nonnegative; `n_screened_above_cap <= melitz_total_above_cap(c)` (screened is a
subset, trivially true by construction but checked, not assumed); and, if `n_trials` (the
length of an independently-collected trial-record vector) is supplied, that
`melitz_total_classified(c) == n_trials` -- i.e. every classified evaluation contributes to
EXACTLY one typed count, with no double counting and no silent drop. Throws `ArgumentError`
with a specific diagnosis rather than returning `false` silently.
"""
function melitz_validate_typed_counters(c::MelitzTypedEvalCounters; n_trials::Union{Nothing,Integer}=nothing)
    for (name, val) in pairs((n_finite_solved=c.n_finite_solved, n_above_cap_evaluated=c.n_above_cap_evaluated,
                               n_screened_above_cap=c.n_screened_above_cap, n_infinite_delta=c.n_infinite_delta,
                               n_numerical_failure=c.n_numerical_failure, n_affine_excluded=c.n_affine_excluded))
        val < 0 && throw(ArgumentError("MelitzTypedEvalCounters.$name is negative: $val"))
    end
    if c.n_screened_above_cap > melitz_total_above_cap(c)
        throw(ArgumentError("n_screened_above_cap ($(c.n_screened_above_cap)) exceeds " *
            "melitz_total_above_cap ($(melitz_total_above_cap(c))) -- screened must be a subset"))
    end
    if n_trials !== nothing && melitz_total_classified(c) != n_trials
        throw(ArgumentError("melitz_total_classified(c)=$(melitz_total_classified(c)) != " *
            "n_trials=$n_trials -- every classified evaluation must contribute to exactly one " *
            "typed classification count, with no double counting and no silent drop"))
    end
    return true
end

"""
    melitz_typed_counters_from_reduced_q_stage(stage_record) -> MelitzTypedEvalCounters

Builds counters from one `MelitzReducedQStageResult` (or any object exposing the same
`n_finite_solved`/`n_above_cap`/`n_infinite_delta`/`n_numerical_failure`/`n_cap_screened`
fields the reduced-q-subspace controller already tracks, `reduced_q_controller.jl`). The
controller's own `n_above_cap` field counts only FULLY EVALUATED AboveEvaluationCap points
(`kind==:AboveEvaluationCap` from `solve_melitz_delta!`) -- `n_cap_screened` (a genuinely
disjoint kind, `:cap_screened`, that never reaches `solve_melitz_delta!`) is kept in its own
field here, not folded into `n_above_cap_evaluated`.
"""
function melitz_typed_counters_from_reduced_q_stage(stage)
    return MelitzTypedEvalCounters(n_finite_solved=stage.n_finite_solved,
        n_above_cap_evaluated=stage.n_above_cap, n_screened_above_cap=stage.n_cap_screened,
        n_infinite_delta=stage.n_infinite_delta, n_numerical_failure=stage.n_numerical_failure,
        n_affine_excluded=0)
end

"Sums per-stage counters (e.g. across every stage of one `MelitzReducedQSearchResult`)."
function melitz_typed_counters_from_reduced_q_stages(stages)
    isempty(stages) && return MelitzTypedEvalCounters()
    cs = melitz_typed_counters_from_reduced_q_stage.(stages)
    return MelitzTypedEvalCounters(
        n_finite_solved=sum(c.n_finite_solved for c in cs),
        n_above_cap_evaluated=sum(c.n_above_cap_evaluated for c in cs),
        n_screened_above_cap=sum(c.n_screened_above_cap for c in cs),
        n_infinite_delta=sum(c.n_infinite_delta for c in cs),
        n_numerical_failure=sum(c.n_numerical_failure for c in cs),
        n_affine_excluded=sum(c.n_affine_excluded for c in cs))
end

"""
    melitz_typed_counters_from_direct_result(res) -> MelitzTypedEvalCounters

Builds counters from a `solve_melitz_finite_delta_bound` result (production `(A,f)` or full
`(A,q)` backends, `finite_delta_outer.jl`), which has no cap-screen and no affine-exclusion
concept of its own (the divergence constraint is a nonlinear KNITRO row, not a pre-screened
linear one) -- both fields are legitimately `0`, not a bug, for these backends.
"""
function melitz_typed_counters_from_direct_result(res)
    return MelitzTypedEvalCounters(n_finite_solved=res.n_inner_solved,
        n_above_cap_evaluated=res.n_above_cap_reject, n_screened_above_cap=0,
        n_infinite_delta=res.n_infinite_delta_reject, n_numerical_failure=res.n_numerical_failure_reject,
        n_affine_excluded=0)
end

# ---------------------------------------------------------------------------------------------
# Standards-compliant CSV export (RFC 4180 minimal quoting: any field containing the delimiter,
# a double quote, or a newline is wrapped in double quotes, with internal double quotes doubled).
# ---------------------------------------------------------------------------------------------

function melitz_csv_field(x)::String
    s = x isa AbstractFloat ? string(x) : string(x)
    needs_quote = occursin(',', s) || occursin('"', s) || occursin('\n', s) || occursin('\r', s)
    if needs_quote
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

melitz_csv_row(fields) = join(melitz_csv_field.(fields), ",")

"""
    melitz_write_typed_counter_csv(path, header::Vector{String}, rows::Vector{<:NamedTuple})

Writes `rows` (each a `NamedTuple` whose keys match `header`, in order) to `path` as a
standards-compliant, properly-quoted CSV. Every row is validated on the fly if it carries a
`counters::MelitzTypedEvalCounters` field (`melitz_validate_typed_counters` is called with
`n_trials` from the row's own `n_trials` field, if present) -- an inconsistent row throws
before anything is written, rather than silently exporting bad data.
"""
function melitz_write_typed_counter_csv(path::AbstractString, header::Vector{String}, rows)
    for r in rows
        if hasproperty(r, :counters)
            n_trials = hasproperty(r, :n_trials) ? getproperty(r, :n_trials) : nothing
            melitz_validate_typed_counters(r.counters; n_trials=n_trials)
        end
    end
    open(path, "w") do io
        println(io, melitz_csv_row(header))
        for r in rows
            println(io, melitz_csv_row(getproperty(r, Symbol(h)) for h in header))
        end
    end
    return path
end

"""
    melitz_write_typed_counter_markdown(path, title, header, rows; note=nothing)

Writes a Markdown table generated from the SAME `header`/`rows` machine-readable records used
by [`melitz_write_typed_counter_csv`](@ref) -- never a hand-copied second rendering of the same
numbers (the exact failure mode that produced the original Phase 12 doc's CSV/prose count
mismatch).
"""
function melitz_write_typed_counter_markdown(path::AbstractString, title::AbstractString,
                                              header::Vector{String}, rows; note::Union{Nothing,AbstractString}=nothing)
    open(path, "w") do io
        println(io, "# ", title)
        println(io)
        note !== nothing && (println(io, note); println(io))
        println(io, "| ", join(header, " | "), " |")
        println(io, "|", repeat("---|", length(header)))
        for r in rows
            vals = [string(getproperty(r, Symbol(h))) for h in header]
            println(io, "| ", join(vals, " | "), " |")
        end
    end
    return path
end
