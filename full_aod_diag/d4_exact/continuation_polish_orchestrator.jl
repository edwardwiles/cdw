# continuation_polish_orchestrator.jl -- reusable continuation-and-polishing campaign engine.
#
# Built for the FULL (non-REDUCED) W=100,000 continuation/polish campaign, 2026-08-03. Wraps the
# existing production driver entry points (run_polish_checkpointed_unified, run_cm_upper_checkpointed,
# run_cm_lower_checkpointed, run_originzc_upper_checkpointed, run_originzc_lower_checkpointed) --
# does NOT reimplement any solve logic, algorithm, or economic model. See
# docs/audits/fullA-continuation-polish-2026-08-03/ for the manifest validation, algorithm
# inventory, and checkpoint-contents findings this module's design decisions are based on.
#
# Caller is responsible for `include`-ing the real driver chain (c10_d20_production_driver.jl,
# flexible_theta.jl, flexible_theta_aspace_production.jl, outer_coordinate_layout.jl,
# c10_d20_production_driver_unified.jl, and the long cm_*/campaign_cm_family_runner.jl chain for
# CM-family/origin-ZC) BEFORE including this file, exactly as the existing campaign runners do --
# this file does not manage includes itself, to avoid duplicating/drifting from the real chain.

using Dates, SHA, Serialization, LinearAlgebra

# No CSV/DataFrames package is a Project.toml dependency in this repo (see json_lite.jl's own
# header for the same convention/reasoning) -- this reader is scoped to exactly the two simple,
# no-embedded-comma CSV shapes this module consumes (MONOTONE_INCUMBENT_ENVELOPE_*.csv,
# CONTINUATION_SEED_MANIFEST_*.csv), not a general-purpose CSV library.
function read_simple_csv(path::AbstractString)
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && return (String[], Vector{Vector{String}}())
    header = String.(split(lines[1], ','))
    rows = [String.(split(l, ',')) for l in lines[2:end]]
    return (header, rows)
end

function csv_rows_as_namedtuples(path::AbstractString)
    header, rows = read_simple_csv(path)
    syms = Symbol.(header)
    return [NamedTuple{Tuple(syms)}(Tuple(row)) for row in rows]
end

"""
    load_checkpoint_w(path, family) -> Vector{Float64}

Extracts the actual best-VERIFIED outer vector from a checkpoint file -- `checkpoint.best_feasible.w`,
NOT `checkpoint.zfree`/`checkpoint.g` (the last, possibly-unverified probe point). See
CHECKPOINT_CONTENTS_VERIFICATION_2026-08-03.md. Requires the caller to have already `include`-d the
real driver chain (`load_checkpoint_unified`/`load_cm_checkpoint` must already be defined).
"""
function load_checkpoint_w(path::AbstractString, family::String)
    ckpt = family == "unrestricted" ? load_checkpoint_unified(path) : load_cm_checkpoint(path)
    ckpt.best_feasible === nothing &&
        error("load_checkpoint_w($path): checkpoint has no best_feasible point (no verified incumbent found in that cell) -- not usable as a seed.")
    return ckpt.best_feasible.w
end

# ---------------------------------------------------------------------------------------------
# Seed representation and monotone envelope
# ---------------------------------------------------------------------------------------------

"""
    Seed

One candidate starting point for a continuation cell. `w` is the family's own native outer vector
(the exact format `run_*_checkpointed*`'s positional `w0`/`w_start_in` argument expects --
`best_feasible.w` from a checkpoint, NOT `checkpoint.zfree`/`checkpoint.g` -- see
CHECKPOINT_CONTENTS_VERIFICATION_2026-08-03.md).
"""
struct Seed
    family::String
    direction::Symbol              # :upper | :lower
    role::String                   # e.g. "A_envelope_incumbent_delta_0.5"
    source_delta::Float64
    source_start::Int
    GT::Float64
    Delta_star::Float64
    w::Vector{Float64}
    outer_vector_sha256::String
    source_path::String
end

direction_is_upper(d::Symbol) = d === :upper

"""
    better(direction, a, b) -> Bool

True if GT value `a` is at least as good as `b` for this direction (upper: larger GT is better;
lower: smaller GT is better). Used both for envelope maintenance and the "never return worse than
inherited incumbent" rule (task spec Section 4/12).
"""
better(direction::Symbol, a::Real, b::Real) = direction_is_upper(direction) ? (a >= b) : (a <= b)
strictly_better(direction::Symbol, a::Real, b::Real) = direction_is_upper(direction) ? (a > b) : (a < b)

"""
    MonotoneEnvelope

In-memory table of the best independently-verified GT at or below each target delta, per
family/direction. `inherit(...)` implements the feasible-set-nesting rule: F(δ_a) ⊆ F(δ_b) for
δ_a <= δ_b, so the envelope at target δ_b must be at least as good as at any δ_a <= δ_b.
"""
mutable struct MonotoneEnvelope
    rows::Dict{Tuple{String,Symbol,Float64},NamedTuple}  # (family,direction,delta) -> best row
end

MonotoneEnvelope() = MonotoneEnvelope(Dict{Tuple{String,Symbol,Float64},NamedTuple}())

"""
    register!(env, family, direction, delta, GT, w, Delta_star, provenance)

`w` is the actual outer vector at this GT, `Delta_star` its verified divergence budget used
(required, not optional) -- the whole point of the in-memory envelope is that `envelope_at` can
hand back a fully usable incumbent (vector + Delta*) directly without a disk round-trip, including
as the fallback point in `apply_never_regress` when nothing improves on an inherited incumbent.
"""
function register!(env::MonotoneEnvelope, family::String, direction::Symbol, delta::Float64,
                    GT::Real, w::Vector{Float64}, Delta_star::Real, provenance::NamedTuple)
    key = (family, direction, round(delta; digits = 10))
    cur = get(env.rows, key, nothing)
    if cur === nothing || strictly_better(direction, GT, cur.GT)
        env.rows[key] = merge((GT = Float64(GT), w = w, Delta_star = Float64(Delta_star)), provenance)
    end
    return env
end

"""
    envelope_at(env, family, direction, target_delta) -> best (GT, provenance) among all
registered deltas <= target_delta, or `nothing` if none registered yet. This is the inheritance
rule: every verified smaller-delta point is a valid incumbent for every larger delta.
"""
function envelope_at(env::MonotoneEnvelope, family::String, direction::Symbol, target_delta::Real)
    best = nothing
    for ((f, d, delta), row) in env.rows
        f == family && d == direction && delta <= target_delta + 1e-12 || continue
        if best === nothing || strictly_better(direction, row.GT, best.GT)
            best = row
        end
    end
    return best
end

"""
    load_envelope_csv(path) -> MonotoneEnvelope

Loads a MONOTONE_INCUMBENT_ENVELOPE_*.csv-shaped file (family,direction,target_delta,envelope_GT,
source_delta,source_start,outer_vector_path,outer_vector_sha256,inner_dual_path,Delta_star,
slack_at_target,inherited_or_discovered).
"""
function load_envelope_csv(path::AbstractString)
    env = MonotoneEnvelope()
    for r in csv_rows_as_namedtuples(path)
        direction = Symbol(r.direction)
        family = String(r.family)
        w = load_checkpoint_w(String(r.outer_vector_path), family)
        register!(env, family, direction, parse(Float64, r.target_delta), parse(Float64, r.envelope_GT), w,
                  parse(Float64, r.Delta_star),
                  (source_delta = parse(Float64, r.source_delta), source_start = parse(Int, r.source_start),
                   outer_vector_path = String(r.outer_vector_path),
                   outer_vector_sha256 = String(r.outer_vector_sha256)))
    end
    return env
end

function load_seed_manifest_csv(path::AbstractString)
    seeds = Seed[]
    for r in csv_rows_as_namedtuples(path)
        lowercase(String(r.available)) == "available" || continue
        w = load_checkpoint_w(String(r.outer_vector_path), String(r.family))
        push!(seeds, Seed(String(r.family), Symbol(r.direction), String(r.seed_role),
                           parse(Float64, r.delta), parse(Int, r.start), parse(Float64, r.GT),
                           parse(Float64, r.Delta_star), w, String(r.outer_vector_sha256),
                           String(r.outer_vector_path)))
    end
    return seeds
end

# ---------------------------------------------------------------------------------------------
# Seed selection / deduplication (task Section 6: at most 3 seeds, dedup by economic-state
# distance and objective value -- do not run 5 fresh starts, do not privilege start numbering)
# ---------------------------------------------------------------------------------------------

"""
    dedup_seeds(seeds; w_tol=1e-9, gt_tol=1e-9) -> Vector{Seed}

Two seeds are duplicates if their outer vectors are within `w_tol` (Euclidean norm, same family
so same coordinate space) AND their GT values are within `gt_tol`. Keeps the first occurrence
(caller should pass seeds in priority order: envelope incumbents before best-direct-discovered).
"""
function dedup_seeds(seeds::Vector{Seed}; w_tol::Float64 = 1e-9, gt_tol::Float64 = 1e-9)
    kept = Seed[]
    for s in seeds
        is_dup = any(kept) do k
            length(k.w) == length(s.w) &&
                norm(k.w .- s.w) < w_tol * max(1.0, norm(k.w)) &&
                abs(k.GT - s.GT) < gt_tol
        end
        is_dup || push!(kept, s)
    end
    return kept
end

# ---------------------------------------------------------------------------------------------
# Algorithm stage kwargs -- per ALGORITHM_INVENTORY_2026-08-03.md. No new solver code; this only
# selects which already-wired kwarg combination to pass.
# ---------------------------------------------------------------------------------------------

@enum AlgoStage EXPLORE_DIRECT_SR1 POLISH_SQP POLISH_DIRECT_BFGS

"""
    algo_kwargs(family, stage) -> NamedTuple of kwargs to splat into the family's run_*_checkpointed call.

`unrestricted` has no `opt_file` kwarg (ALGORITHM_INVENTORY finding) so POLISH_SQP is not valid for
it -- use POLISH_DIRECT_BFGS instead. Passing POLISH_SQP for `unrestricted` is a programming error,
not a silent fallback (caught explicitly below), matching the "no function may silently substitute
a scientific/methodological default" spirit of this repo's own hardening rule.
"""
function algo_kwargs(family::String, stage::AlgoStage)
    if stage == EXPLORE_DIRECT_SR1
        return (outer_direct_hessopt = :sr1,)
    elseif stage == POLISH_SQP
        family == "unrestricted" &&
            error("algo_kwargs: POLISH_SQP is not wired for family=unrestricted (no opt_file kwarg on " *
                  "run_polish_checkpointed_unified -- see ALGORITHM_INVENTORY_2026-08-03.md). Use POLISH_DIRECT_BFGS.")
        return (opt_file = "csw_outer_phaseB_sqp_bfgs_maxit15.opt",)
    elseif stage == POLISH_DIRECT_BFGS
        family == "unrestricted" && return (outer_direct_hessopt = :bfgs,)
        return (opt_file = "csw_outer_phaseB_direct_bfgs_maxit15.opt",)
    else
        error("algo_kwargs: unhandled stage $stage")
    end
end

# ---------------------------------------------------------------------------------------------
# Cell result normalization -- the two driver families return NamedTuples with different field
# names for the same concepts (unified: kappa/best_feasible/wall_ext; CM/origin-ZC: kappa/best/wall).
# ---------------------------------------------------------------------------------------------

struct CellResult
    GT::Float64                 # kappa
    w::Union{Nothing,Vector{Float64}}   # best_feasible.w / best.w, nothing if no verified point
    Delta_star::Float64
    knitro_status::Int
    wall_s::Float64
    n_eval::Int
    checkpoint_path::String
end

function normalize_result(family::String, raw::NamedTuple)
    best = family == "unrestricted" ? raw.best_feasible : raw.best
    wall = family == "unrestricted" ? raw.wall_ext : raw.wall
    ckpt_path = family == "unrestricted" ? (raw.final_checkpoint isa AbstractString ? raw.final_checkpoint : "") : raw.ckpt_path
    if best === nothing
        return CellResult(NaN, nothing, NaN, Int(raw.knitro_status), Float64(wall), Int(raw.n_eval), ckpt_path)
    end
    return CellResult(Float64(raw.kappa), best.w, Float64(best.Delta), Int(raw.knitro_status), Float64(wall), Int(raw.n_eval), ckpt_path)
end

# ---------------------------------------------------------------------------------------------
# Never-regress rule (task Section 4/12): the runner must never return a result worse than the
# inherited incumbent. If the solver terminates without improvement, export the inherited point
# with result_source=inherited_incumbent.
# ---------------------------------------------------------------------------------------------

struct FinalCellReport
    family::String
    direction::Symbol
    target_delta::Float64
    algorithm::String
    seed_role::String
    inherited_GT::Union{Nothing,Float64}
    new_GT::Union{Nothing,Float64}
    final_GT::Float64
    final_w::Vector{Float64}
    final_Delta_star::Float64
    result_source::String        # "solved" | "inherited_incumbent"
    knitro_status::Int
    wall_s::Float64
    n_eval::Int
    checkpoint_path::String
end

"""
    apply_never_regress(direction, inherited, candidate) -> (final_GT, final_w, final_Delta, source)

`inherited` is `nothing` (no prior incumbent) or a NamedTuple with GT/w/Delta_star fields.
`candidate` is a `CellResult` (possibly with `w === nothing`, i.e. no verified point found).
"""
function apply_never_regress(direction::Symbol, inherited, candidate::CellResult)
    cand_ok = candidate.w !== nothing && isfinite(candidate.GT)
    if inherited === nothing
        cand_ok || error("apply_never_regress: no inherited incumbent AND no verified candidate point -- " *
                          "cannot produce a result for this cell (task spec requires at least one).")
        return (candidate.GT, candidate.w, candidate.Delta_star, "solved")
    end
    if cand_ok && better(direction, candidate.GT, inherited.GT)
        return (candidate.GT, candidate.w, candidate.Delta_star, "solved")
    end
    return (inherited.GT, inherited.w, inherited.Delta_star, "inherited_incumbent")
end

# ---------------------------------------------------------------------------------------------
# Manifest compatibility gate (task Section 3/10: revalidate every imported point under the
# target manifest; do not carry target-incompatible cache entries)
# ---------------------------------------------------------------------------------------------

struct ManifestMismatchError <: Exception
    field::String
    expected::Any
    actual::Any
end

Base.showerror(io::IO, e::ManifestMismatchError) =
    print(io, "ManifestMismatchError: field=", e.field, " expected=", e.expected, " actual=", e.actual)

"""
    assert_manifest_compatible(target::NamedTuple, seed_manifest::NamedTuple)

Both are NamedTuples of scalar scientific-manifest fields (W, delta grid basis, sigma, draw_seed,
draw_design, destination_sample, A_coordinate_mode, ...). Throws `ManifestMismatchError` on the
first field that differs. Deliberately field-by-field (not a single hash compare) so a mismatch is
diagnosable, not just a refusal.
"""
function assert_manifest_compatible(target::NamedTuple, seed_manifest::NamedTuple)
    for k in fieldnames(typeof(target))
        haskey(seed_manifest, k) || continue
        tv, sv = getfield(target, k), getfield(seed_manifest, k)
        tv == sv || throw(ManifestMismatchError(String(k), tv, sv))
    end
    return true
end

# ---------------------------------------------------------------------------------------------
# Checkpoint / resume for the orchestrator's OWN state (distinct from the underlying driver's own
# per-cell .jls checkpoints, which it still writes/reads itself unchanged)
# ---------------------------------------------------------------------------------------------

struct OrchestratorRunState
    family::String
    direction::Symbol
    target_delta::Float64
    stage::AlgoStage
    seed_role::String
    started_at::String
    report::Union{Nothing,FinalCellReport}
end

function save_run_state(path::AbstractString, state::OrchestratorRunState)
    tmp = path * ".tmp"
    serialize(tmp, state)
    mv(tmp, path; force = true)
    return path
end

load_run_state(path::AbstractString) = deserialize(path)::OrchestratorRunState

"""
    run_target_cell!(env, family, direction, target_delta, seeds, exploration_stage, polish_stage,
                      run_fn, ckpt_dir; explore_budget_s, polish_budget_s)

Top-level orchestration for ONE (family, direction, target_delta): resolve the inherited
incumbent from `env`, run the exploration stage from the best deduplicated seed, run the polish
stage from the best point exploration found, apply the never-regress rule, register the result
back into `env`, and return a `FinalCellReport`.

`run_fn(family, w0, algo_stage, budget_s, ckpt_dir; find_smallest) -> NamedTuple` is injected by
the caller (the real driver dispatch, e.g. calling `run_polish_checkpointed_unified`/
`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed` with `algo_kwargs(family, stage)...`
spliced in) -- kept as a parameter so this function's pure orchestration logic is unit-testable
with a mock `run_fn`, without invoking KNITRO.
"""
function run_target_cell!(env::MonotoneEnvelope, family::String, direction::Symbol, target_delta::Float64,
                           seeds::Vector{Seed}, exploration_stage::AlgoStage, polish_stage::AlgoStage,
                           run_fn::Function, ckpt_dir::AbstractString;
                           explore_budget_s::Float64, polish_budget_s::Float64)
    inherited = envelope_at(env, family, direction, target_delta)
    dedup = dedup_seeds(seeds)
    isempty(dedup) && inherited === nothing &&
        error("run_target_cell!($family,$direction,δ=$target_delta): no seeds and no inherited incumbent.")

    # Exploration: try each deduplicated seed, keep the best.
    best_explore = nothing
    best_seed_role = inherited === nothing ? "none" : "inherited_only"
    for s in dedup
        raw = run_fn(family, s.w, exploration_stage, explore_budget_s, ckpt_dir; find_smallest = direction_is_upper(direction))
        cand = normalize_result(family, raw)
        if cand.w !== nothing && (best_explore === nothing || better(direction, cand.GT, best_explore.GT))
            best_explore = cand
            best_seed_role = s.role
        end
    end

    # Polish: from the best exploration point (if any), else from the best seed directly.
    polish_start = best_explore !== nothing ? best_explore.w : (isempty(dedup) ? nothing : dedup[1].w)
    polished = nothing
    if polish_start !== nothing
        raw = run_fn(family, polish_start, polish_stage, polish_budget_s, ckpt_dir; find_smallest = direction_is_upper(direction))
        polished = normalize_result(family, raw)
    end

    candidate = if polished !== nothing && polished.w !== nothing &&
                   (best_explore === nothing || better(direction, polished.GT, best_explore.GT))
        polished
    elseif best_explore !== nothing
        best_explore
    else
        CellResult(NaN, nothing, NaN, -9999, 0.0, 0, "")
    end

    final_GT, final_w, final_Delta, source = apply_never_regress(direction, inherited, candidate)

    report = FinalCellReport(family, direction, target_delta,
        source == "solved" ? string(polish_stage) : "inherited", best_seed_role,
        inherited === nothing ? nothing : inherited.GT,
        candidate.w === nothing ? nothing : candidate.GT,
        final_GT, final_w, final_Delta, source,
        candidate.knitro_status, candidate.wall_s, candidate.n_eval, candidate.checkpoint_path)

    register!(env, family, direction, target_delta, final_GT, final_w, final_Delta,
              (source_delta = target_delta, source_start = 0,
               outer_vector_path = report.checkpoint_path,
               outer_vector_sha256 = ""))

    return report
end
