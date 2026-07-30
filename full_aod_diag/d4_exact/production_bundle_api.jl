# ================================================================================================
# architecture/production-operator-bundle-hardening-2026-07-30
#
# The shared, family-agnostic API that makes "every production runner constructs OperatorPsiBundle,
# with no runtime choice of representation" a structural property of the call graph rather than a
# convention every family's own code has to independently get right (and, per
# dense_bundle_incident_postmortem_2026-07-29.zip, independently got wrong -- twice, in two
# different families, on two different days).
#
# This file does NOT reimplement any family's economic/CM/Frechet/ZC state construction -- that
# stays exactly where it already lives (build_cm_production_context, build_cm_frechet_production_
# context, build_cm_meanzc_production_context, build_originzc_production_context,
# build_unrestricted_operator_ctx). What lives here is the part that was previously re-implemented,
# slightly differently, once per family: the production/diagnostic purpose distinction, the
# type-safe wrapper, the live invariant assertion, and the backend manifest. See
# dense_reference_diagnostics.jl (companion file) for the diagnostic-only dense-construction path.
# ================================================================================================

isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))

using Dates: now

# ---- §2/§3: typed purpose dispatch (not Symbols) -----------------------------------------------

"Marker type hierarchy for why a bundle is being constructed -- dispatched on, not stringly-typed."
abstract type RunPurpose end

"The only purpose a real production campaign/checkpoint/standalone runner may declare."
struct ProductionPurpose <: RunPurpose end

"""
    DenseReferencePermit(; reason, caller)

Required to construct a dense-capable bundle outside a pure builder-vs-builder equivalence test.
Both fields are mandatory and free text -- this is a paper trail, not an access-control mechanism;
the actual gate is `run purpose isa ProductionPurpose` (fatal) vs not (banner + counter).
"""
struct DenseReferencePermit
    reason::String
    caller::String
end
DenseReferencePermit(; reason::AbstractString, caller::AbstractString) =
    DenseReferencePermit(String(reason), String(caller))

"A diagnostic/test run that has an explicit, audited reason to construct a dense reference bundle."
struct DenseReferencePurpose <: RunPurpose
    permit::DenseReferencePermit
end

# ---- §6: live counters for dense-reference construction, independent of no_dense_g_counters.jl's
# evaluation-time counters (§8 keeps these two axes separate on purpose) ------------------------

const DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS = Ref(0)
"Each entry: (reason, caller, bundle_type::String, at::String, backtrace::Vector). Small, in-memory, gate/report-only."
const DENSE_REFERENCE_CONSTRUCTION_LOG = NamedTuple[]

reset_dense_reference_construction_log!() =
    (DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] = 0; empty!(DENSE_REFERENCE_CONSTRUCTION_LOG); nothing)

# ---- production SHA, resolved once, used in every fatal message/manifest ----------------------

const PRODUCTION_SHA = Ref{String}("")
function _production_sha!()
    isempty(PRODUCTION_SHA[]) || return PRODUCTION_SHA[]
    PRODUCTION_SHA[] = try
        strip(read(`git -C $(@__DIR__) rev-parse HEAD`, String))
    catch
        "unknown (git rev-parse failed)"
    end
    return PRODUCTION_SHA[]
end

# ---- §3: type-safe production vs. dense-reference contexts -------------------------------------

"Fields that must be structurally absent from a production bundle. Names, not values -- see assert_production_operator_bundle! for the check that they are not merely nothing/empty but ABSENT."
const LEGACY_DENSE_BUNDLE_FIELDS = (:H, :H_copy, :G, :K, :ones, :moments!)

"""
    ProductionContext{C,B<:OperatorPsiBundle}

The ONLY context type a production runner may hold after context construction. `obj` is typed
`B<:OperatorPsiBundle` -- not `Any`, not the `PsiObjectiveBundle` abstract supertype (which
`PsiObjectiveBundleImplicit` also subtypes), not a `Union` that could ever admit the dense bundle.
`inner` is the family's own native context object (whatever `build_*_production_context` /
`build_unrestricted_operator_ctx` already returns) -- this file does not re-shape that, it wraps it.
"""
struct ProductionContext{C,B<:OperatorPsiBundle}
    family::Symbol
    runner::String
    inner::C
    obj::B
end

"""
    DenseReferenceContext{C,B}

Structurally distinct from `ProductionContext` (a different type, not the same type with a wider
bundle bound) -- a function typed to accept `ProductionContext` cannot accept this by construction,
and vice versa. Diagnostic/test-only; see dense_reference_diagnostics.jl.
"""
struct DenseReferenceContext{C,B}
    permit::DenseReferencePermit
    family::Symbol
    runner::String
    inner::C
    obj::B
end

"""
    DenseReferencePsiObjectiveBundle

Prominent rename/alias for `PsiObjectiveBundleImplicit` (`cc_algo/PsiObjectiveBundle.jl`), used in
all NEW diagnostic code from this task onward so it cannot be mistaken for the ordinary production
representation. The underlying type is unchanged (this is an alias, not a new struct) -- existing
code that names `PsiObjectiveBundleImplicit` directly (equivalence tests, legacy call sites) is
unaffected and continues to work.
"""
const DenseReferencePsiObjectiveBundle = CS.PsiObjectiveBundleImplicit

# ---- §1 helper: does inner.obj resolve to a bundle field the same way for every family? --------
# Every restricted family (flexible_cm, common_frechet, cm_meanzc, origin_zc) returns a context
# with a `.ctx_cm.obj` field (confirmed live: cm_checkpoint.jl reads `pcx.ctx_cm.obj` identically
# for all three of its branches; cm_originzc_checkpoint.jl reads the same shape). unrestricted's
# `build_unrestricted_operator_ctx` returns a context with `.obj` directly. This helper is the
# ONE place that duck-types across that difference, so nothing else has to.
function _resolve_bundle(inner)
    hasproperty(inner, :ctx_cm) && return inner.ctx_cm.obj
    hasproperty(inner, :obj) && return inner.obj
    error("_resolve_bundle: context of type $(typeof(inner)) has neither .ctx_cm.obj nor .obj -- " *
          "cannot locate the bundle to validate. This is a new family shape prepare_production_run " *
          "does not yet know how to inspect; add a case here rather than skipping validation.")
end

# ---- §4: the one shared production factory ------------------------------------------------------

"""
    PreparedRun

Standardized return value from `prepare_production_run` -- both a runner and a test can consume
this without either reaching into family-specific globals/stashes to find out what happened.
"""
struct PreparedRun{C,B<:OperatorPsiBundle}
    ctx::ProductionContext{C,B}
    manifest::NamedTuple
end

"""
    prepare_production_run(family::Symbol, runner::String, build_inner::Function; extra_manifest=NamedTuple()) -> PreparedRun

The one authoritative, family-agnostic preparation entry point every real production driver calls
immediately after building its own family-specific state.

`build_inner` is a zero-argument closure, supplied by the calling driver, that performs the
family-specific construction (calling `build_cm_production_context`/`build_cm_frechet_production_
context`/`build_cm_meanzc_production_context`/`build_originzc_production_context`/
`build_unrestricted_operator_ctx` with THAT family's own real kwargs) and returns the family's
native context object. **`build_inner` must not expose or forward any representation choice** --
by convention (enforced by review/the static guard, §13, not by this function, which cannot see
inside an opaque closure) every real driver's `build_inner` hardcodes `moment_representation =
:operator` in its call to the family builder.

Steps performed here (task §4.1-§4.6): family-specific state is built by the closure (1); this
function only validates that the result is an `OperatorPsiBundle` (2) -- it does not itself choose
representations; wraps it in a type-safe `ProductionContext` (3); derives a live backend manifest
(4); asserts the production invariant, fatally (5); returns the `PreparedRun` (6).
"""
function prepare_production_run(family::Symbol, runner::String, build_inner::Function;
                                 extra_manifest::NamedTuple = NamedTuple())
    inner = build_inner()
    obj = _resolve_bundle(inner)
    obj isa OperatorPsiBundle ||
        error("prepare_production_run($family, $runner): build_inner() returned a context whose " *
              "bundle is $(typeof(obj)), not <: OperatorPsiBundle. Production runners may only " *
              "construct OperatorPsiBundle -- if a dense reference bundle was genuinely intended, " *
              "use DenseReferenceDiagnostics.prepare_context instead of prepare_production_run.")
    pcx = ProductionContext(family, runner, inner, obj)
    manifest = derive_backend_manifest(pcx; purpose_label = "production", extra = extra_manifest)
    assert_production_operator_bundle!(pcx; where_ = "prepare_production_run($family, $runner)")
    return PreparedRun(pcx, manifest)
end

# ---- §5: the fatal live assertion ---------------------------------------------------------------

struct ProductionInvariantViolation <: Exception
    message::String
end
Base.showerror(io::IO, e::ProductionInvariantViolation) = print(io, e.message)

"""
    assert_production_operator_bundle!(pcx::ProductionContext; where_="")

Fatal (throws `ProductionInvariantViolation`, terminating the cell/campaign) unless ALL of:
  - `pcx.obj isa OperatorPsiBundle`
  - none of `H`/`H_copy`/`G`/`K`/`ones`/`moments!` are fields of `typeof(pcx.obj)`
  - `select_G_from_H(pcx.obj, ...)` throws (dense G access is not applicable/reachable)
  - `DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 0`

Call this immediately after context construction, before outer KNITRO initialization, on every
checkpoint resume, and before writing a campaign cell's RUNNING marker (task §5). A first-callback
defense-in-depth check is optional and NOT implemented here -- see production_bundle_preflight.jl's
own smoke-callback step for that layer instead of duplicating the assertion inside every family's
own FG callback.
"""
function assert_production_operator_bundle!(pcx::ProductionContext; where_::String = "")
    obj = pcx.obj
    problems = String[]

    obj isa OperatorPsiBundle || push!(problems, "ctx.obj is $(typeof(obj)), not <: OperatorPsiBundle")

    for f in LEGACY_DENSE_BUNDLE_FIELDS
        hasfield(typeof(obj), f) && push!(problems, "bundle type $(typeof(obj)) has forbidden legacy field :$f")
    end

    select_g_reachable = try
        select_G_from_H(obj)
        true
    catch
        false
    end
    select_g_reachable && push!(problems, "select_G_from_H(obj) did not throw -- dense G access is reachable")

    DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 0 ||
        push!(problems, "dense-reference construction count is $(DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[]), expected 0")

    isempty(problems) && return true

    bt = sprint(Base.show_backtrace, backtrace())
    msg = string(
        "\n", "="^80, "\n",
        "FATAL: production-operator-bundle invariant violated\n",
        "  Family:          ", pcx.family, "\n",
        "  Runner:          ", pcx.runner, "\n",
        "  Context type:    ", typeof(pcx), "\n",
        "  Bundle type:     ", typeof(obj), "\n",
        "  Production SHA:  ", _production_sha!(), "\n",
        "  Location:        ", where_, "\n",
        "  Problems:\n", join(("    - " * p for p in problems), "\n"), "\n",
        "  Diagnostic API to use if a dense reference bundle was genuinely intended:\n",
        "    DenseReferenceDiagnostics.prepare_context(family, runner, build_inner; permit=DenseReferencePermit(reason=..., caller=...))\n",
        "  Stack trace:\n", bt, "\n",
        "="^80, "\n")
    throw(ProductionInvariantViolation(msg))
end

# ---- §7/§8: live backend manifest, structural vs. evaluation kept separate ----------------------

"""
    derive_backend_manifest(pcx; purpose_label, extra=NamedTuple()) -> NamedTuple

Every field is READ off live state (`typeof`, `hasfield`, `no_dense_g_report()`) -- none of this is
a hardcoded string literal, unlike the pre-existing `NO-H BUNDLE FACTS: bundle_type=OperatorPsiBundle`
print blocks this replaces (postmortem §4). `structural` and `evaluation` are reported as separate
sub-namedtuples per task §8 -- a clean `evaluation` never implies a correct `structural`, and this
manifest shape makes that impossible to conflate by construction (a caller has to explicitly reach
into `.evaluation` to get the counters; `.structural` never contains them).
"""
function derive_backend_manifest(pcx; purpose_label::String, extra::NamedTuple = NamedTuple())
    obj = pcx.obj
    T = typeof(obj)
    legacy_fields_present = Dict{String,Bool}(String(f) => hasfield(T, f) for f in LEGACY_DENSE_BUNDLE_FIELDS)
    select_g_applicable = try
        select_G_from_H(obj)
        true
    catch
        false
    end
    counters = no_dense_g_report()

    structural = (
        run_purpose = purpose_label,
        family = pcx.family,
        runner = pcx.runner,
        context_type = string(typeof(pcx)),
        bundle_type = string(T),
        legacy_fields_present = legacy_fields_present,
        any_legacy_field_present = any(values(legacy_fields_present)),
        select_G_from_H_applicable = select_g_applicable,
        dense_reference_construction_count = DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[],
        production_sha = _production_sha!(),
    )
    evaluation = (
        full_G_materializations = counters.full_G_materializations,
        dense_economic_G_materializations = counters.dense_economic_G_materializations,
        dense_CM_G_materializations = counters.dense_CM_G_materializations,
        dense_ZC_G_materializations = counters.dense_ZC_G_materializations,
        dense_Frechet_G_materializations = counters.dense_Frechet_G_materializations,
        generic_dense_FG_calls = counters.generic_dense_FG_calls,
        operator_FG_calls = counters.operator_FG_calls,
        dense_cross_hessian_calls = counters.dense_cross_hessian_calls,
        operator_cross_hessian_calls = counters.operator_cross_hessian_calls,
        dense_reference_verification_calls = counters.dense_reference_verification_calls,
        operator_verification_calls = counters.operator_verification_calls,
    )
    bundle_invariant_pass = (obj isa OperatorPsiBundle) &&
                             !structural.any_legacy_field_present &&
                             !select_g_applicable &&
                             DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 0

    base = (structural = structural, evaluation = evaluation,
            bundle_invariant_pass = bundle_invariant_pass,
            generated_at = string(now()))
    return merge(base, extra)
end

# ---- minimal, dependency-free JSON writer (this repo has no JSON.jl dependency; the manifest
# shape is simple enough -- nested NamedTuple/Dict/Vector/primitives -- not to warrant adding one) --

_json_escape(s::AbstractString) = replace(replace(String(s), "\\" => "\\\\"), "\"" => "\\\"")

function _json_write(io::IO, x::AbstractString, indent::Int)
    print(io, '"', _json_escape(x), '"')
end
function _json_write(io::IO, x::Symbol, indent::Int)
    _json_write(io, String(x), indent)
end
function _json_write(io::IO, x::Bool, indent::Int)
    print(io, x)
end
function _json_write(io::IO, x::Union{Integer,AbstractFloat}, indent::Int)
    print(io, x)
end
function _json_write(io::IO, ::Nothing, indent::Int)
    print(io, "null")
end
function _json_write(io::IO, x::NamedTuple, indent::Int)
    isempty(x) && (print(io, "{}"); return)
    pad = "  "^(indent + 1)
    print(io, "{\n")
    ks = keys(x)
    for (i, k) in enumerate(ks)
        print(io, pad, '"', _json_escape(String(k)), "\": ")
        _json_write(io, getfield(x, k), indent + 1)
        i < length(ks) && print(io, ",")
        print(io, "\n")
    end
    print(io, "  "^indent, "}")
end
function _json_write(io::IO, x::AbstractDict, indent::Int)
    isempty(x) && (print(io, "{}"); return)
    pad = "  "^(indent + 1)
    print(io, "{\n")
    ks = collect(keys(x))
    for (i, k) in enumerate(ks)
        print(io, pad, '"', _json_escape(String(k)), "\": ")
        _json_write(io, x[k], indent + 1)
        i < length(ks) && print(io, ",")
        print(io, "\n")
    end
    print(io, "  "^indent, "}")
end
function _json_write(io::IO, x::AbstractVector, indent::Int)
    isempty(x) && (print(io, "[]"); return)
    pad = "  "^(indent + 1)
    print(io, "[\n")
    for (i, v) in enumerate(x)
        print(io, pad)
        _json_write(io, v, indent + 1)
        i < length(x) && print(io, ",")
        print(io, "\n")
    end
    print(io, "  "^indent, "]")
end
_json_write(io::IO, x, indent::Int) = _json_write(io, string(x), indent)   # fallback: stringify (e.g. Symbol keys already handled above)

"""
    write_backend_manifest_atomic(manifest, path) -> path

Writes to a same-directory temp file, then `mv(...; force=true)` -- the rename is atomic on a
POSIX filesystem, so a reader of `path` never observes a partially-written manifest (task §7's
"may not move from PREPARING to RUNNING unless bundle_invariant_pass=true has been written
atomically").
"""
function write_backend_manifest_atomic(manifest, path::AbstractString)
    dir = dirname(path)
    isdir(dir) || mkpath(dir)
    tmp = joinpath(dir, ".$(basename(path)).tmp.$(getpid())")
    open(tmp, "w") do io
        _json_write(io, manifest, 0)
        print(io, "\n")
    end
    mv(tmp, path; force = true)
    return path
end
