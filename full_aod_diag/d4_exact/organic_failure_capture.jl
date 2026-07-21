# ============================================================================
# Organic -300 (and other genuine, non-screen-certified) inner-failure capture
# (task §7) + one-command replay (task §8).
#
# Purely ADDITIVE driver-level instrumentation: wraps the existing, unmodified
# `screened_eval` call sites in c10_d20_production_driver.jl's cb_F!/cb_G! (no change
# to infeasibility_screen.jl/fast_range_screen.jl/oracle.jl/oracle_fast.jl/the inner
# CC objective, gradient, Hessian, or warm-start policy). A failure is "organic" iff
# it passed every fast exact screen (pairwise/witness/winner-scan/envelope/winning-
# range/safety-net -- i.e. inner_status is NOT one of this repo's own <=-9000 exact-
# screen sentinels, see knitro_status.jl) and genuinely reached KNITRO's inner CC dual
# solve, which then failed to return a feasible status.
# ============================================================================
using JLD2, Dates

mutable struct OrganicFailureCollector
    outdir::String
    max_n::Int
    saved::Int
end

"`OrganicFailureCollector(outdir; max_n=5)` -- captures at most the first `max_n` organic failures seen through this collector, then goes silent (further organic failures still reject the KNITRO trial as before, just aren't archived)."
OrganicFailureCollector(outdir::AbstractString; max_n::Int = 5) = (mkpath(outdir); OrganicFailureCollector(outdir, max_n, 0))

"True iff `status` is a genuine organic inner failure: not a feasible code, and not one of this repo's own exact-screen sentinels (<=-9000, see knitro_status.jl)."
is_organic_failure(status::Integer) = !(status in FEASIBLE_CODES) && status > -9000

"Hex digest via Base.hash (no extra package dependency) -- collision-resistant enough for a driver-level dedup/reproducibility tag, not a cryptographic guarantee."
_hexhash(s::AbstractString) = string(hash(s), base = 16)

function _point_hash(w::AbstractVector{<:Real}, δ::Real)
    return _hexhash(string(round.(w; digits = 12), "|", δ))
end

function _config_hash(; W::Integer, δ::Real, find_smallest::Bool, draw_design::Symbol, draw_seed::Integer)
    return _hexhash(string(W, "|", δ, "|", find_smallest, "|", draw_design, "|", draw_seed))
end

"""
    maybe_capture_organic_failure!(collector, label, w, xf, r, ctx, pe, sc, n_eval, knitro_iter,
        δ, find_smallest, draw_design, draw_seed, warm_source, dual_before, checkpoint_parent)

Called from cb_F!/cb_G! immediately after a `screened_eval` result comes back. If
`is_organic_failure(r.inner_status)` and the collector hasn't hit `max_n` yet, archives a
complete, reproducible record via JLD2 (exact) + a readable JSON summary (grep-able).
No-op (fast, one field comparison) otherwise -- safe to call on every evaluation.
"""
function maybe_capture_organic_failure!(collector::OrganicFailureCollector, label::String,
        w::AbstractVector{<:Real}, xf, r, ctx, pe, sc, n_eval::Integer, knitro_iter::Integer,
        δ::Real, find_smallest::Bool, draw_design::Symbol, draw_seed::Integer,
        warm_source::Symbol, dual_before::AbstractVector{<:Real},
        checkpoint_parent::Union{Nothing,AbstractString})
    is_organic_failure(r.inner_status) || return nothing
    collector.saved >= collector.max_n && return nothing
    collector.saved += 1
    idx = collector.saved

    zfree = collect(w[2:end]); g = w[1]
    logA_full = pivot_expand(zfree, pe)
    status_info = decode_knitro_status(r.inner_status)
    ph = _point_hash(collect(w), δ)
    ch = _config_hash(W = ctx.W, δ = δ, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed)

    record = (
        idx = idx, label = label, timestamp = string(now()),
        g = g, zfree = zfree, logA_full = logA_full,
        point_hash = ph, config_hash = ch,
        draw_checksum_uniform = ctx.draw_meta.checksum_uniform,
        draw_checksum_transformed = ctx.draw_meta.checksum_transformed,
        δ = δ, find_smallest = find_smallest, direction = find_smallest ? :upper : :lower,
        W = ctx.W, draw_design = draw_design, draw_seed = draw_seed,
        screen_counts = as_namedtuple(sc),
        inner_status = r.inner_status, status_name = status_info.name, status_category = status_info.category,
        status_meaning = status_info.meaning,
        Delta_dual = get(r, :Delta_dual, NaN),
        gravity_value = get(r, :gravity_value, NaN),
        warm_source = warm_source,
        dual_before = collect(dual_before),
        dual_after_if_finite = all(isfinite, ctx.obj.x) ? collect(ctx.obj.x) : nothing,
        n_eval_at_capture = n_eval, knitro_iter_at_capture = knitro_iter,
        checkpoint_parent = checkpoint_parent,
        # RNG-independent reproduction: build_fullA_context + set_context_delta! + a direct
        # screened_eval call at this exact w -- no dependence on the outer KNITRO trajectory
        # that happened to discover this point.
        reproduction_command = "include(\"c10_d20_production_driver.jl\"); ctx=build_fullA_context(W=$(ctx.W), δ=$(δ), find_smallest=$(find_smallest), draw_design=:$(draw_design), draw_seed=$(draw_seed)); replay_organic_failure(\"$(joinpath(collector.outdir, "organic_failure_$(idx).jld2"))\")",
    )

    jld2_path = joinpath(collector.outdir, "organic_failure_$(idx).jld2")
    json_path = joinpath(collector.outdir, "organic_failure_$(idx)_summary.json")
    jldsave(jld2_path; record = record)
    open(json_path, "w") do io
        # Minimal hand-rolled JSON writer (no JSON.jl dependency assumed) -- summary fields only,
        # the JLD2 file is the authoritative exact record (full zfree/logA_full/dual vectors).
        println(io, "{")
        println(io, "  \"idx\": ", idx, ",")
        println(io, "  \"label\": \"", label, "\",")
        println(io, "  \"point_hash\": \"", ph, "\",")
        println(io, "  \"config_hash\": \"", ch, "\",")
        println(io, "  \"delta\": ", δ, ",")
        println(io, "  \"direction\": \"", find_smallest ? "upper" : "lower", "\",")
        println(io, "  \"inner_status\": ", r.inner_status, ",")
        println(io, "  \"status_name\": \"", status_info.name, "\",")
        println(io, "  \"status_meaning\": \"", replace(status_info.meaning, "\"" => "'"), "\",")
        println(io, "  \"n_eval_at_capture\": ", n_eval, ",")
        println(io, "  \"knitro_iter_at_capture\": ", knitro_iter, ",")
        println(io, "  \"warm_source\": \"", warm_source, "\",")
        println(io, "  \"jld2_path\": \"", jld2_path, "\"")
        println(io, "}")
    end
    @info "[$(label)] organic failure #$(idx)/$(collector.max_n) captured: status=$(r.inner_status) ($(status_info.name)) -> $(jld2_path)"
    return record
end

"""
    replay_organic_failure(jld2_path; ctx=nothing) -> NamedTuple

One-command replay (task §8) of a captured organic failure: loads the record, verifies
the point/draw-checksum match, and runs (1) all fast exact screens, (2) a cold inner
solve, (3) a production warm-start solve seeded from the record's own `dual_before`.
Builds its own context via `build_fullA_context` if `ctx` isn't supplied (costs the
usual ~65-83s real-data setup). Does NOT implement dual-ray extraction / cutting-plane
phase I / alternative KNITRO settings -- those are explicitly out of scope for this pass
(task §8), this only re-verifies exact reproducibility and reports fresh diagnostics.
"""
function replay_organic_failure(jld2_path::AbstractString; ctx = nothing, pe = nothing, rsc = nothing)
    rec = load(jld2_path, "record")
    if ctx === nothing
        built = build_fullA_context(W = rec.W, δ = rec.δ, find_smallest = rec.find_smallest,
            draw_design = rec.draw_design, draw_seed = rec.draw_seed)
        ctx = built.ctx; pe = built.pe; rsc = built.rsc
    end
    checksum_ok = ctx.draw_meta.checksum_uniform == rec.draw_checksum_uniform &&
                  ctx.draw_meta.checksum_transformed == rec.draw_checksum_transformed
    checksum_ok || @warn "replay_organic_failure: draw checksum MISMATCH -- this replay is NOT using the exact same draws as the captured failure. Reported diagnostics below are not a faithful reproduction." rec.point_hash

    w = vcat(rec.g, rec.zfree)
    ph_now = _point_hash(w, rec.δ)
    point_ok = ph_now == rec.point_hash
    point_ok || @warn "replay_organic_failure: reconstructed point hash does not match the recorded one." ph_now rec.point_hash

    xf = x_free_from_w(w, pe)
    sc = ScreenCounters(); n_eval = Ref(0)

    r_screen, meta_screen = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
    r_cold, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = false)
    ctx.obj.x .= rec.dual_before
    r_warm, _ = screened_eval(xf, ctx, rsc, sc, n_eval; warm = true, zfree = rec.zfree)

    return (checksum_ok = checksum_ok, point_ok = point_ok, screens = as_namedtuple(sc),
            original_status = rec.inner_status, original_status_name = rec.status_name,
            cold_status = r_cold.inner_status, cold_status_decoded = decode_knitro_status(r_cold.inner_status),
            warm_from_recorded_dual_status = r_warm.inner_status,
            warm_from_recorded_dual_status_decoded = decode_knitro_status(r_warm.inner_status),
            cold_Delta_dual = get(r_cold, :Delta_dual, NaN), warm_Delta_dual = get(r_warm, :Delta_dual, NaN))
end
