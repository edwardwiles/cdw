# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §10: a
# trusted, INDEPENDENT full-rebuild fixed-dual reference for ANY family
# satisfying the five-accessor contract (profiled_outer_gradient_layout_
# contract_2026-08-01.jl). DIAGNOSTIC NAMESPACE (all public names prefixed
# `diag_`) -- slow (O(W*D*Ddest) per probe, one full `build_compressed_factual`
# rebuild), deliberately NOT using the shared engine's incremental winner
# cache (`ProfiledLFixCache`/`update_winner_o1`), so it cannot share a bug
# with it. Generalizes the already-gated unrestricted-only
# `profiled_lfix_at`/`profiled_composite_gradient_at`
# (profiled_outer_gradient_fd_2026-08-01.jl) to add ONE more independent
# additive term -- a restriction contribution, evaluated once at the base
# point and held fixed across every +/- probe -- so it can serve as ground
# truth for restricted families too, and specifically prove (by direct
# numerical comparison, not by assumption) that the shared engine's
# incremental treatment of `restriction_contrib0` (baked into `q0` once,
# never revisited) agrees with an entirely independent full rebuild that
# ALSO holds the restriction contribution fixed across probes but recomputes
# everything else from scratch every time.
# Never materializes dense economic G (uses `reduced_homogeneous_dual_
# contraction`, the same O(W) linear-functional kernel the shared engine's
# own cache-build step and `profiled_lfix_at` already use, unchanged).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :decode_outer_profiled) || error("profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl requires outer_coordinate_layout_profiled_2026-07-31.jl to be included first.")
isdefined(Main, :reduced_homogeneous_dual_contraction) || error("profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first.")
isdefined(Main, :validate_family_layout_contract) || error("profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")

"""
    diag_profiled_lfix_at_reduced(w, ctx, pe, layout, β_econ, ζ, obj, restriction_contrib_w) -> Float64

Full-rebuild fixed-dual `L_fix` at outer point `w` (`== Delta_dual`'s own sign
convention), restricted-family generalization of `profiled_lfix_at`. Rebuilds
`compressed_factual` fresh (no incremental cache) via the unchanged production
`build_compressed_factual`, contracts ONLY the economic dual slice `β_econ`
against `layout` via the unchanged `reduced_homogeneous_dual_contraction`
kernel, then adds `restriction_contrib_w` (a per-draw vector, already
evaluated once at the ORIGINAL fixed dual point -- never recomputed as `w`
varies, per task §3's economic-gradient theorem) directly into `q` before
`Psi`. Zero-vector `restriction_contrib_w` reduces this exactly to
`profiled_lfix_at`'s own formula.

CONVENTION (matches `build_shared_profiled_lfix_cache`'s `q0` construction
exactly, not an independent choice): `restriction_contrib_w` passed here must
ALREADY be `SW`-weighted -- `reduced_homogeneous_dual_contraction`'s own
output `t_econ` is `SW`-weighted internally (this is why the pre-existing
unrestricted-only `profiled_lfix_at`/`build_profiled_lfix_cache` equivalence
holds to machine precision with no separate `SW` factor visible in either),
and `restriction_contrib0(fctx,ev)` is specified (contract file) to be the
UNWEIGHTED per-draw quantity, `SW`-weighted once by the caller -- exactly
like `const_part`/`contrib0`/`cf_raw_κcf`, which are also pre-`SW`
quantities inside `acc` before `t0 = SW[w]*acc`. `diag_profiled_full_rebuild_
gradient` below performs this `SW` weighting once before calling this
function repeatedly, matching the shared cache's own convention bit-for-bit.
"""
function diag_profiled_lfix_at_reduced(w::AbstractVector{Float64}, ctx, pe::PivotGravityElimOnRetained,
        layout::ProfiledEconomicMomentLayout, β_econ::AbstractVector{Float64}, ζ::Float64, obj,
        restriction_contrib_w::AbstractVector{Float64})
    decoded = decode_outer_profiled(w, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    cf = build_compressed_factual(θ_full, ctx; check_ties = false)
    t_econ = reduced_homogeneous_dual_contraction(β_econ, cf, ctx, θ_full, layout)
    q = -ζ .- t_econ .- restriction_contrib_w
    Psi_q = similar(q)
    obj.Psi!(Psi_q, q)
    return -(sum(Psi_q) / obj.M + ζ)
end

"""
    diag_profiled_full_rebuild_gradient(w0, ctx, fctx, ev; h=0.01, include_restriction=true) -> (g, meta)

Independent (non-incremental) reference gradient for ANY family satisfying
the five-accessor contract (task §10). Rebuilds the complete profiled
economic state at every `+/-` probe. `include_restriction=true` (default)
folds `restriction_contrib0(fctx, ev)` into every probe's `q`, exactly
mirroring the shared engine's own treatment (task §10: "Restriction
contributions may be included directly to prove they cancel for A/gp
probes") -- `include_restriction=false` zeroes it out, letting a caller
directly measure how much (if any) a given restriction contribution's
magnitude perturbs the resulting gradient (the genuine, checkable version of
the "cancellation" claim: NOT that the gradient is independent of the
restriction contribution's value, but that the shared engine's O(1)
incremental treatment of a FIXED restriction contribution agrees with this
full O(W*D*Ddest) independent rebuild of the SAME fixed treatment, to machine
precision).

`h` is either a single scalar (same step for every coordinate -- matches
`profiled_composite_gradient_at`'s own fixed-h convention) or a length-
`outer_dim_profiled(pe)` vector of PER-COORDINATE steps. **IMPORTANT,
verified live (not assumed) 2026-08-01**: the shared engine
(`profiled_composite_gradient_from_cache`) uses an ADAPTIVE, per-coordinate
bandwidth (`profiled_select_bandwidth`, targeting a switch-mass window), NOT
a single fixed `h`. Comparing it against this reference at a mismatched
fixed `h` (e.g. the default `h=0.01`) reproduces only the ALREADY-KNOWN,
pre-existing incremental-vs-full-rebuild bandwidth-mismatch gap (cos_sim
~0.9999, max_rel_err ~2-3.6 at D4 -- confirmed by literally re-running the
pre-existing, already-committed
`test_profiled_incremental_vs_fullrebuild_2026-08-01.jl` gate script, which
does NOT reproduce that file's own header comment's claimed
"1.0000000000/6e-6" machine-precision result; see
PROFILED_RESTRICTED_OUTER_GRADIENT_MASTER_2026-08-01.md for the full
writeup) -- NOT a defect in this reference or in the shared engine's
restriction-block generalization. Passing `h_used` (a caller's
`shared_family_outer_gradient` meta's own `h_used` vector, coordinate 1
ignored since gp uses the exact analytic formula, not FD) gives the true
apples-to-apples comparison this function is meant for, confirmed to reduce
the gap to `~1e-16` (machine precision) once bandwidths are matched.
"""
function diag_profiled_full_rebuild_gradient(w0::AbstractVector{Float64}, ctx, fctx, ev;
        h::Union{Float64,AbstractVector{Float64}} = 0.01, include_restriction::Bool = true)
    v = validate_family_layout_contract(fctx)
    layout = v.layout; erange = v.economic_dual_range; pe = v.pe
    β_econ = ev.result.beta[erange]
    ζ = ev.result.zeta; obj = ev.obj
    W = ev.st.cf.W
    SW = ev.st.cf.SW
    rc_raw = include_restriction ? restriction_contrib0(fctx, ev) : zeros(W)
    length(rc_raw) == W || error("diag_profiled_full_rebuild_gradient: restriction_contrib0 length $(length(rc_raw)) != W=$W")
    rc = SW .* rc_raw  # SW-weight ONCE here, matching build_shared_profiled_lfix_cache's q0 convention exactly

    n_total = outer_dim_profiled(pe)
    h_vec = h isa AbstractVector ? h : fill(h, n_total)
    length(h_vec) == n_total || error("diag_profiled_full_rebuild_gradient: h vector length $(length(h_vec)) != n_total=$n_total")
    g = Vector{Float64}(undef, n_total)
    @inbounds for k in 1:n_total
        hk = h_vec[k]
        wp = copy(w0); wp[k] += hk
        wm = copy(w0); wm[k] -= hk
        Lp = diag_profiled_lfix_at_reduced(wp, ctx, pe, layout, β_econ, ζ, obj, rc)
        Lm = diag_profiled_lfix_at_reduced(wm, ctx, pe, layout, β_econ, ζ, obj, rc)
        g[k] = (Lp - Lm) / (2 * hk)
    end
    return g, (w0 = collect(Float64, w0), h = h_vec, include_restriction = include_restriction)
end
