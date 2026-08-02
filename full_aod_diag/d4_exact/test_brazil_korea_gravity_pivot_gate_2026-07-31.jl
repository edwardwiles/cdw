# ============================================================================
# Brazil->Korea gravity-exclusion task (2026-07-31), task §7 "Required pivot tests" +
# §2 country resolution, run against the REAL D=20 economy (Brazil/Korea only exist there).
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_brazil_korea_gravity_pivot_gate_2026-07-31.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["context_real_d20.jl", "gravity_elimination.jl", "country_resolve.jl"]
    include(joinpath(_D4E, f))
end
using DelimitedFiles, Random, Printf

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

# ---- §2: country resolution ----
realDataDir = joinpath(_D4E, "..", "..", "real_data", "noah_D20")
countries = vec(readdlm(joinpath(realDataDir, "countries.csv"), ',', String))
bra_idx = resolve_country_index(countries, "Brazil")
kor_idx = resolve_country_index(countries, "Korea")
row_idx_global = findfirst(==("row"), countries)
check("exactly one Brazil origin resolved (index=$bra_idx)", bra_idx isa Int)
check("exactly one Korea destination resolved (index=$kor_idx)", kor_idx isa Int)
check("Brazil != Korea", bra_idx != kor_idx)
D = length(countries)
named_dest = filter(!=(row_idx_global), 1:D)
kor_slot = global_to_dest_slot(kor_idx, named_dest)
gravity_exclude_cells = [(bra_idx, kor_slot)]
println("bra_idx=$bra_idx kor_idx=$kor_idx kor_dest_slot=$kor_slot named_dest=$named_dest")

# ---- build the real D=20 context WITH the Brazil->Korea exclusion, small W (diagnostic, not
# the campaign W=500,000 -- pivot/residual properties are W-independent, draws don't enter them) ----
Random.seed!(20260731)
ctx = d20_real_setup(W = 2000, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
                      gravity_exclude_cells = gravity_exclude_cells, build_screen = false)
check("ctx.D_dest == 19 (ROW excluded)", ctx.D_dest == 19)
check("ctx.gravity_exclude_cells == [(bra,kor_slot)]", ctx.gravity_exclude_cells == gravity_exclude_cells)

n_eligible = count(ctx.q_tilde .!= 0.0)
n_expected = ctx.D * ctx.D_dest - ctx.D_dest - 1   # minus diagonal (D_dest cells), minus the 1 BK cell
check("eligible cell count matches expectation ($n_eligible == $n_expected)", n_eligible == n_expected)
check("q_tilde[bra_idx, kor_slot] == 0 exactly (masked)", ctx.q_tilde[bra_idx, kor_slot] == 0.0)

pe = build_pivot_elimination(ctx)
pivot_o = ((pe.pivot_lin - 1) % ctx.D) + 1
pivot_d = ((pe.pivot_lin - 1) ÷ ctx.D) + 1
println("pivot cell (o=$pivot_o, d=$pivot_d), pivot_lin=$(pe.pivot_lin)")
check("Brazil->Korea is NOT selected as pivot", (pivot_o, pivot_d) != (bra_idx, kor_slot))
check("pivot cell itself remains eligible (q_tilde nonzero there)", ctx.q_tilde[pivot_o, pivot_d] != 0.0)

# find (bra_idx, kor_slot)'s position within pe.other_idx (it's excluded from the gravity equality,
# but remains an ordinary free A_od outer coordinate -- NOT removed from other_idx, since only the
# single argmax pivot cell is removed)
bk_lin = (kor_slot - 1) * ctx.D + bra_idx
bk_k = findfirst(==(bk_lin), pe.other_idx)
check("Brazil->Korea cell is an ordinary free (non-pivot) coordinate", bk_k !== nothing)
check("gravity coefficient c[Brazil,Korea] == 0 exactly", pe.c[bk_lin] == 0.0)

# ---- §7 test 1: calibration + several random valid free-coordinate vectors -> machine-zero residual ----
z_calib_full = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*ctx.D_dest], ctx.D, ctx.D_dest))
z_free_calib = pivot_reduce(z_calib_full, pe)
resid_calib = gravity_from_logz(pivot_expand(z_free_calib, pe), ctx)
check("calibration point: masked gravity residual is machine-zero ($resid_calib)", abs(resid_calib) < 1e-10)

exclusion_rows = Vector{NamedTuple}()
push!(exclusion_rows, (label = "calibration", residual = resid_calib, pass = abs(resid_calib) < 1e-10))

rng = MersenneTwister(20260731)
n_free = length(z_free_calib)
for trial in 1:5
    z_free_rand = z_free_calib .+ 0.05 .* randn(rng, n_free)
    resid = gravity_from_logz(pivot_expand(z_free_rand, pe), ctx)
    ok = abs(resid) < 1e-9
    check("random valid free-coordinate vector #$trial: machine-zero residual ($resid)", ok)
    push!(exclusion_rows, (label = "random_valid_$trial", residual = resid, pass = ok))
end

# ---- §7 test 2: perturb ONLY A[Brazil,Korea] -> pivot unchanged, residual unchanged, derivative zero ----
z_free_base = copy(z_free_calib)
pivot_base = pivot_expand(z_free_base, pe)[pivot_o, pivot_d]
resid_base = gravity_from_logz(pivot_expand(z_free_base, pe), ctx)
for δ in (0.01, -0.03, 0.5)
    z_free_pert = copy(z_free_base)
    z_free_pert[bk_k] += δ
    full_pert = pivot_expand(z_free_pert, pe)
    pivot_pert = full_pert[pivot_o, pivot_d]
    resid_pert = gravity_from_logz(full_pert, ctx)
    ok_pivot = pivot_pert == pivot_base   # exact: c[bk]=0 means the pivot formula literally doesn't move
    ok_resid = abs(resid_pert - resid_base) < 1e-9
    check("perturb A[Brazil,Korea] by δ=$δ: pivot value unchanged (exact)", ok_pivot)
    check("perturb A[Brazil,Korea] by δ=$δ: masked residual unchanged", ok_resid)
    push!(exclusion_rows, (label = "perturb_BK_delta_$δ", residual = resid_pert, pass = ok_pivot && ok_resid))
end
analytic_deriv_bk = -pe.c[pe.other_idx[bk_k]] / pe.c[pe.pivot_lin]
check("pivot derivative w.r.t. A[Brazil,Korea] is exactly zero", analytic_deriv_bk == 0.0)

# ---- §7 test 3: perturb an INCLUDED nonpivot A_od -> pivot adjusts, residual stays zero, FD match ----
included_k = findfirst(k -> pe.c[pe.other_idx[k]] != 0.0, eachindex(pe.other_idx))
inc_lin = pe.other_idx[included_k]
inc_o = ((inc_lin - 1) % ctx.D) + 1
inc_d = ((inc_lin - 1) ÷ ctx.D) + 1
println("included non-pivot test cell: (o=$inc_o, d=$inc_d)")
h = 1e-6
z_free_p = copy(z_free_base); z_free_p[included_k] += h
z_free_m = copy(z_free_base); z_free_m[included_k] -= h
pivot_p = pivot_expand(z_free_p, pe)[pivot_o, pivot_d]
pivot_m = pivot_expand(z_free_m, pe)[pivot_o, pivot_d]
fd_deriv = (pivot_p - pivot_m) / (2h)
analytic_deriv_inc = -pe.c[pe.other_idx[included_k]] / pe.c[pe.pivot_lin]
check("included cell: pivot value DOES adjust", pivot_p != pivot_base)
resid_p = gravity_from_logz(pivot_expand(z_free_p, pe), ctx)
check("included cell perturbation: masked residual remains zero ($resid_p)", abs(resid_p) < 1e-9)
fd_ok = abs(fd_deriv - analytic_deriv_inc) < 1e-6 * max(1.0, abs(analytic_deriv_inc))
check("included cell: analytic derivative matches finite difference (analytic=$analytic_deriv_inc, fd=$fd_deriv)", fd_ok)

derivative_rows = [
    (cell = "Brazil_Korea_excluded", analytic = analytic_deriv_bk, fd = 0.0, pass = analytic_deriv_bk == 0.0),
    (cell = "included_nonpivot_o$(inc_o)_d$(inc_d)", analytic = analytic_deriv_inc, fd = fd_deriv, pass = fd_ok),
]

# ---- ROW/other-eligible-cells sanity ----
check("no eligible cell references destination ROW (D_dest=19 shape already excludes it)", ctx.D_dest == 19)
check("all non-diagonal, non-BK cells remain eligible", n_eligible == n_expected)

# ---- write deliverables ----
open(joinpath(_D4E, "..", "..", "GRAVITY_PIVOT_EXCLUSION_GATE_2026-07-31.csv"), "w") do io
    println(io, "label,residual,pass")
    for r in exclusion_rows
        println(io, "$(r.label),$(r.residual),$(r.pass)")
    end
end
open(joinpath(_D4E, "..", "..", "GRAVITY_PIVOT_DERIVATIVE_GATE_2026-07-31.csv"), "w") do io
    println(io, "cell,analytic_derivative,finite_difference,pass")
    for r in derivative_rows
        println(io, "$(r.cell),$(r.analytic),$(r.fd),$(r.pass)")
    end
end

println("\n=== SUMMARY ===")
println(length(FAILURES) == 0 ? "ALL PASS ($(length(exclusion_rows) + length(derivative_rows) + 10) checks)" : "FAILURES: $(length(FAILURES))")
for f in FAILURES
    println("  FAIL: ", f)
end
