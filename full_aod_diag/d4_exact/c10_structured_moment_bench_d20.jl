# ============================================================================
# Continuation 10, Part 2: D=20/W=80,000 three-way moment-construction
# comparison:
#   (a) generic dense construction -- obj.moments! = EK_moments_gammanorm_directgp!
#       (production baseline, nested loop via hFunction!/hFunctionCounter!,
#       evaluates ALL D origins' sigma-values then zeros losers)
#   (b) structured rank-one (BLAS.ger! or broadcast) + winner-scatter
#       (structured_moment_build.jl, THIS task's new construction)
#   (c) existing compressed representation (build_compressed_factual, prior
#       Continuation 8/9 work) + materialize_dense_factual! (also prior work,
#       a single-pass nested loop reconstruction from the compressed form)
#
# Measures BOTH isolated moment-construction time alone, AND complete
# cold-inner-solve time using each construction as the front end feeding the
# SAME dense-Hessian KNITRO inner solve (Hessian/FG callbacks entirely
# UNCHANGED across all four variants -- oracle_fast.jl's inner_loop_KNITRO_profiled,
# hessopt=exact + full H_copy materialize-then-gemm, per this task's own
# "do not touch the production default" instruction; only the moment-build
# FRONT END differs).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c10_structured_moment_bench_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_structured_moment_bench_d20.jl starting at ", now(), "  nthreads=", Threads.nthreads())

function vmhwm_gb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2]) / 1e6
    end
    return -1.0
end

t0 = time()
ctx = d20_real_setup(W = 80000)
logprint("d20_real_setup(W=80000) wall = ", round(time() - t0, digits = 2), "s  VmHWM=", round(vmhwm_gb(), digits = 2), " GB")
obj = ctx.obj
D = ctx.D; W = size(obj.U, 1)
xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)
ncol = obj.outer_constr_index - 1
logprint("D=", D, " W=", W, " ncol=", ncol)

# ============================================================================
# Part 1: isolated moment-construction timing + correctness
# ============================================================================
logprint("\n", "="^90)
logprint("Isolated moment-construction timing (median of N reps), + correctness vs dense")
logprint("="^90)

N_REPS = 5

# (a) dense -- warm up then time
K_dense = zeros(W); G_dense = zeros(W, obj.d)
obj.moments!(K_dense, G_dense, θ_full, obj.U, obj)   # warmup
t_dense = Float64[]
for _ in 1:N_REPS
    t = @elapsed obj.moments!(K_dense, G_dense, θ_full, obj.U, obj)
    push!(t_dense, t)
end
med_dense = median(t_dense)
logprint(@sprintf("  (a) dense (obj.moments!):              median=%.4fs  reps=%s", med_dense, round.(t_dense, digits=4)))
G_dense_inner = G_dense[:, 1:ncol]

# build cf ONCE for (b)/(c) correctness (shared winner-search) -- timed separately below
cf = build_compressed_factual(θ_full, ctx; check_ties = true)
logprml = cf.n_tied
logprint("  n_tied (real D=20 data) = ", cf.n_tied)

# (b) structured, ger! variant -- includes build_compressed_factual (winner search) in the timed cost,
#     since that IS the moment-construction cost for this variant (winner search is irreducible, O(W*D^2),
#     but skips evaluating LOSERS' sigma-values, per compressed_moments.jl's own header).
G_struct = zeros(W, ncol)
a_coef, FixedCol = structured_coeffs(cf)
structured_fill_chunk!(G_struct, cf, a_coef, FixedCol, 1:W; use_ger = true)   # warmup
t_struct_ger_full = Float64[]   # winner-search + fill
t_struct_ger_fillonly = Float64[]   # fill only, reusing cf
for _ in 1:N_REPS
    t1 = @elapsed begin
        cf_i = build_compressed_factual(θ_full, ctx; check_ties = true)
        a_i, FC_i = structured_coeffs(cf_i)
        structured_fill_chunk!(G_struct, cf_i, a_i, FC_i, 1:W; use_ger = true)
    end
    push!(t_struct_ger_full, t1)
    t2 = @elapsed structured_fill_chunk!(G_struct, cf, a_coef, FixedCol, 1:W; use_ger = true)
    push!(t_struct_ger_fillonly, t2)
end
logprint(@sprintf("  (b1) structured+ger! (incl winner search): median=%.4fs  reps=%s", median(t_struct_ger_full), round.(t_struct_ger_full, digits=4)))
logprint(@sprintf("  (b1) structured+ger! (fill only, cf reused): median=%.4fs  reps=%s", median(t_struct_ger_fillonly), round.(t_struct_ger_fillonly, digits=4)))

# (b2) structured, broadcast variant
G_struct_b = zeros(W, ncol)
structured_fill_chunk!(G_struct_b, cf, a_coef, FixedCol, 1:W; use_ger = false)   # warmup
t_struct_bcast_fillonly = Float64[]
for _ in 1:N_REPS
    t2 = @elapsed structured_fill_chunk!(G_struct_b, cf, a_coef, FixedCol, 1:W; use_ger = false)
    push!(t_struct_bcast_fillonly, t2)
end
logprint(@sprintf("  (b2) structured+broadcast (fill only, cf reused): median=%.4fs  reps=%s", median(t_struct_bcast_fillonly), round.(t_struct_bcast_fillonly, digits=4)))

# (c) compressed + materialize_dense_factual!
G_matref = zeros(W, ncol)
materialize_dense_factual!(G_matref, cf)   # warmup
t_matref_full = Float64[]
t_matref_fillonly = Float64[]
for _ in 1:N_REPS
    t1 = @elapsed begin
        cf_i = build_compressed_factual(θ_full, ctx; check_ties = true)
        materialize_dense_factual!(G_matref, cf_i)
    end
    push!(t_matref_full, t1)
    t2 = @elapsed materialize_dense_factual!(G_matref, cf)
    push!(t_matref_fillonly, t2)
end
logprint(@sprintf("  (c) compressed+materialize (incl winner search): median=%.4fs  reps=%s", median(t_matref_full), round.(t_matref_full, digits=4)))
logprint(@sprintf("  (c) compressed+materialize (fill only, cf reused): median=%.4fs  reps=%s", median(t_matref_fillonly), round.(t_matref_fillonly, digits=4)))

# correctness
d_b_dense = maximum(abs.(G_struct .- G_dense_inner))
d_c_dense = maximum(abs.(G_matref .- G_dense_inner))
d_b_c = maximum(abs.(G_struct .- G_matref))
d_b_bcast = maximum(abs.(G_struct .- G_struct_b))
logprint(@sprintf("\n  max|structured - dense|       = %.3e  (bit_identical=%s)", d_b_dense, G_struct == G_dense_inner))
logprint(@sprintf("  max|compressed_mat - dense|   = %.3e  (bit_identical=%s)", d_c_dense, G_matref == G_dense_inner))
logprint(@sprintf("  max|structured - compressed_mat| = %.3e  (bit_identical=%s)", d_b_c, G_struct == G_matref))
logprint(@sprintf("  max|structured(ger) - structured(bcast)| = %.3e  (bit_identical=%s)", d_b_bcast, G_struct == G_struct_b))

logprint(@sprintf("\n  SPEEDUP vs dense (fill-only / winner-search-reused basis): structured=%.3fx  compressed_mat=%.3fx",
          med_dense / median(t_struct_ger_fillonly), med_dense / median(t_matref_fillonly)))
logprint(@sprintf("  SPEEDUP vs dense (full, incl own winner search): structured=%.3fx  compressed_mat=%.3fx",
          med_dense / median(t_struct_ger_full), med_dense / median(t_matref_full)))

# ============================================================================
# Part 2: complete cold inner-solve using each construction as the front end
#         (SAME dense-Hessian KNITRO solve for all four -- oracle_fast.jl,
#         completely unchanged; only obj.H's K/G columns are populated
#         differently before the solve)
# ============================================================================
logprint("\n", "="^90)
logprint("Complete cold inner-solve, each moment-construction as front end (dense-Hessian solve UNCHANGED)")
logprint("="^90)

function solve_with_builder!(obj, θ_full, ctx, builder::Symbol)
    # NOTE: obj.d (=402 at D=20) is the FULL moment-column count, but the
    # INNER dual solve only ever reads columns 1:oci-1 (=401, the bilateral
    # D^2 + counterfactual-price-index block CompressedFactual covers) --
    # obj.outer_constr_index == obj.d exactly at this ctx (asserted in
    # context_real_d20.jl), so the one REMAINING column (obj.d itself, the
    # gravity moment) is read only by outer-loop constraint recovery, never
    # by inner_loop_KNITRO_profiled's FG/Hessian callbacks (which never pass
    # `constr`). Filled anyway below for full production fidelity, reusing
    # compressed_live.jl's existing `compressed_gravity_raw`/
    # `fill_gravity_column!` (NOT re-derived) -- matches that file's own
    # documented reasoning for why this one column sits outside the
    # compressed/structured winner-form representation (a single
    # draw-independent scalar, not O(W*D^2) work either way).
    W = size(obj.U, 1)
    if builder === :dense
        obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_full, obj.U, obj)
    elseif builder === :structured_ger || builder === :structured_bcast
        cf_i = build_compressed_factual(θ_full, ctx; check_ties = true)
        ncolI = cf_i.oci - 1
        a_i, FC_i = structured_coeffs(cf_i)
        structured_fill_chunk!(@view(obj.H[:, 3:2+ncolI]), cf_i, a_i, FC_i, 1:W; use_ger = (builder === :structured_ger))
        fill_gravity_column!(obj, compressed_gravity_raw(θ_full, ctx))
        fill_K_directgp!(@view(obj.H[:, 1]), θ_full, ctx)
    elseif builder === :compressed_materialize
        cf_i = build_compressed_factual(θ_full, ctx; check_ties = true)
        ncolI = cf_i.oci - 1
        materialize_dense_factual!(@view(obj.H[:, 3:2+ncolI]), cf_i)
        fill_gravity_column!(obj, compressed_gravity_raw(θ_full, ctx))
        fill_K_directgp!(@view(obj.H[:, 1]), θ_full, ctx)
    else
        error("unknown builder $builder")
    end
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_profiled(obj)
    if nStatus in (0, -100, -101, -103)
        obj.x .= x
    else
        obj.x .= NaN
    end
    return obj.H_save, x, nStatus, n_fg, n_hess
end

obj.x .= NaN
t0 = time(); K_ref, x_ref, status_ref, nfg_ref, nhess_ref = solve_with_builder!(obj, θ_full, ctx, :dense); t_ref = time() - t0
logprint(@sprintf("  (a) dense:                cold status=%d n_fg=%d n_hess=%d wall=%.3fs K=%.10f", status_ref, nfg_ref, nhess_ref, t_ref, K_ref))

for (label, sym) in [("structured+ger!", :structured_ger), ("structured+broadcast", :structured_bcast), ("compressed+materialize", :compressed_materialize)]
    obj.x .= NaN
    t0 = time(); K_v, x_v, status_v, nfg_v, nhess_v = solve_with_builder!(obj, θ_full, ctx, sym); t_v = time() - t0
    dK = abs(K_v - K_ref); dx = maximum(abs.(x_v .- x_ref))
    logprint(@sprintf("  (%-22s) cold status=%d(ref %d) n_fg=%d(ref %d) n_hess=%d(ref %d) wall=%.3fs(ref %.3fs) speedup=%.3fx |dK|=%.3e max|dx|=%.3e",
              label, status_v, status_ref, nfg_v, nfg_ref, nhess_v, nhess_ref, t_v, t_ref, t_ref / t_v, dK, dx))
end

# ============================================================================
# Part 3: JIT-order control. Same lesson as Part 1's chunked-Hessian sweep --
# the ABOVE comparison ran :dense strictly FIRST in this process, so it alone
# pays first-call JIT for `inner_loop_KNITRO_profiled`/its callbacks (SHARED
# by all four builders -- only the moment-build FRONT END differs; the
# dual-solve function itself is the exact same compiled code for all four).
# Re-run cold solves for all four builders AGAIN, now that every one of them
# is already JIT-warm from the pass above, to isolate the genuine
# moment-build-time difference from JIT-compilation-order noise.
# ============================================================================
logprint("\n", "="^90)
logprint("Part 3: JIT-order control -- re-run all four cold solves now that everything is JIT-warm")
logprint("="^90)
for (label, sym) in [(:dense, :dense), (:structured_ger, :structured_ger), (:structured_bcast, :structured_bcast), (:compressed_materialize, :compressed_materialize)]
    obj.x .= NaN
    t0 = time(); K_v, x_v, status_v, nfg_v, nhess_v = solve_with_builder!(obj, θ_full, ctx, sym); t_v = time() - t0
    logprint(@sprintf("  %-22s cold(JIT-warm) status=%d n_fg=%d n_hess=%d wall=%.3fs", label, status_v, nfg_v, nhess_v, t_v))
end

logprint("\nVmHWM at end = ", round(vmhwm_gb(), digits = 2), " GB")
logprint("\nc10_structured_moment_bench_d20.jl COMPLETE at ", now())
close(LOGIO)
