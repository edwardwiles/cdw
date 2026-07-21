# Production integration continuation, Section 9: exclusive/inclusive timing breakdown of the
# Architecture C structured Hessian callback (`hessian_cm_structured!`, cm_hessian_architectures.jl)
# at D20/L=50/W=80000 -- the production (cm_basis=:cumulative, cm_hessian_backend=:structured)
# configuration. Identifies which portion explains the ~3.6-4s Hessian callback previously observed.
#
# `hessian_cm_structured!` itself is NOT modified (stays exactly as production uses it) -- this
# file defines `hessian_cm_structured_profiled!`, a byte-for-byte functional clone with `@prof`
# wrapped around each of the brief's 7 applicable stages (this codebase's Architecture C has no
# separate "transfer into KNITRO" or "factorization" stage of its own to time -- packing into `h`
# IS the transfer, and Newton/KKT factorization happens inside KNITRO's C library, unobservable
# from Julia; both are called out explicitly below rather than silently omitted).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics

"""
    hessian_cm_structured_profiled!(h, obj, cctx) -> h

Functional clone of `hessian_cm_structured!` (cm_hessian_architectures.jl:301-369), split into
`@prof`-timed stages. Any change to the production function must be mirrored here by hand -- this
is a diagnostic instrument, not a code-reuse abstraction (the brief asks for a profile of the
EXISTING callback's stages, not a refactor of it).
"""
function hessian_cm_structured_profiled!(h, obj, cctx::CMBinHessCtx)
    @prof "0_TOTAL_inclusive" begin
        @unpack H, M, arg0, arg2, ddPsi! = obj
        @prof "1_draw_level_weight_construction_ddPsi" begin
            ddPsi!(arg2, arg0)
        end
        w = arg2
        NCORE = cctx.NCORE; ncm = cctx.ncm; L = cctx.L; nO = cctx.nO; D = cctx.D
        refIndex1 = cctx.refIndex1; origins = cctx.origins

        E = @view H[:, 2:1+NCORE]
        @prof "2_weighted_bin_contingency_accumulation" begin
            build_bin_tables!(cctx, E, w)
        end
        @prof "3_cumulative_prefix_sums" begin
            prefix_sum_tables!(cctx)
        end

        Hfull = cctx.Hfull
        fill!(Hfull, 0.0)

        @prof "6_core_economic_hessian_block_HEE" begin
            Ews = cctx.Ews
            @views Ews .= E .* sqrt.(w)
            HEE = @view Hfull[1:NCORE, 1:NCORE]
            BLAS.gemm!('T', 'N', 1 / M, Ews, Ews, 0.0, HEE)
        end

        @prof "5_economic_common_cross_block_HEC" begin
            CS_ = cctx.CScum
            Hraw_EC = Matrix{Float64}(undef, NCORE, nO)
            @inbounds for l in 1:L
                for (oi, o) in enumerate(origins)
                    for j in 1:NCORE
                        Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
                    end
                end
                cols = NCORE + (l-1)*nO + 1 : NCORE + l*nO
                block_ec = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
                @views Hfull[1:NCORE, cols] .= block_ec
                @views Hfull[cols, 1:NCORE] .= transpose(block_ec)
            end
        end

        @prof "4_common_common_block_HCC" begin
            CT = cctx.CT
            Hraw_CC = Matrix{Float64}(undef, nO, nO)
            @inbounds for l in 1:L
                for lp in 1:L
                    for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
                        Hraw_CC[oi, pi] = (CT[o, p, l, lp] - CT[o, refIndex1, l, lp] - CT[refIndex1, p, l, lp] + CT[refIndex1, refIndex1, l, lp]) / M
                    end
                    rows = NCORE + (l-1)*nO + 1 : NCORE + l*nO
                    cols = NCORE + (lp-1)*nO + 1 : NCORE + lp*nO
                    block = cctx.R === nothing ? Hraw_CC : (cctx.R' * Hraw_CC * cctx.R)
                    @views Hfull[rows, cols] .= block
                end
            end
        end

        @prof "7_symmetrize_and_pack" begin
            n = NCORE + ncm
            k = 1
            @inbounds for i in 1:n
                for j in i:n
                    h[k] = 0.5 * (Hfull[i, j] + Hfull[j, i])
                    k += 1
                end
            end
        end
        # NOTE (brief items 8/9): "transfer into KNITRO" is exactly the packing loop just timed
        # (h IS the KNITRO-owned buffer -- no separate copy/transfer step exists in this
        # architecture); "factorization/linear solve" happens inside KNITRO's C library after this
        # callback returns control, not observable from Julia-side instrumentation. Both noted
        # explicitly rather than fabricating a stage that doesn't exist in this code.
    end
    return h
end

W = 80000
DELTA = 1.0
L = 50
N_REPS = 30

println(">>> building D20 real-data context, W=$W ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
@printf ">>> ctx built in %.1fs. D=%d W=%d\n" (time() - t0) D W

x_free_calib = ctx.θ0_up[ctx.free_idx]

println(">>> building CM production context (L=$L, cumulative, structured, calibration point)...")
cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = L,
               cm_basis = :cumulative, cm_hessian_backend = :structured)
pcx = build_cm_production_context_v2(ctx, CS, cfg)
ctx_cm = pcx.ctx_cm
# build_cm_production_context_v2 doesn't expose cctx directly for the cumulative path (it's
# closed over inside pcx.hess_cb_builder) -- rebuild it explicitly here, identically, since this
# profiling script needs direct access to call hessian_cm_structured_profiled! itself.
cctx = build_cm_bin_ctx(ctx, pcx.aug)

println(">>> warm-solving at calibration to populate obj.arg0/arg1 at the true solution (production precondition)...")
base = cm_base_state_v2(x_free_calib, pcx)
println("    inner_status=", base.inner_status, " ζstar=", base.ζstar)

_archC_prep_for_hessian!(ctx_cm.obj, vcat(base.ζstar, base.λstar))

n = pcx.aug.ncore + pcx.aug.ncm
h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
@printf ">>> Hessian dimension n=%d (packed length %d), ncm=%d, NCORE=%d\n" n length(h) pcx.aug.ncm pcx.aug.ncore

println(">>> warmup call (JIT) ...")
hessian_cm_structured_profiled!(h, ctx_cm.obj, cctx)
prof_reset!()

println(">>> timed calls: n=$N_REPS ...")
for i in 1:N_REPS
    hessian_cm_structured_profiled!(h, ctx_cm.obj, cctx)
end

println()
println("="^110)
println("EXCLUSIVE + INCLUSIVE STAGE BREAKDOWN, D=$D L=$L W=$W, n_reps=$N_REPS")
println("="^110)
rows = prof_summary()
@printf "%-42s %6s %10s %10s %10s %10s %14s %10s\n" "label" "n" "median_s" "mean_s" "p90_s" "std_s" "mean_alloc_MB" "gc_s"
for r in rows
    @printf "%-42s %6d %10.5f %10.5f %10.5f %10.5f %14.2f %10.5f\n" r.label r.n r.median_s r.mean_s r.p90_s r.std_s (r.mean_alloc_bytes/1e6) r.mean_gc_s
end

total_row = only(filter(r -> r.label == "0_TOTAL_inclusive", rows))
println()
println("Exclusive-stage share of TOTAL median wall time:")
for r in rows
    r.label == "0_TOTAL_inclusive" && continue
    @printf "  %-42s %6.1f%%\n" r.label (100 * r.median_s / total_row.median_s)
end

mkpath(joinpath(@__DIR__, "..", "..", "docs"))
write_csv_rows(joinpath(@__DIR__, "..", "..", "docs", "fullA_archC_hessian_profile_d20_L$(L)_W$(W).csv"), rows)
println()
println("DONE")
