# ============================================================================
# Section 1 + 2 measurement driver for winner_certificate.jl.
#
# Measures, at fixed U and (mu,sigma), for the WINNER-COMPUTATION component of a
# nearby-point evaluation (the part the certificate/coordinate-update replace):
#   - trusted full scan time / allocs (compute_winners_fast winner-finding)
#   - certificate time / allocs + certified/rescan/switch fractions, per step size
#   - coordinate-update time / allocs (1-cell and 2-cell steps)
#   - winners_from_certificate (winner+wval) vs build_compressed_factual (full)
# across step sizes spanning accepted / rejected-line-search / continuation
# regimes, plus a genuine coordinate (FD-gradient) probe.
#
# Also emits a machine-readable CSV.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
using BenchmarkTools, Random, Printf, LinearAlgebra, Statistics

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.5
BenchmarkTools.DEFAULT_PARAMETERS.samples = 400

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; W = size(ctx.U, 1)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]

nthreads = Threads.nthreads()
println("="^96)
println("WINNER-CERTIFICATE / COORDINATE-UPDATE BENCHMARK  (D=$D, W=$W, threads=$nthreads)")
println("="^96)

med_ms(b) = median(b).time / 1e6
alloc_kb(b) = median(b).memory / 1024

# ---- reference & baseline ----
xf0 = x_free_from_w(w_up40)
ref = build_winner_ref(xf0, ctx)
θ0 = CS.reconstruct_full(xf0, ctx.m)

# trusted full scan (winner-finding only)
b_full = @benchmark compute_winners_fast($θ0, $ctx)
full_ms = med_ms(b_full); full_kb = alloc_kb(b_full)
@printf("\n[baseline] trusted full scan compute_winners_fast : %.4f ms, %.1f KB/call\n", full_ms, full_kb)

# build reference cost (one-time, amortized over many nearby points)
b_ref = @benchmark build_winner_ref($xf0, $ctx)
@printf("[baseline] build_winner_ref (one-time preprocessing) : %.4f ms, %.1f KB\n", med_ms(b_ref), alloc_kb(b_ref))

# ---- Section 1: certificate over step sizes ----
println("\n" * "-"^96)
println("SECTION 1: winner-margin certificate vs full scan, by step size")
println("-"^96)
@printf("%-22s %8s %8s %8s %10s %10s %8s %8s\n",
        "regime (step)", "cert%", "rescan%", "switch%", "cert_ms", "full_ms", "speedup", "cert_KB")

rng = MersenneTwister(11)
csv = IOBuffer()
println(csv, "regime,step,cert_frac,rescan_frac,switch_frac,cert_ms,full_ms,speedup,cert_kb,full_kb,maxabs_delta")
regimes = [("accepted", 1e-3), ("accepted", 5e-3), ("line-search", 2e-2), ("line-search", 5e-2),
           ("continuation", 1e-1), ("continuation", 2e-1), ("far", 5e-1), ("far", 1.0)]
for (name, sz) in regimes
    # aggregate certified fractions over several random directions of this size
    tot_cert = 0; tot_res = 0; tot_sw = 0; tot_cells = 0; maxd = 0.0
    local xf_rep
    for r in 1:8
        w′ = w_up40 .+ sz .* randn(rng, length(w_up40))
        xf′ = x_free_from_w(w′)
        r == 1 && (xf_rep = xf′)
        _, st = certified_winner_update(ref, ctx, xf′)
        tot_cert += st.n_certified; tot_res += st.n_rescan; tot_sw += st.n_switched; tot_cells += st.n_cells
        maxd = max(maxd, st.maxabs_delta)
    end
    cfrac = tot_cert / tot_cells; rfrac = tot_res / tot_cells; sfrac = tot_sw / max(tot_res, 1)
    b_cert = @benchmark certified_winner_update($ref, $ctx, $xf_rep)
    cert_ms = med_ms(b_cert); cert_kb = alloc_kb(b_cert)
    sp = full_ms / cert_ms
    @printf("%-14s %-7.3g %7.1f%% %7.1f%% %7.1f%% %10.4f %10.4f %7.2fx %8.1f\n",
            name, sz, 100cfrac, 100rfrac, 100sfrac, cert_ms, full_ms, sp, cert_kb)
    println(csv, "$name,$sz,$cfrac,$rfrac,$sfrac,$cert_ms,$full_ms,$sp,$cert_kb,$full_kb,$maxd")
end

# ---- Section 2: coordinate updates (FD-gradient probe) ----
println("\n" * "-"^96)
println("SECTION 2: coordinate-update specialization vs full scan (single-coordinate FD probe)")
println("-"^96)
# a direct coord (1 A-cell + pivot cell => usually 2 cells, 1-2 dests) and gamma' (0 cells)
h = 1e-6
wout = Matrix{Int}(undef, W, D)
# gamma' coordinate (coord 1): no A cell changes -> winners identical, O(0) A work
θ_g = CS.reconstruct_full(x_free_from_w([w_up40[1]+h; w_up40[2:end]]), ctx.m)
b_gamma = @benchmark coord_winner_update!($wout, $ref, $ctx, $θ_g, $(affected_cells(pe, 1)))
# a representative z_free coordinate (coord 5): direct + pivot cells (2 cells)
w5 = copy(w_up40); w5[5] += h
θ_5 = CS.reconstruct_full(x_free_from_w(w5), ctx.m)
cells5 = affected_cells(pe, 5)
b_c5 = @benchmark coord_winner_update!($wout, $ref, $ctx, $θ_5, $cells5)
@printf("gamma' coord (0 A-cells): %.4f ms, %.1f KB   (cells=%s)\n", med_ms(b_gamma), alloc_kb(b_gamma), affected_cells(pe,1))
@printf("z_free coord (2 A-cells): %.4f ms, %.1f KB   (cells=%s)  vs full %.4f ms  => %.1fx\n",
        med_ms(b_c5), alloc_kb(b_c5), cells5, full_ms, full_ms/med_ms(b_c5))
# count how many cells / dests a coordinate touches (audit of the O(1)/O(D) tiers)
ncells = [length(affected_cells(pe, c)) for c in 1:length(w_up40)]
ndests = [length(unique(last.(affected_cells(pe, c)))) for c in 1:length(w_up40)]
@printf("coordinate audit: A-cells touched per coord = %s\n", ncells)
@printf("                  destinations touched per coord = %s (max %d)\n", ndests, maximum(ndests))

# ---- winner+wval (moment-build component) apples-to-apples ----
println("\n" * "-"^96)
println("moment-build winner component: winners_from_certificate vs build_compressed_factual (full)")
println("-"^96)
w_mb = w_up40 .+ 2e-2 .* randn(MersenneTwister(3), length(w_up40))
xf_mb = x_free_from_w(w_mb)
θ_mb = CS.reconstruct_full(xf_mb, ctx.m)
b_wc = @benchmark winners_from_certificate($ref, $ctx, $xf_mb)
b_cf = @benchmark build_compressed_factual($θ_mb, $ctx)
@printf("winners_from_certificate : %.4f ms, %.1f KB\n", med_ms(b_wc), alloc_kb(b_wc))
@printf("build_compressed_factual : %.4f ms, %.1f KB   => %.2fx\n",
        med_ms(b_cf), alloc_kb(b_cf), med_ms(b_cf)/med_ms(b_wc))

# ---- write CSV ----
outdir = joinpath(@__DIR__, "..", "..", "results", "winner_certificate")
mkpath(outdir)
open(joinpath(outdir, "bench_section12.csv"), "w") do io
    write(io, String(take!(csv)))
end
println("\nCSV -> ", joinpath(outdir, "bench_section12.csv"))
println("="^96)
