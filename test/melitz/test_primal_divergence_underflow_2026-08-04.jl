# 2026-08-04 verifier false-rejection fix regression + BigFloat validation.
#
# Governing task: prove that melitz_primal_divergence's fix (src/melitz/delta_star.jl, commit
# 1d97e4c) is mathematically correct, not just "makes a symptom go away" -- and that genuinely
# invalid points are still rejected. No KNITRO/real-data dependency: melitz_primal_divergence
# is a pure function of a weights vector, so a synthetic deterministic fixture is sufficient
# and lets this run in seconds rather than minutes.
#
# Fixture: arg0 values spanning from well within Float64 range down to arg0=-1100 (matching
# the live-confirmed extreme-tail range in the 2026-08-04 audit report), i.e. dPsi!(arg0) =
# exp(arg0) spans from a normal Float64 down through subnormal into literal 0.0 well before
# the true mathematical value reaches zero. BigFloat is used as the independent
# arbitrary-precision reference: BigFloat can represent exp(-1100) as a genuinely tiny but
# nonzero positive number (its exponent range is unbounded, unlike Float64's ~[-1022,1023]),
# so BigFloat weights are the ground truth for "is this really positive."

using Test
using Printf

const _PDU_REPO = joinpath(@__DIR__, "..", "..")
const _PDU_MELITZ = joinpath(_PDU_REPO, "src", "melitz")
# Minimal prefix of include_melitz.jl's own load order needed to define
# melitz_primal_divergence (delta_star.jl) without pulling in cc_algo/KNITRO at all --
# melitz_primal_divergence is a pure function of a weights vector and does not touch KNITRO.
for f in ("profiling.jl", "knitro_compat.jl", "backend_config.jl", "run_diagnostics.jl",
          "types.jl", "gravity_sample.jl", "bounded_cache.jl", "inner_solve_policy.jl",
          "pareto.jl", "firm_quantities.jl", "equilibrium.jl", "moments.jl", "sorted_tail.jl",
          "sorted_dual_argument.jl", "moment_operator.jl", "delta_star.jl")
    include(joinpath(_PDU_MELITZ, f))
end

# --- The OLD (buggy) formula, reproduced verbatim from before commit 1d97e4c, for comparison ---
function _old_melitz_primal_divergence(weights::AbstractVector, W::Int)
    e = exp(1)
    acc = 0.0
    @inbounds for p in weights
        m = p * W
        if !(m > 0) || !isfinite(m)
            return Inf
        elseif m <= e
            acc += m * log(m) - m + 1
        else
            acc += m^2 / (2e) - e / 2 + 1
        end
    end
    return acc / W
end

@testset "melitz_primal_divergence underflow false-rejection fix (2026-08-04)" begin

    @testset "Case B: unavoidable post-normalization underflow — fixed formula agrees with BigFloat" begin
        # W=1000 synthetic draws. arg0 ranges from -5 (a normal, well-conditioned weight) down
        # to -1100 (genuinely below Float64's ~-745 exp() underflow threshold), evenly spaced,
        # matching the live report's "-800 to -1100" extreme-tail range.
        W = 1000
        arg0 = collect(range(-5.0, -1100.0; length=W))

        # Float64 materialization (production code path: dPsi! is exp(arg0) for arg0<=1).
        LFD64 = exp.(arg0)
        s64 = sum(LFD64)
        weights64 = LFD64 ./ s64
        n_zero = count(==(0.0), weights64)

        @test n_zero > 0   # the fixture must actually exercise underflow-to-zero
        @printf("  Float64 materialized zeros: %d / %d\n", n_zero, W)

        # BigFloat ground truth: exp() at BigFloat precision does not underflow across this
        # range at all (BigFloat's exponent range vastly exceeds Float64's), so every entry is
        # a genuine, distinguishable positive number.
        setprecision(BigFloat, 256) do
            arg0_bf = BigFloat.(arg0)
            LFD_bf = exp.(arg0_bf)
            @test all(x -> x > 0, LFD_bf)   # ground truth: EVERY weight is mathematically positive
            s_bf = sum(LFD_bf)
            weights_bf = LFD_bf ./ s_bf

            # The entries that materialized to Float64 0.0 are still genuinely positive in BigFloat.
            zero_idxs = findall(==(0.0), weights64)
            @test !isempty(zero_idxs)
            @test all(i -> weights_bf[i] > 0, zero_idxs)
            # ... and their true mass is astronomically below Float64's smallest subnormal
            # (5e-324), i.e. this is genuinely UNAVOIDABLE underflow (Case B), not an avoidable
            # common-offset shift (Case A) -- shifting by the max (arg0=-5) would not rescue
            # arg0=-1100, since exp(-1100 - (-5)) = exp(-1095) is still far below Float64 range.
            @test all(i -> weights_bf[i] < 1e-300, zero_idxs)

            # Stable weighted-moment check: a moment computed from the Float64 weights (which
            # silently drop the underflowed entries' contribution, since they're exactly 0)
            # must still agree with the BigFloat moment to high precision -- proving the omitted
            # mass is numerically negligible, not silently wrong.
            g = collect(1.0:W)  # arbitrary test moment vector
            moment64 = sum(weights64 .* g)
            moment_bf = sum(weights_bf .* BigFloat.(g))
            @test abs(Float64(moment_bf) - moment64) < 1e-9

            # --- The actual point of this test: primal divergence ---
            W_int = W
            div_fixed = melitz_primal_divergence(weights64, W_int)
            div_old = _old_melitz_primal_divergence(weights64, W_int)

            @test isfinite(div_fixed)          # FIXED formula: finite, as it mathematically must be
            @test div_old == Inf               # OLD formula: incorrectly Inf — this is the bug, reproduced

            # BigFloat reference divergence, computed with the SAME piecewise phi(m) definition
            # applied at full precision (no m==0 special case needed — BigFloat m is never
            # exactly 0 here, so the ordinary m*log(m)-m+1 branch applies throughout).
            e_bf = exp(BigFloat(1))
            acc_bf = BigFloat(0)
            for p in weights_bf
                m = p * W_int
                acc_bf += m <= e_bf ? (m * log(m) - m + 1) : (m^2 / (2 * e_bf) - e_bf / 2 + 1)
            end
            div_bf = Float64(acc_bf / W_int)

            @test isapprox(div_fixed, div_bf; atol=1e-6, rtol=1e-6)
            @printf("  div_fixed=%.10f  div_bf=%.10f  div_old=%s\n", div_fixed, div_bf, string(div_old))
        end
    end

    @testset "Known-good point: fix is byte-identical to old formula when nothing underflows" begin
        W = 500
        arg0 = collect(range(-3.0, -0.5; length=W))  # no underflow anywhere
        LFD64 = exp.(arg0)
        weights64 = LFD64 ./ sum(LFD64)
        @test count(==(0.0), weights64) == 0
        @test melitz_primal_divergence(weights64, W) == _old_melitz_primal_divergence(weights64, W)
    end

    @testset "Genuinely invalid points are still rejected (Inf), unaffected by the fix" begin
        W = 100
        # Negative weight (invalid recovery / broken conjugate-domain state).
        w_neg = fill(1.0 / W, W); w_neg[7] = -1e-6
        @test melitz_primal_divergence(w_neg, W) == Inf

        # NaN weight.
        w_nan = fill(1.0 / W, W); w_nan[13] = NaN
        @test melitz_primal_divergence(w_nan, W) == Inf

        # Inf weight (should not occur post-normalization but must still be caught).
        w_inf = fill(1.0 / W, W); w_inf[42] = Inf
        @test melitz_primal_divergence(w_inf, W) == Inf

        # All-zero (total degeneracy) is finite under the fixed formula (each contributes the
        # correct limit value 1.0) — this is intentional: melitz_recover_lfd's OWN upstream
        # gate (`s > 0` on the raw pre-normalization sum) is what rejects total degeneracy, not
        # melitz_primal_divergence itself, which is a pure per-element phi-divergence function.
        w_allzero = zeros(W)
        @test melitz_primal_divergence(w_allzero, W) == 1.0
    end

    @testset "Old verifier fails the same valid fixture the fixed one accepts (regression demo)" begin
        W = 1000
        arg0 = collect(range(-5.0, -1100.0; length=W))
        weights64 = exp.(arg0) ./ sum(exp.(arg0))
        @test _old_melitz_primal_divergence(weights64, W) == Inf     # OLD: false rejection
        @test isfinite(melitz_primal_divergence(weights64, W))       # FIXED: correctly accepted
    end
end
