# Micro-benchmark of the packed-write loop at the real L=10 dimensions, isolating the three
# candidate causes of the 86 s the profiler reported.
using Printf
const NROW = 15570
const NCORE = 382
const N = NCORE + NROW

mutable struct FakeEvalResult; hess::Vector{Float64}; end

# (1) PRODUCTION PATTERN: loop lives in a function; `res` arrives as an untyped argument, exactly as
#     the KNITRO callback's `evalResult` does. Julia specializes on the concrete argument type.
function pack_in_function(res, HRR, hee, HEQ, n, ncore)
    hess = res.hess
    Threads.@threads :dynamic for i in 1:n
        k = (i - 1) * n - div((i - 1) * (i - 2), 2)
        @inbounds if i <= ncore
            for j in i:ncore; k += 1; hess[k] = hee[1]; end
            for j in ncore+1:n; k += 1; hess[k] = HEQ[i, j - ncore]; end
        else
            ii = i - ncore
            @simd for j in i:n
                hess[k+j-i+1] = HRR[j - ncore, ii]
            end
        end
    end
end

# (2) Same, but WITHOUT hoisting `res.hess` -- the thing I "fixed".
function pack_no_hoist(res, HRR, hee, HEQ, n, ncore)
    Threads.@threads :dynamic for i in 1:n
        k = (i - 1) * n - div((i - 1) * (i - 2), 2)
        @inbounds if i <= ncore
            for j in i:ncore; k += 1; res.hess[k] = hee[1]; end
            for j in ncore+1:n; k += 1; res.hess[k] = HEQ[i, j - ncore]; end
        else
            ii = i - ncore
            for j in i:n
                res.hess[k+j-i+1] = HRR[j - ncore, ii]
            end
        end
    end
end

# (3) Serial + ROW walk (the original production loop), in a function.
function pack_serial_rowwalk(res, HRR, hee, HEQ, n, ncore)
    hess = res.hess
    k = 0
    @inbounds for i in 1:n
        if i <= ncore
            for j in i:ncore; k += 1; hess[k] = hee[1]; end
            for j in ncore+1:n; k += 1; hess[k] = HEQ[i, j - ncore]; end
        else
            for j in i:n
                k += 1
                hess[k] = HRR[i - ncore, j - ncore]   # ROW of a column-major matrix
            end
        end
    end
end

println("allocating: HRR $(NROW)^2 = $(round(NROW^2*8/2^30, digits=2)) GiB, hess $(div(N*(N+1),2)) = $(round(div(N*(N+1),2)*8/2^30, digits=2)) GiB")
HRR = zeros(NROW, NROW); hee = zeros(NCORE*(NCORE+1)÷2); HEQ = zeros(NCORE, NROW)
res = FakeEvalResult(Vector{Float64}(undef, div(N*(N+1), 2)))
println("threads = ", Threads.nthreads())

for (name, f) in (("(1) function + hoisted + threaded + col-walk", pack_in_function),
                  ("(2) function + NOT hoisted + threaded", pack_no_hoist),
                  ("(3) function + serial + ROW-walk (original)", pack_serial_rowwalk))
    f(res, HRR, hee, HEQ, N, NCORE)                 # warm up / compile
    t = @elapsed f(res, HRR, hee, HEQ, N, NCORE)
    @printf("%-46s %8.2f s\n", name, t)
end

# (4) THE PROFILER'S PATTERN: same loop at TOP-LEVEL scope reading NON-CONST GLOBALS.
GH = HRR; GE = HEQ; Ghee = hee; Gn = N; Gncore = NCORE
gv = Vector{Float64}(undef, div(N*(N+1), 2))
t_global = @elapsed begin
    local k = 0
    for i in 1:Gn
        if i <= Gncore
            for j in i:Gncore; k += 1; gv[k] = Ghee[1]; end
            for j in Gncore+1:Gn; k += 1; gv[k] = GE[i, j - Gncore]; end
        else
            for j in i:Gn
                k += 1
                gv[k] = GH[i - Gncore, j - Gncore]
            end
        end
    end
end
@printf("%-46s %8.2f s\n", "(4) TOP-LEVEL scope, non-const globals", t_global)
