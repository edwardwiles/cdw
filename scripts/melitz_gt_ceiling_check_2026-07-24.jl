# Quick, cheap (no KNITRO) check: is the "wall" beyond which Delta blows up simply the
# theoretical GT ceiling lambda_dd^(1/(sigma-1)) (achieved only as Delta->infinity, per the
# paper's own theorem), rather than a numerical/W-conditioning artifact?
using DelimitedFiles

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
    return calib, lambdaData, focal
end

kappa_of_g(g::Real, calib) = (calib.w_prime / calib.w[calib.target_country]) * exp(g)^(1 / (calib.sigma - 1))

calib, lambdaData, focal = load_calibration()
lambda_dd = lambdaData[focal, focal]
sigma = calib.sigma
kappa_min = lambda_dd^(1 / (sigma - 1))   # the FLOOR on kappa (achieved as Delta->infinity)
GT_ceiling = 1 - kappa_min                # the CEILING on GT = 1-kappa (NOT kappa_min itself)

println("focal country index (fra) = $focal")
println("lambda_dd (domestic trade share, from pi.csv) = $lambda_dd")
println("sigma = $sigma")
println("kappa_min = lambda_dd^(1/(sigma-1)) = $kappa_min")
println("theoretical GT ceiling = 1 - kappa_min = $GT_ceiling")
println()

gs = [-0.4188, -0.497333, -0.49783321, -0.50783, -0.51783, -0.52783, -0.53783, -0.54783, -0.55783, -0.547083]
println("g            gamma_prime   kappa       GT=1-kappa   GT/ceiling")
for g in gs
    kappa = kappa_of_g(g, calib)
    GT = 1 - kappa
    println(rpad(string(round(g, digits=6)), 12), " ", rpad(string(round(exp(g), digits=6)), 12), " ",
        rpad(string(round(kappa, digits=6)), 11), " ", rpad(string(round(GT, digits=6)), 12), " ",
        round(100 * GT / GT_ceiling, digits=2), "%")
end

g_at_ceiling = log((kappa_min / (calib.w_prime / calib.w[calib.target_country]))^(sigma - 1))
println("\ng at which GT = ceiling exactly (Delta -> infinity limit): g = $g_at_ceiling")
