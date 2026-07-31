# Independent reproduction of the calibration gravity regression (theta_star), separate from
# prestep/master_prestep.jl's production helper, using the shared gravity eligibility mask.
#
# Regression: two-way (origin, destination) fixed-effects OLS of ln(pi) [trade share] on ln(tau)
# [trade cost], sample restricted to a mask; thetaHat = -sum(W_lnpi.*W_lntau)/sum(W_lntau.^2)
# (matches master_prestep.jl's estimator and its documented FWL equivalence to two-way-FE OLS).
# gravity_coefficient (the reported regression coefficient of ln(pi) on ln(tau)) = -thetaHat.
#
# Usage: julia --project=. full_aod_diag/d4_exact/gravity_regression_independent_check.jl [exclude_diagonal] [destination_sample]

using DelimitedFiles
include(joinpath(@__DIR__, "..", "..", "misc", "doubleDiff.jl"))

function resolve_country_index(countries::Vector{String}, name_or_alias::AbstractString)
    aliases = Dict(
        "brazil" => "bra", "bra" => "bra",
        "korea" => "kor", "south korea" => "kor", "korea, rep." => "kor",
        "republic of korea" => "kor", "kor" => "kor",
    )
    key = lowercase(strip(name_or_alias))
    iso3 = get(aliases, key, key)
    matches = findall(==(iso3), countries)
    length(matches) == 1 || error("resolve_country_index($name_or_alias): expected exactly one match for iso3='$iso3', found $(length(matches)) in $countries")
    return matches[1]
end

function run_check(; exclude_diagonal::Bool=true, destination_sample::Symbol=:exclude_row)
    realDataDir = joinpath(@__DIR__, "..", "..", "real_data", "noah_D20")
    countries = vec(readdlm(joinpath(realDataDir, "countries.csv"), ',', String))
    lambda = readdlm(joinpath(realDataDir, "pi.csv"), ',', Float64)
    tau = readdlm(joinpath(realDataDir, "tau.csv"), ',', Float64)
    D = size(lambda, 1)

    bra_idx = resolve_country_index(countries, "brazil")
    kor_idx = resolve_country_index(countries, "korea")
    row_idx = findfirst(==("row"), countries)
    @assert bra_idx != kor_idx "Brazil and Korea resolved to the same index"
    @assert row_idx !== nothing "no 'row' (ROW) entry found in countries.csv"

    named_dest = destination_sample == :exclude_row ? filter(!=(row_idx), 1:D) : collect(1:D)
    Ddest = length(named_dest)
    lambda_dest = lambda[:, named_dest]
    tau_dest = tau[:, named_dest]

    # dest-space column index for Korea (position of kor_idx within named_dest)
    kor_dest_col = findfirst(==(kor_idx), named_dest)
    @assert kor_dest_col !== nothing "Korea's index $kor_idx not present in named_dest=$named_dest"

    mask = trues(D, Ddest)
    if exclude_diagonal
        for o in 1:D, d in 1:Ddest
            mask[o, d] &= (o != d)
        end
    end
    mask[bra_idx, kor_dest_col] = false  # exclude Brazil->Korea from the gravity-identification sample
    n_eligible = count(mask)
    n_excluded = D * Ddest - n_eligible

    Wlambda = within_transform_masked(lambda_dest, mask)
    Wtau = within_transform_masked(tau_dest, mask)
    thetaHat = -sum(Wlambda .* Wtau) / sum(Wtau .* Wtau)
    gravity_coefficient = -thetaHat

    return (; exclude_diagonal, destination_sample, D, Ddest, bra_idx, kor_idx, kor_dest_col,
              row_idx, n_eligible, n_excluded, thetaHat, gravity_coefficient)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== Independent Brazil->Korea-excluded gravity regression check ===")
    for exclude_diagonal in (true, false), destination_sample in (:exclude_row, :all_legacy)
        r = run_check(; exclude_diagonal, destination_sample)
        rounds_ok = round(r.gravity_coefficient; digits=2) == -7.43
        println("exclude_diagonal=$(r.exclude_diagonal) destination_sample=$(r.destination_sample): " *
                "n_eligible=$(r.n_eligible) n_excluded=$(r.n_excluded) " *
                "thetaHat=$(r.thetaHat) gravity_coefficient=$(r.gravity_coefficient) " *
                "rounds_to_-7.43=$(rounds_ok)")
    end
end
