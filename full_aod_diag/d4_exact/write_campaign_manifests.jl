# write_campaign_manifests.jl -- writes one campaign manifest per family/direction (task §5, "Write
# one campaign manifest for each family specification and direction. All ten processes must print
# and save the exact manifest hash."), for all 10 chains of the post-verifier-fix W=100k rerun +
# fresh K=3 campaign.
using Dates

const D4E = @__DIR__
isdefined(Main, :MANIFEST_K3_HASH) || include(joinpath(D4E, "w100k_manifest.jl"))

const OUT_DIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/manifests"
isdir(OUT_DIR) || mkpath(OUT_DIR)

const NONZC_DELTAS = [0.01, 0.1, 0.5, 1.0, 2.0]
const NONZC_BUDGET = (explore_s = 2100.0, polish_s = 1500.0)   # 35min + 25min
const K3_WAVE1_DELTAS = [0.01, 0.1, 0.5]
const K3_WAVE2_DELTAS = [1.0, 2.0]
const K3_BUDGET = (explore_s = 4500.0, polish_s = 2700.0)   # 75min + 45min

algorithm_for(family) = family == "unrestricted" ? "Direct+BFGS" :
                        family in ("flexible_cm", "common_frechet") ? "SQP" :
                        family in ("origin_zc", "cm_meanzc") ? "SQP" :
                        error("unknown family $family")

function chain_manifest(family::String, direction::String)
    is_k3 = family in ("origin_zc", "cm_meanzc")
    manifest = is_k3 ? MANIFEST_K3 : MANIFEST_K1
    mhash = is_k3 ? MANIFEST_K3_HASH : MANIFEST_K1_HASH
    (
        family = family, direction = direction,
        is_k3_family = is_k3,
        manifest_hash = mhash,
        scientific_manifest = (
            formulation = manifest.formulation, D = manifest.D, D_dest = manifest.D_dest,
            focal = manifest.focal, sigma = manifest.sigma, gravity_mask = manifest.gravity_mask,
            exclude_diagonal_gravity = manifest.exclude_diagonal_gravity,
            gravity_exclude_brazil_korea = manifest.gravity_exclude_brazil_korea,
            destination_sample = manifest.destination_sample, draw_design = manifest.draw_design,
            draw_seed = manifest.draw_seed, W = manifest.W, A_coordinate_mode = manifest.A_coordinate_mode,
            CM_L = manifest.CM_L, cm_contrasts = manifest.cm_contrasts,
            K_mean = is_k3 ? manifest.ORIGINZC_K_mean : nothing,
            K_pair = is_k3 ? manifest.ORIGINZC_K_pair : nothing,
        ),
        delta_grid = is_k3 ? vcat(K3_WAVE1_DELTAS, K3_WAVE2_DELTAS) : NONZC_DELTAS,
        wave1_deltas = is_k3 ? K3_WAVE1_DELTAS : nothing,
        wave2_deltas = is_k3 ? K3_WAVE2_DELTAS : nothing,
        budget_per_delta_s = is_k3 ? K3_BUDGET : NONZC_BUDGET,
        polish_algorithm = algorithm_for(family),
        exploration_algorithm = "Direct+SR1",
        generated_at = string(Dates.now()),
    )
end

"Minimal hand-rolled JSON value formatter (this repo's own convention -- see
production_backend_manifest.jl -- no JSON.jl dependency in this project)."
jval(x::AbstractString) = "\"" * x * "\""
jval(x::Symbol) = "\"" * string(x) * "\""
jval(x::Bool) = x ? "true" : "false"
jval(x::Nothing) = "null"
jval(x::Real) = string(x)
jval(x::AbstractVector) = "[" * join(jval.(x), ", ") * "]"
function jval(x::NamedTuple)
    "{\n" * join(["    \"$(k)\": " * jval(getfield(x, k)) for k in keys(x)], ",\n") * "\n  }"
end

function write_manifest_json(io, m::NamedTuple)
    println(io, "{")
    ks = keys(m)
    for (i, k) in enumerate(ks)
        v = getfield(m, k)
        sep = i == length(ks) ? "" : ","
        println(io, "  \"$(k)\": ", jval(v), sep)
    end
    println(io, "}")
end

function write_all()
    chains = [
        ("unrestricted", "upper"), ("unrestricted", "lower"),
        ("flexible_cm", "upper"), ("flexible_cm", "lower"),
        ("common_frechet", "upper"), ("common_frechet", "lower"),
        ("origin_zc", "upper"), ("origin_zc", "lower"),
        ("cm_meanzc", "upper"), ("cm_meanzc", "lower"),
    ]
    for (family, direction) in chains
        m = chain_manifest(family, direction)
        path = joinpath(OUT_DIR, "$(family)_$(direction)_manifest.json")
        open(path, "w") do io
            write_manifest_json(io, m)
        end
        println("Wrote ", path, " manifest_hash=", m.manifest_hash)
    end
    println("Wrote ", length(chains), " campaign manifests to ", OUT_DIR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_all()
end
