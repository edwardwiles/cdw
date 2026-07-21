# Fail-fast KNITRO native-library version check.
#
# KNITRO.jl's deps/deps.jl (in the shared Julia depot, `~/.julia/packages/KNITRO/<hash>/deps/deps.jl`)
# hardcodes an absolute path to the native `libknitro.so` at `Pkg.build` time; it does NOT
# re-resolve `KNITRODIR`/`LD_LIBRARY_PATH` at runtime, so sourcing `.knitro_env.sh` alone is not
# sufficient to change which version actually loads. This check exists because that mismatch
# previously went undetected for an entire benchmarking session (everything ran on 13.0.1 while
# docs/env scripts claimed 14.2.0).
#
# 14.2.0 (and 14.0.0) are installed on this host but are NOT covered by the site's current Ziena
# license (/etc/sharedsw_licenses/ziena.txt, dated 2022-05-02): `KN_new()` fails with return code
# -520 ("Could not find a valid license") on both, confirmed live on 2026-07-21, while 13.0.1
# succeeds under the identical license file. 13.0.1 is therefore the only currently-usable
# version and is declared production here; migrating to 14.2.0 requires a license renewal from
# Artelys (licensing@artelys.com) first, independent of any code change.
const KNITRO_PRODUCTION_VERSION = "13.0.1"

function verify_knitro_version(expected::AbstractString = KNITRO_PRODUCTION_VERSION)
    rel = zeros(UInt8, 128)
    KNITRO.@ccall KNITRO.libknitro.KN_get_release(length(rel)::Cint, rel::Ptr{Cchar})::Cint
    release_string = unsafe_string(pointer(rel))
    loaded_path = KNITRO.libknitro

    println("KNITRO.jl loaded library path: ", loaded_path)
    println("KNITRO.jl package version: ", pkgversion(KNITRO))
    println("KN_get_release: ", release_string)

    if !occursin(expected, release_string)
        error(
            "KNITRO version mismatch: expected native release containing \"$expected\" " *
            "(declared production version), but KN_get_release() returned \"$release_string\" " *
            "(loaded from $loaded_path). Rebuild KNITRO.jl against the intended KNITRODIR " *
            "(source .knitro_env.sh, set KNITRO_JL_USE_KNITRO_JLL=false, then " *
            "`julia --project=. -e 'import Pkg; Pkg.build(\"KNITRO\")'`) before proceeding.",
        )
    end
    return release_string
end
