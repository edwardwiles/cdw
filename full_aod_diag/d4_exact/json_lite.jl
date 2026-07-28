# json_lite.jl -- minimal dependency-free JSON reader (2026-07-28 shakedown campaign).
# No JSON package is a Project.toml dependency in this repo. This parser is scoped to read back
# exactly the manifest shape common_five_starts_search.jl's own jesc/jnum/jvec/jmat writers produce
# (nested objects/arrays/numbers/strings/true/false/null, no unicode escapes beyond \" and \\) --
# not a general-purpose JSON library.

function json_parse(s::AbstractString)
    i = firstindex(s)
    n = lastindex(s)
    skipws() = (while i <= n && isspace(s[i]); i = nextind(s, i); end)
    local parse_value
    function parse_string()
        @assert s[i] == '"'
        i = nextind(s, i)
        buf = IOBuffer()
        while s[i] != '"'
            c = s[i]
            if c == '\\'
                i = nextind(s, i)
                c2 = s[i]
                write(buf, c2 == 'n' ? '\n' : c2 == 't' ? '\t' : c2)
            else
                write(buf, c)
            end
            i = nextind(s, i)
        end
        i = nextind(s, i)
        return String(take!(buf))
    end
    function parse_number()
        j = i
        while i <= n && (isdigit(s[i]) || s[i] in ('-', '+', '.', 'e', 'E'))
            i = nextind(s, i)
        end
        tok = s[j:prevind(s, i)]
        return occursin('.', tok) || occursin('e', tok) || occursin('E', tok) ? parse(Float64, tok) : parse(Int, tok)
    end
    function parse_array()
        @assert s[i] == '['
        i = nextind(s, i); skipws()
        out = Any[]
        if s[i] == ']'
            i = nextind(s, i); return out
        end
        while true
            skipws(); push!(out, parse_value()); skipws()
            if s[i] == ','
                i = nextind(s, i); skipws()
            elseif s[i] == ']'
                i = nextind(s, i); break
            else
                error("json_lite: expected , or ] at char $i")
            end
        end
        return out
    end
    function parse_object()
        @assert s[i] == '{'
        i = nextind(s, i); skipws()
        out = Dict{String,Any}()
        if s[i] == '}'
            i = nextind(s, i); return out
        end
        while true
            skipws(); k = parse_string(); skipws()
            @assert s[i] == ':'
            i = nextind(s, i); skipws()
            v = parse_value()
            out[k] = v
            skipws()
            if s[i] == ','
                i = nextind(s, i); skipws()
            elseif s[i] == '}'
                i = nextind(s, i); break
            else
                error("json_lite: expected , or } at char $i")
            end
        end
        return out
    end
    parse_value = function ()
        skipws()
        c = s[i]
        if c == '{'
            return parse_object()
        elseif c == '['
            return parse_array()
        elseif c == '"'
            return parse_string()
        elseif startswith(SubString(s, i), "true")
            i += 4; return true
        elseif startswith(SubString(s, i), "false")
            i += 5; return false
        elseif startswith(SubString(s, i), "null")
            i += 4; return nothing
        else
            return parse_number()
        end
    end
    skipws()
    v = parse_value()
    return v
end

json_load(path::AbstractString) = json_parse(read(path, String))

"Convert a JSON-parsed Vector{Any} of numbers to Vector{Float64}."
jf64(v) = Float64.(v)
