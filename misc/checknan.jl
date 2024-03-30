function checknan(x)
    for i in firstindex(x):64:(lastindex(x)-63)
        s = zero(eltype(x))
        for j in 0:63
            s += x[i+j] * 0
        end
        !isfinite(s) && return false
    end
    return all(isfinite, @view x[max(end - 64, begin):end])
end