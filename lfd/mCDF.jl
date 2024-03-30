    function mCDF(x, o, d, i, up_down)
        o1 = o + (d - 1) * D
        mcdf = zeros(length(x))
        nornamization_factor = 0
        for ω = 1:W
            nornamization_factor += (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i]))*StrataWeight[ω] / W
            for j = 1:length(x)
                mcdf[j] += Ū[ω, o1, 1] < x[j] ? (up_down == 0 ? 1 : (up_down == 1 ? LFD_upper[ω, i] : LFD_lower[ω, i])) *StrataWeight[ω]/ W : 0
            end
        end
        return mcdf / nornamization_factor
    end