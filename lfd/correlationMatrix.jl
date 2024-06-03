function correlationMatrix(U, SamplingWeights, LFD_upper, LFD_lower, δ_grid_size, D)

	U_Means = zeros(D^2, 3, δ_grid_size)
	U_Vars = zeros(D^2, 3, δ_grid_size)
	corrMatrix = zeros(D^2, D^2, 3, δ_grid_size)  # D^2, D^2, lower/Initial/upper, δ
	W = length(SamplingWeights)
	RN = zeros(W, 3, δ_grid_size)

	for δ ∈ 1:δ_grid_size
		@. RN[:, 1, δ] = LFD_lower[:, δ] .* SamplingWeights[:]
		@. RN[:, 2, δ] = SamplingWeights[:] 
		@. RN[:, 3, δ] = LFD_upper[:, δ] .* SamplingWeights[:]
	end

	for δ ∈ 1:δ_grid_size
		for i ∈ 1:3
			for o1 ∈ 1:D^2
				U_Means[o1, i, δ] = mean(U[:, o1] .* RN[:, i, δ])
				U_Vars[o1, i, δ] = mean(U[:, o1] .* U[:, o1] .* RN[:, i, δ])
			end
		end
	end


	@. U_Vars[:, :, :] = U_Vars[:, :, :] .- U_Means[:, :, :] .* U_Means[:, :, :]


	for o1 ∈ 1:D^2
		for c1 ∈ 1:D^2
			if o1 > c1
				for δ ∈ 1:δ_grid_size
					for i ∈ 1:3
						corrMatrix[o1, c1, i, δ] = (mean(U[:, o1] .* U[:, c1] .* RN[:, i, δ]) - U_Means[o1, i, δ]*U_Means[c1, i, δ]) / sqrt(U_Vars[o1, i, δ] * U_Vars[c1, i, δ])
					end
				end
			end
		end
	end
	return corrMatrix
end
