
function drawU(SamplingWeight, params)
	@unpack W, D, importanceSampling, importanceSamplingFactor, UoModel = params

	sizeU = UoModel == 1 ? D : D * D
	U = zeros(W, sizeU)

	if importanceSampling == 1
		genExpRandsImportanceSampling!(U, importanceSamplingFactor, SamplingWeight)
	elseif importanceSampling == 2
		genExpRandsStratified!(U, SamplingWeight)
	else
		genExpRands!(U)
	end

	return U

end
function drawUCopulaForStartingPoint(params, data, upper)

	if params.useFrechetCopulaStartingPoint == 1
		return drawUSimpleCorrelation(params, upper)
	elseif upper == 1#useFrechetCopulaStartingPoint =2
		return drawUCopulaUp(params, data)
	else #useFrechetCopulaStartingPoint =2
		return drawUCopulaLow(params, data)
	end

end
function drawUSimpleCorrelation(params, upper)
	@unpack W, D, UoModel, baseIndex, FrechetCopulaInitParamUp, FrechetCopulaInitParamLow = params

	sizeU = UoModel == 1 ? D : D * D
	U = zeros(W, sizeU)
	correlation = upper == 1 ? FrechetCopulaInitParamUp : FrechetCopulaInitParamLow

	# set the correlation structure. We want only U[baseIndex, baseIndex] to be correlated with the U[o,baseIndex]. The rest is independent.
	target_corr = zeros(sizeU, sizeU)
	for o in 1:sizeU
		target_corr[o, o] = 1
	end


	baseIndexbaseIndex = baseIndex + (UoModel == 0 ? (baseIndex - 1) * D : 0)
	for o in 1:D
		if o != baseIndex
			obaseIndex = o + (UoModel == 0 ? (baseIndex - 1) * D : 0)
			target_corr[obaseIndex, baseIndexbaseIndex] = correlation
			target_corr[baseIndexbaseIndex, obaseIndex] = correlation
		end
	end

	margins = UnivariateDistribution[]
	for o in 1:sizeU
		push!(margins, Exponential(1))
	end

	U[:] = (rvec(W, target_corr, margins))[:]

	return U

end

function drawUCopulaLow(params, data)
	@unpack W, D, UoModel, baseIndex, FrechetCopulaInitParamLow = params
	lambda = data.λData

	sizeU = UoModel == 1 ? D : D * D

	U = zeros(W, sizeU)
	#rand!(U)

	genExpRands!(U)
	V = zeros(W, 1)
	rand!(V)
	obaseIndex_start = 1 + (UoModel == 0 ? (baseIndex - 1) * D : 0)
	obaseIndex_end = D + (UoModel == 0 ? (baseIndex - 1) * D : 0)
	baseIndexbaseIndex = baseIndex + (UoModel == 0 ? (baseIndex - 1) * D : 0)

	u = range(0, 10, length = 1000)
	CDF_u = zeros(length(u))
	PDF_u = zeros(length(u))
	lambda_factor = (1 - lambda[baseIndex, baseIndex]) / lambda[baseIndex, baseIndex]
	z = FrechetCopulaInitParamLow
	z_ω = floor(W * z)

	for i ∈ 1:length(u)
		CDF_u[i] = 1 - (exp(-u[i]) - z * exp(-u[i] * lambda_factor)) / (1 - z)
		PDF_u[i] = (exp(-u[i]) - z * lambda_factor * exp(-u[i] * lambda_factor)) / (1 - z)

		#CDF_u[i] =1- exp(-u[i])
	end
	@show PDF_u
	@. PDF_u[:] = max.(PDF_u[:],0)
	CDF_Size = length(u)
	δ_1 = 0
	δ_2 = 0
	for ω ∈ 1:W
		U_ω = U[ω, obaseIndex_start:obaseIndex_end] .* lambda[baseIndex, baseIndex] ./ lambda[:, baseIndex]
		Min_U_rw = minimum(U_ω[Not(baseIndex)])


		Inverse_CDF_idx = searchsortedfirst(CDF_u[:], V[ω])
		if Inverse_CDF_idx == 1
			Inverse_CDF_exact = V[ω] * u[1] / CDF_u[1]
		elseif Inverse_CDF_idx == CDF_Size + 1
			Inverse_CDF_exact = (V[ω] - CDF_u[end]) * (11 - u[end]) / (1 - CDF_u[end]) + u[end]
			Inverse_CDF_exact = max(u[end], min(Inverse_CDF_exact, 11))
		else
			Slope_exact = (u[Inverse_CDF_idx] - u[Inverse_CDF_idx-1]) / (CDF_u[Inverse_CDF_idx] - CDF_u[Inverse_CDF_idx-1])
			Inverse_CDF_exact = (V[ω] - CDF_u[Inverse_CDF_idx-1]) * Slope_exact + u[Inverse_CDF_idx-1]
			Inverse_CDF_exact = max(u[Inverse_CDF_idx-1], min(Inverse_CDF_exact, u[Inverse_CDF_idx]))
		end


		δ_1 += phi_x(SmoothDirac(0.0001, U[ω, baseIndexbaseIndex] - Min_U_rw) * exp(U[ω, baseIndexbaseIndex])) * z / W
		δ_2 += phi_x(PDF_u[min(CDF_Size, Inverse_CDF_idx)]* V[ω]) * (1 - z) / W

		if ω < z_ω
			U[ω, baseIndexbaseIndex] = Min_U_rw * (1.001)
		else
			U[ω, baseIndexbaseIndex] = Inverse_CDF_exact
		end


	end

	@show δ_1
	@show δ_2
	@show δ_2 + δ_1

	return U

end

function drawUCopulaUp(params, data)
	@unpack W, D, UoModel, baseIndex, FrechetCopulaInitParamUp = params
	lambda = data.λData

	sizeU = UoModel == 1 ? D : D * D

	U = zeros(W, sizeU)
	genExpRands!(U)
	V = zeros(W, 1)
	rand!(V)
	obaseIndex_start = 1 + (UoModel == 0 ? (baseIndex - 1) * D : 0)
	obaseIndex_end = D + (UoModel == 0 ? (baseIndex - 1) * D : 0)
	baseIndexbaseIndex = baseIndex + (UoModel == 0 ? (baseIndex - 1) * D : 0)

	u = range(0, 10, length = 1000)
	CDF_u = zeros(length(u))
	PDF_u = zeros(length(u))
	lambda_factor = (1 - lambda[baseIndex, baseIndex]) / lambda[baseIndex, baseIndex]
	z = FrechetCopulaInitParamUp
	z_ω = floor(W * z)
	K = 10 # gains from trade for product ω in z_ω

	for i ∈ 1:length(u)
		CDF_u[i] = 1 - (exp(-u[i]) - z * exp(-u[i] * lambda_factor / (1 + K))) / (1 - z)
		PDF_u[i] = (exp(-u[i]) - z * (lambda_factor / (1 + K)) * exp(-u[i] * lambda_factor / (1 + K))) / (1 - z)
		#CDF_u[i] =1- exp(-u[i])
	end

	@show PDF_u
	@. PDF_u[:] = max.(PDF_u[:],0)
	CDF_Size = length(u)
	δ_1 = 0
	δ_2 = 0

	for ω ∈ 1:W
		U_ω = U[ω, obaseIndex_start:obaseIndex_end] .* lambda[baseIndex, baseIndex] ./ lambda[:, baseIndex]
		Min_U_rw = minimum(U_ω[Not(baseIndex)])

		Inverse_CDF_idx = searchsortedfirst(CDF_u[:], V[ω])
		if Inverse_CDF_idx == 1
			Inverse_CDF_exact = V[ω] * u[1] / CDF_u[1]
		elseif Inverse_CDF_idx == CDF_Size + 1
			Inverse_CDF_exact = (V[ω] - CDF_u[end]) * (11 - u[end]) / (1 - CDF_u[end]) + u[end]
			Inverse_CDF_exact = max(u[end], min(Inverse_CDF_exact, 11))
		else
			Slope_exact = (u[Inverse_CDF_idx] - u[Inverse_CDF_idx-1]) / (CDF_u[Inverse_CDF_idx] - CDF_u[Inverse_CDF_idx-1])
			Inverse_CDF_exact = (V[ω] - CDF_u[Inverse_CDF_idx-1]) * Slope_exact + u[Inverse_CDF_idx-1]
			Inverse_CDF_exact = max(u[Inverse_CDF_idx-1], min(Inverse_CDF_exact, u[Inverse_CDF_idx]))
		end


		δ_1 += phi_x(SmoothDirac(0.0001, U[ω, baseIndexbaseIndex] - Min_U_rw * (1 + K)) * exp(U[ω, baseIndexbaseIndex])) * z / W
		δ_2 += phi_x(PDF_u[min(CDF_Size, Inverse_CDF_idx)]* V[ω]) * (1 - z) / W

		if ω < z_ω
			U[ω, baseIndexbaseIndex] = Min_U_rw * (1 + K)
		else
			U[ω, baseIndexbaseIndex] = Inverse_CDF_exact
		end

	end
	@show δ_1
	@show δ_2
	@show δ_2 + δ_1

	return U

end

function phi_x(x)
	return x * log(x) - x + 1
end
