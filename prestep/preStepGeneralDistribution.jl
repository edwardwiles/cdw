#=
function preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, distributionType, corr, vol, autocorr=0)

	D = size(lambda, 1)
	W = 1000000 # number of products for calculating the expectations, we use 1 million products so to emulate the "theoretical" quantities.
	# construct matrix of draws from distributionType
	U_init = zeros(W, D * D)
	Random.seed!(2^12)
	genRands!(U_init, distributionType, corr, vol, autocorr, tau)

	return preStepGeneralDistribution(L, LPrime, tau, tauPrime, lambda, thetaIn, sigma, baseIndex, counterType, U_init)

end
=#

function preStepGeneralDistribution(data, counters, globalParams)

	#A function that  does the full pre-step to get the initial guess of parameters (theta_upper and theta_lower ) that corresponds with the useFrechetCopulaStartingPoint parameters 
	#it provides also the Frechet Theta_init 

	Random.seed!(1234567890123)
	upper = 1
	U_init = drawUCopulaForStartingPoint(globalParams, data, upper)
	output_upper = preStepGeneralDistribution(data, counters, globalParams, U_init)

	upper = 0
	U_init = drawUCopulaForStartingPoint(globalParams, data, upper)
	output_lower = preStepGeneralDistribution(data, counters, globalParams, U_init)

	output = (μHat = output_upper.μHat,
		wHat = output_upper.wHat[:],
		cHat = output_upper.cHat,
		λPrime = output_upper.λPrime, #this is not used anywhere downstream, we calculate it just to get wagePrime
		wPrimeHat = output_upper.wPrimeHat[:],
		γHat = output_upper.γHat[:],
		γPrimeHat = output_upper.γPrimeHat[:],
		Aod_initial = output_upper.Aod_initial[:],
		wPrimeHat_upper = output_upper.wPrimeHat_upper[:],
		γHat_upper = output_upper.γHat_upper[:],
		γPrimeHat_upper = output_upper.γPrimeHat_upper[:],
		Aod_initial_upper = output_upper.Aod_initial_upper[:],
		wPrimeHat_lower = output_lower.wPrimeHat_lower[:],
		γHat_lower = output_lower.γHat_lower[:],
		γPrimeHat_lower = output_lower.γPrimeHat_lower[:],
		Aod_initial_lower = output_lower.Aod_initial_lower[:])

	# Gain from trade for each distribution
	κ = (output.γHat[globalParams.baseIndex] / output.γPrimeHat[globalParams.baseIndex])^(globalParams.σHat / (globalParams.σHat - 1)) - 1
	κ_upper = (output.γHat_upper[globalParams.baseIndex] / output.γPrimeHat_upper[globalParams.baseIndex])^(globalParams.σHat / (globalParams.σHat - 1)) - 1
	κ_lower = (output.γHat_lower[globalParams.baseIndex] / output.γPrimeHat_lower[globalParams.baseIndex])^(globalParams.σHat / (globalParams.σHat - 1)) - 1

	println("Gains from Trade":println(string("Frechet = ", κ)))
	println("Gains from Trade":println(string("Lower Copula = ", κ_lower)))
	println("Gains from Trade":println(string("Upper Copula = ", κ_upper)))
	return output
end

function preStepGeneralDistribution(data, counters, globalParams, U_init)
	#A function that   does the full pre-step to get the initial guess of parameters (theta_upper and theta_lower are equal in this case) that corresponds with the supplied U_init
	#it provides also the Frechet Theta_init 
	lambda = data.λData
	L = data.LData
	tau = data.τData

	# unpack counterfactual info 
	tauPrime = counters.τPrime
	LPrime = counters.LPrime

	# unpack parameter info
	thetaIn = globalParams.θHat
	sigma = globalParams.σHat
	baseIndex = globalParams.baseIndex
	counterType = globalParams.counterType
	D = globalParams.D
	UoModel = globalParams.UoModel

	UoModel_factor = UoModel == 1 ? 0 : 1

	W = size(U_init, 1)

	# Step 1: Estimate thetaHat via gravity or prespecified
	if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
		deltaLambda = doubleDiff(lambda)
		deltaTau = doubleDiff(tau)
		meanTau = mean(deltaTau)
		thetaHat = -sum(deltaLambda .* (deltaTau .- meanTau)) / sum(deltaTau .* deltaTau .- meanTau^2)
	elseif thetaIn > 0
		thetaHat = thetaIn
	end

	# Step 2: Estimate baseline wages
	wHat = ones(D)
	iterWagesPreStep!(wHat, L, lambda)
	wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

	# Step 3: Estimate cHat

	UPow_init = zeros(W, D * (UoModel == 1 ? 1 : D))
	UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)

	# use Frechet solution as the initial guess
	AHat_Frechet = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
	cHat_Frechet = AHat_Frechet .^ (-1) # we define c as 1/A 


	cHat = ones(D, D)
	# force cHat[1,:] =1
	normalizeAs = true
	if normalizeAs
		for d ∈ 1:D
			function cHat_solver_func!(cHat_x)
				return gFunction_d!(sigma, tau, UPow_init, wHat, vcat(ones(1), cHat_x), lambda, d, UoModel_factor) # we force the cHat[1,:] =1
			end
			cHat_solver_results = nlsolve(cHat_solver_func!, (cHat_Frechet[2:D, d]) .^ (1.0 / thetaHat)) #, autodiff = :forward
			@show cHat_solver_results.residual_norm ## check it converged
			cHat[2:D, d] .= abs.(cHat_solver_results.zero)[:]
			@show cHat[:, d]
		end
	else
		for d ∈ 1:D
			function cHat_solver_func2!(cHat_x)
				return gFunction_d2!(sigma, tau, UPow_init, wHat, cHat_x, lambda, d, UoModel_factor)
			end
			cHat_solver_results = nlsolve(cHat_solver_func2!, (cHat_Frechet[:, d]) .^ (1.0 / thetaHat)) #, autodiff = :forward
			#@show cHat_solver_results.residual_norm ## check it converged
			cHat[:, d] .= abs.(cHat_solver_results.zero)[:]
			#@show cHat[:, d]
		end
	end
	GC.gc()


	# Step 4: Estimate wPrimeHat
	#lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise
	lambdaPrime = zeros(D, D)
	wPrimeHat = ones(D)


	if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
		function wPrimeHat_solver_func!(wHatPrime_x)
			func_output = gFunction!(sigma, tauPrime, UPow_init, wHatPrime_x, cHat, lambda, UoModel_factor)

			wage_moment = zeros(Float64, D)
			for o ∈ 1:D
				wage_moment[o] = sum(func_output.expenditure_ratio[o, :] .* wHatPrime_x .* LPrime) - wHatPrime_x[o] * LPrime[o]
			end

			return wage_moment
		end

		wPrimeHat_solver_results = nlsolve(wPrimeHat_solver_func!, wHat)

		@show wPrimeHat_solver_results.residual_norm ## check it converged

		wPrimeHat = abs.(wPrimeHat_solver_results.zero)
		wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 

		lambdaPrime = (gFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambda, UoModel_factor)).expenditure_ratio
	else
		for o ∈ 1:D
			lambdaPrime[o, o] = 1
		end
	end

	price_index = ones(D)
	price_index_Prime = ones(D)

	price_index[:] .= ((gFunction!(sigma, tau, UPow_init, wHat, cHat, lambda, UoModel_factor)).price_index)[:]
	price_index_Prime[:] .= ((gFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambdaPrime, UoModel_factor)).price_index)[:]

	# Step 5: Compute gammaHat and gammaPrimeHat
	gammaHat = ones(D)
	gammaPrimeHat = ones(D)

	gammaHat = (price_index ./ (wHat .* L)) .^ (1 / sigma)
	gammaPrimeHat = (price_index_Prime ./ (wPrimeHat .* LPrime)) .^ (1 / sigma)

	#	gammaHat[:] = (price_index[:] ./ (wHat[:] .* L[:])) .^ (1.0 / sigma)
	#	gammaPrimeHat[:] = (price_index_Prime[:] ./ (wPrimeHat[:] .* LPrime[:])) .^ (1.0 / sigma)


	# we were using above cHat^(1/theta), for numerical efficiency. We adjust the power before returning the values.
	@. cHat[:] = cHat[:] .^ (thetaHat)

	lambdaPrimeFrechet = Array{Float64, 2}(undef, D, D)  # initialise
	wPrimeHatFrechet = ones(D)

	if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
		iterWagesTheory!(wPrimeHatFrechet, LPrime, AHat_Frechet, tauPrime, thetaHat, lambdaPrimeFrechet)
		wPrimeHatFrechet[:] = wPrimeHatFrechet ./ wPrimeHatFrechet[baseIndex] # normalise 
	end

	gammaHatFrechet = computeGamma(AHat_Frechet, tau, wHat, thetaHat, sigma, L)
	gammaPrimeHatFrechet = computeGamma(AHat_Frechet, tauPrime, wPrimeHatFrechet, thetaHat, sigma, LPrime)

	additional_theta = globalParams.theta_init == 0 ? thetaHat : globalParams.theta_init
	AHat_additional = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (additional_theta)) .* (lambda ./ lambda[1, :]')
	cHat_additional = AHat_additional .^ (-1) # we define c as 1/A 

	Aod_initial = cHat_Frechet ./ cHat


	output = (μHat = 1 ./ thetaHat,
		wHat = wHat[:],
		cHat = cHat_Frechet,
		λPrime = lambdaPrime,
		wPrimeHat = wPrimeHat[:],
		γHat = gammaHatFrechet[:],
		γPrimeHat = gammaPrimeHatFrechet[:],
		Aod_initial = ones(D^2),
		wPrimeHat_upper = copy(wPrimeHat[:]),
		γHat_upper = copy(gammaHat[:]),
		γPrimeHat_upper = copy(gammaPrimeHat[:]),
		Aod_initial_upper = reshape(Aod_initial, D^2)[:],
		wPrimeHat_lower = copy(wPrimeHat[:]),
		γHat_lower = copy(gammaHat[:]),
		γPrimeHat_lower = copy(gammaPrimeHat[:]),
		Aod_initial_lower = reshape(Aod_initial, D^2)[:],
	)


	return output
end

function gFunction!(σ, τ, UPow_init, w_, cPow_, lambda, UoModel_factor)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	cPow = abs.(cPow_)

	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)
	bilateral_prices_pow = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D, D)
	expenditure_ratio_residual = zeros(eltype(cPow), D, D)
	expenditure_ratio = zeros(eltype(cPow), D, D)
	trade_share_residual = zeros(eltype(cPow), D, D)
	price_index = zeros(eltype(cPow), D)
	exporter_idx = 0.0

	for d ∈ 1:D
		for ω ∈ 1:W
			for o ∈ 1:D
				o1 = o + UoModel_factor * (d - 1) * D
				bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o, d]) / (UPow_init[ω, o1]))^(1 - σ)
			end

			prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
			expenditure[exporter_idx, d] += prices_pow
			price_index[d] += prices_pow

		end

		for o ∈ 1:D
			expenditure_ratio_residual[o, d] = lambda[o, d] / lambda[1, d] - expenditure[o, d] / expenditure[1, d] # we could have some cases where expenditure on country 1 goods is zero...
			expenditure_ratio[o, d] = expenditure[o, d] / price_index[d]
			trade_share_residual[o, d] = expenditure_ratio[o, d] - lambda[o, d]
		end
	end

	#@show expenditure

	return (expenditure_ratio_residual = expenditure_ratio_residual, price_index = price_index ./ W, expenditure_ratio = expenditure_ratio, trade_share_residual = trade_share_residual, expenditure = expenditure / W)

end
function gFunction_d!(σ, τ, UPow_init, w_, cPow_, lambda, d, UoModel_factor)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	cPow = abs.(cPow_)

	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)
	bilateral_prices_pow = zeros(eltype(cPow), D)
	bilateral_prices = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D)
	expenditure_ratio_residual = zeros(eltype(cPow), D - 1)

	price_index = 0.0
	exporter_idx = 0.0
	DiscreteImplementation = true
	pricesIdx = zeros(eltype(cPow), D)

	for ω ∈ 1:W

		for o ∈ 1:D
			o1 = o + UoModel_factor * (d - 1) * D
			bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
			bilateral_prices[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))
		end

		if DiscreteImplementation
			prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
			expenditure[exporter_idx] += prices_pow
			price_index += prices_pow
		else
			smoothMinIndNew!(pricesIdx, bilateral_prices, D, -100.0)

			for o ∈ 1:D
				prices_pow = pricesIdx[o] * bilateral_prices_pow[o]
				expenditure[o] += prices_pow
				price_index += prices_pow
			end
		end

	end

	for o ∈ 2:D
		relative_error = lambda[o, d] / lambda[1, d] - expenditure[o] / expenditure[1]
		absolute_error = lambda[o, d] - expenditure[o] / price_index
		#expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
		expenditure_ratio_residual[o-1] = absolute_error # this version works as well, sum of lambdas = 1, probably better as this is the moment condition in the algo
	end
	return expenditure_ratio_residual

end
function gFunction_d2!(σ, τ, UPow_init, w_, cPow_, lambda, d, UoModel_factor)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	cPow = abs.(cPow_)

	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)
	bilateral_prices_pow = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D)
	expenditure_ratio_residual = zeros(eltype(cPow), D)
	pricesIdx = zeros(eltype(cPow), D)
	price_index = 0.0
	exporter_idx = 0.0
	DiscreteImplementation = true

	for ω ∈ 1:W

		for o ∈ 1:D
			o1 = o + UoModel_factor * (d - 1) * D
			bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
		end


		if DiscreteImplementation
			prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
			expenditure[exporter_idx] += prices_pow
			price_index += prices_pow
		else
			smoothMinIndNew!(pricesIdx, bilateral_prices_pow, D, 100.0) # we are looking for the max, so the positive tuner sign is correct.

			for o ∈ 1:D
				prices_pow = pricesIdx[o] * bilateral_prices_pow[o]
				expenditure[o] += prices_pow
				price_index += prices_pow
			end
		end
	end

	for o ∈ 1:D
		#expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
		expenditure_ratio_residual[o] = lambda[o, d] - expenditure[o] / price_index # this version works as well
	end
	return expenditure_ratio_residual

end


function preStepGeneralDistributionRN(data, counters, globalParams, U_init)
	#A function that   does the full pre-step to get the initial guess of parameters (theta_upper and theta_lower are equal in this case) that corresponds with the supplied U_init
	#it provides also the Frechet Theta_init 
	lambda = data.λData
	L = data.LData
	tau = data.τData

	# unpack counterfactual info 
	tauPrime = counters.τPrime
	LPrime = counters.LPrime

	# unpack parameter info
	thetaIn = globalParams.θHat
	sigma = globalParams.σHat
	baseIndex = globalParams.baseIndex
	counterType = globalParams.counterType
	D = globalParams.D
	UoModel = globalParams.UoModel

	UoModel_factor = UoModel == 1 ? 0 : 1

	W = size(U_init, 1)

	# Step 1: Estimate thetaHat via gravity or prespecified
	if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
		deltaLambda = doubleDiff(lambda)
		deltaTau = doubleDiff(tau)
		meanTau = mean(deltaTau)
		thetaHat = -sum(deltaLambda .* (deltaTau .- meanTau)) / sum(deltaTau .* deltaTau .- meanTau^2)
	elseif thetaIn > 0
		thetaHat = thetaIn
	end

	# Step 2: Estimate baseline wages
	wHat = ones(D)
	iterWagesPreStep!(wHat, L, lambda)
	wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

	# Step 3: Estimate cHat

	UPow_init = zeros(W, D * (UoModel == 1 ? 1 : D))
	UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)

	# use Frechet solution as the initial guess
	AHat_Frechet = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
	cHat_Frechet = AHat_Frechet .^ (-1) # we define c as 1/A 

	lambdaPrimeFrechet = Array{Float64, 2}(undef, D, D)  # initialise
	wPrimeHatFrechet = ones(D)

	if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
		iterWagesTheory!(wPrimeHatFrechet, LPrime, AHat_Frechet, tauPrime, thetaHat, lambdaPrimeFrechet)
		wPrimeHatFrechet[:] = wPrimeHatFrechet ./ wPrimeHatFrechet[baseIndex] # normalise 
	end

	gammaHatFrechet = computeGamma(AHat_Frechet, tau, wHat, thetaHat, sigma, L)
	gammaPrimeHatFrechet = computeGamma(AHat_Frechet, tauPrime, wPrimeHatFrechet, thetaHat, sigma, LPrime)

	κ_Frechet = (gammaHatFrechet[baseIndex] / gammaPrimeHatFrechet[baseIndex])^(sigma / (sigma - 1)) - 1


	cHat = ones(D, D)
	RN = ones(W)
	# force cHat[1,:] =1
	normalizeAs = true
	upper_lower = [-1, 1]

	gammaHat = ones(D, 2)
	gammaPrimeHat = ones(D, 2)
	Aod_initial = ones(D, D, 2)

	for i_upper_lower in 1:2
		# We restrict to Gains from Trade Counterfactual. For generalization, we need to solve for all ds at the same time.

		# First find the RN derivative and the A_ods for d = baseIndex

		function cHat_baseIndex_solver_func!(cHat_x)
			return gmFunction_baseIndex!(sigma, tau, UPow_init, wHat, cHat_Frechet[:, baseIndex] .^ (1.0 / thetaHat), cHat_x[1], cHat_x[2], lambda, baseIndex, UoModel_factor, upper_lower[i_upper_lower], globalParams.δ_ref, RN, κ_Frechet, thetaHat) # we force the cHat[1,:] =1
		end
		cHat_solver_results = nlsolve(cHat_baseIndex_solver_func!, vcat((cHat_Frechet[baseIndex, baseIndex]) .^ (1.0 / thetaHat), 0.5), ftol = 1e-3) #, autodiff = :forward
		@show cHat_solver_results
		@show cHat_solver_results.residual_norm ## check it convergedqw2wa
		@show cHat_baseIndex_solver_func!(cHat_solver_results.zero)
		cHat[:, baseIndex] .= cHat_Frechet[:, baseIndex] .^ (1.0 / thetaHat)
		cHat[baseIndex, baseIndex] = abs.(cHat_solver_results.zero)[1]
		#α_RN = abs.(cHat_solver_results.zero)[D:D+2]
		#K_RN = abs.(cHat_solver_results.zero)[D+3]
		@show cHat[:, baseIndex]
		#@show α_RN
		#@show K_RN

		# Actually this is needed only for Uo model
		for d ∈ 1:D
			if d != baseIndex
				function cHat_solver_func!(cHat_x)
					return gmFunction_d!(sigma, tau, UPow_init, wHat, cHat_Frechet[:, d] .^ (1.0 / thetaHat), cHat_x[1], lambda, d, UoModel_factor, RN, baseIndex) # we force the cHat[1,:] =1
				end
				cHat_solver_results = nlsolve(cHat_solver_func!, vcat((cHat_Frechet[baseIndex, d])^(1.0 / thetaHat))) #, autodiff = :forward
				@show cHat_solver_results.residual_norm ## check it converged
				cHat[:, d] .= cHat_Frechet[:, d] .^ (1.0 / thetaHat)
				cHat[baseIndex, d] = abs.(cHat_solver_results.zero)[1]
				@show cHat[:, d]
			end
		end

		GC.gc()


		# Step 4: Estimate wPrimeHat
		#lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise
		lambdaPrime = zeros(D, D)
		wPrimeHat = ones(D)


		if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
			function wPrimeHat_solver_func!(wHatPrime_x)
				func_output = gmFunction!(sigma, tauPrime, UPow_init, wHatPrime_x, cHat, lambda, UoModel_factor, RN)

				wage_moment = zeros(Float64, D)
				for o ∈ 1:D
					wage_moment[o] = sum(func_output.expenditure_ratio[o, :] .* wHatPrime_x .* LPrime) - wHatPrime_x[o] * LPrime[o]
				end

				return wage_moment
			end

			wPrimeHat_solver_results = nlsolve(wPrimeHat_solver_func!, wHat)

			@show wPrimeHat_solver_results.residual_norm ## check it converged

			wPrimeHat = abs.(wPrimeHat_solver_results.zero)
			wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 

			lambdaPrime = (gmFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambda, UoModel_factor, RN)).expenditure_ratio
		else
			for o ∈ 1:D
				lambdaPrime[o, o] = 1
			end
		end

		price_index = ones(D)
		price_index_Prime = ones(D)

		price_index[:] .= ((gmFunction!(sigma, tau, UPow_init, wHat, cHat, lambda, UoModel_factor, RN)).price_index)[:]
		price_index_Prime[:] .= ((gmFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambdaPrime, UoModel_factor, RN)).price_index)[:]

		# Step 5: Compute gammaHat and gammaPrimeHat


		gammaHat[:, i_upper_lower] = deepcopy((price_index ./ (wHat .* L)) .^ (1 / sigma))
		gammaPrimeHat[:, i_upper_lower] = deepcopy((price_index_Prime ./ (wPrimeHat .* LPrime)) .^ (1 / sigma))

		# we were using above cHat^(1/theta), for numerical efficiency. We adjust the power before returning the values.
		@. cHat[:] = cHat[:] .^ (thetaHat)
		Aod_initial[:, :, i_upper_lower] = deepcopy(cHat_Frechet[:, :] ./ cHat[:, :])
	end
	@show gammaHat[:, 1]

	@show gammaHat[:, 2]
	@show Aod_initial[:, :, 1]
	@show Aod_initial[:, :, 2]

	output = (μHat = 1 ./ thetaHat,
		wHat = wHat[:],
		cHat = cHat_Frechet,
		λPrime = lambdaPrimeFrechet,
		wPrimeHat = wPrimeHatFrechet[:],
		γHat = gammaHatFrechet[:],
		γPrimeHat = gammaPrimeHatFrechet[:],
		Aod_initial = ones(D^2),
		wPrimeHat_upper = copy(wPrimeHatFrechet[:]),
		γHat_upper = copy(gammaHat[:, 2]),
		γPrimeHat_upper = copy(gammaPrimeHat[:, 2]),
		Aod_initial_upper = reshape(Aod_initial[:, :, 2], D^2)[:],
		wPrimeHat_lower = copy(wPrimeHatFrechet[:]),
		γHat_lower = copy(gammaHat[:, 1]),
		γPrimeHat_lower = copy(gammaPrimeHat[:, 1]),
		Aod_initial_lower = reshape(Aod_initial[:, :, 1], D^2)[:],
	)


	return output
end

function gmFunction_baseIndex!(σ, τ, UPow_init, w_, cPowFrechet, cPow_, α_m_, lambda, d, UoModel_factor, lower_bound, δ_max, m, κ_Frechet, thetaHat)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)

	cPow = zeros(D)
	cPow[:] = cPowFrechet[:]
	cPow[d] = abs.(cPow_)

	α_m = abs.(α_m_)

	bilateral_prices_pow = zeros(eltype(cPow), D)
	bilateral_prices = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D)
	expenditure_ratio_residual = zeros(eltype(cPow), D)

	price_index = 0.0
	exporter_idx = 0.0
	autarky_price_index = 0
	m_expectation = 0
	DiscreteImplementation = true
	pricesIdx = zeros(eltype(cPow), D)
#=
	phi_total = 0
	for o ∈ 1:D
		phi_total += (w[o] * τ[o, d] * cPow[o])^(-thetaHat)
	end
	phi_d = (w[d] * τ[d, d] * cPow[d])^(-thetaHat)

	phi_rw = phi_total - phi_d

	dirac_expectation = thetaHat * (phi_rw * phi_d) / phi_total

	phi_rw_pow = phi_rw^(-1 / thetaHat)

	shift_factor_expectation = phi_rw / phi_total

	@show lambda[d, d]

=#
	α_m = min(1 - 0.0001, α_m)

	dirac_exp2 = 0
	dirac_exp3 = 0
	shift_factor_expectation2 = 0
	for ω ∈ 1:W

		for o ∈ 1:D
			o1 = o + UoModel_factor * (d - 1) * D
			bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
			bilateral_prices[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))
		end

		#if DiscreteImplementation
		prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
		prices_rw_pow, exporter_rw_idx = findmax(bilateral_prices_pow[Not(d)])
		dom_rw_price_ratio = bilateral_prices_pow[d] / prices_rw_pow
		shift_factor = dom_rw_price_ratio < 1 ? 1 : 0
		#m[ω] = α_m[1] * SmoothDirac(0.001, dom_rw_price_ratio - 1) + (dom_rw_price_ratio > K ? α_m[2] : α_m[3])
		#UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)
		d1 = d + UoModel_factor * (d - 1) * D
		U = UPow_init[ω, d1]^(-thetaHat)
		U_signed = lower_bound == -1 ? U : -log(1 - exp(-U))
		U_pow = U_signed^(1.0 / thetaHat)
		V = exp(-U_signed)
		#U_rw_pow = (bilateral_prices[Not(d)])[exporter_rw_idx]/(w[d] * τ[d, d] * cPow[d])
		U_rw_pow = (bilateral_prices[Not(d)])[exporter_rw_idx] / phi_rw_pow
		V_rw = exp(-U_rw_pow^(thetaHat))
		#m[ω] = α_m[1] * SmoothDirac(0.001, U_pow-U_rw_pow)/dirac_expectation + 1- α_m[1]
		dirac_term = SmoothOneSidedDirac(0.01, -(V - V_rw))
		m[ω] = α_m * dirac_term + (1 - α_m)
		dirac_exp3 += dirac_term / W
		m_expectation += m[ω] / W
		expenditure[exporter_idx] += prices_pow * m[ω] / W
		price_index += prices_pow * m[ω] / W
		autarky_price_index += bilateral_prices_pow[d] * m[ω] / W
		shift_factor_expectation2 += shift_factor / W
	end

	@. expenditure[:] = expenditure[:] / m_expectation
	price_index = price_index / m_expectation
	autarky_price_index = autarky_price_index / m_expectation
	@. m[:] = m[:] ./ m_expectation
	δ = sum(@.phi_x.(m[:])) / W



	gamma_gamma_prime = price_index / autarky_price_index
	kappa = gamma_gamma_prime^(1 / (σ - 1)) - 1

	@show δ
	@show α_m
	@show m_expectation
	@show κ_Frechet
	@show kappa
	@show dirac_exp3

	#delta constraint
	#expenditure_ratio_residual[D] = (δ < δ_max ? 0 : 10000)
	expenditure_ratio_residual[D] = (δ - δ_max)
	#maximize/minimize counterfactual 
	#expenditure_ratio_residual[D+1] = exp(lower_bound*(κ_Frechet - kappa)/κ_Frechet)

	for o ∈ 2:D
		relative_error = lambda[o, d] / lambda[1, d] - expenditure[o] / expenditure[1]
		absolute_error = lambda[o, d] - expenditure[o] / price_index
		#expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
		expenditure_ratio_residual[o-1] = absolute_error # this version works as well, sum of lambdas = 1, probably better as this is the moment condition in the algo
	end
	expenditure_ratio_residual_baseIndex = vcat(expenditure_ratio_residual[d-1], (δ - δ_max))
	return expenditure_ratio_residual_baseIndex

end
function gmFunction!(σ, τ, UPow_init, w_, cPow_, lambda, UoModel_factor, RN)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	cPow = abs.(cPow_)

	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)
	bilateral_prices_pow = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D, D)
	expenditure_ratio_residual = zeros(eltype(cPow), D, D)
	expenditure_ratio = zeros(eltype(cPow), D, D)
	trade_share_residual = zeros(eltype(cPow), D, D)
	price_index = zeros(eltype(cPow), D)
	exporter_idx = 0.0

	for d ∈ 1:D
		for ω ∈ 1:W
			for o ∈ 1:D
				o1 = o + UoModel_factor * (d - 1) * D
				bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o, d]) / (UPow_init[ω, o1]))^(1 - σ)
			end

			prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
			expenditure[exporter_idx, d] += prices_pow * RN[ω]
			price_index[d] += prices_pow * RN[ω]

		end

		for o ∈ 1:D
			expenditure_ratio_residual[o, d] = lambda[o, d] / lambda[1, d] - expenditure[o, d] / expenditure[1, d] # we could have some cases where expenditure on country 1 goods is zero...
			expenditure_ratio[o, d] = expenditure[o, d] / price_index[d]
			trade_share_residual[o, d] = expenditure_ratio[o, d] - lambda[o, d]
		end
	end

	#@show expenditure

	return (expenditure_ratio_residual = expenditure_ratio_residual, price_index = price_index ./ W, expenditure_ratio = expenditure_ratio, trade_share_residual = trade_share_residual, expenditure = expenditure / W)

end
function gmFunction_d!(σ, τ, UPow_init, w_, cHatFrechet, cPow_, lambda, d, UoModel_factor, RN, baseIndex)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	#cPow = abs.(cPow_)

	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)

	cPow = zeros(D)
	cPow[:] = cHatFrechet[:]
	cPow[baseIndex] = abs.(cPow_)


	bilateral_prices_pow = zeros(eltype(cPow), D)
	bilateral_prices = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D)
	expenditure_ratio_residual = zeros(eltype(cPow), D - 1)

	price_index = 0.0
	exporter_idx = 0.0
	DiscreteImplementation = true
	pricesIdx = zeros(eltype(cPow), D)

	for ω ∈ 1:W

		for o ∈ 1:D
			o1 = o + UoModel_factor * (d - 1) * D
			bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
			bilateral_prices[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))
		end

		if DiscreteImplementation
			prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
			expenditure[exporter_idx] += prices_pow * RN[ω]
			price_index += prices_pow * RN[ω]
		else
			smoothMinIndNew!(pricesIdx, bilateral_prices, D, -100.0)

			for o ∈ 1:D
				prices_pow = pricesIdx[o] * bilateral_prices_pow[o]
				expenditure[o] += prices_pow * RN[ω]
				price_index += prices_pow * RN[ω]
			end
		end

	end

	for o ∈ 2:D
		relative_error = lambda[o, d] / lambda[1, d] - expenditure[o] / expenditure[1]
		absolute_error = lambda[o, d] - expenditure[o] / price_index
		#expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
		expenditure_ratio_residual[o-1] = absolute_error # this version works as well, sum of lambdas = 1, probably better as this is the moment condition in the algo
	end
	return expenditure_ratio_residual[baseIndex-1]

end
function phi_x(x)
	const_e = exp(1)

	if x <= const_e
		return x * log(x) - x + 1
	else
		return (1 / (2 * const_e)) * (x - const_e)^2 + (x - const_e) + 1
	end

end

function preStepGeneralDistributionRNBB(data, counters, globalParams, U_init)
	#A function that   does the full pre-step to get the initial guess of parameters (theta_upper and theta_lower are equal in this case) that corresponds with the supplied U_init
	#it provides also the Frechet Theta_init 
	lambda = data.λData
	L = data.LData
	tau = data.τData

	# unpack counterfactual info 
	tauPrime = counters.τPrime
	LPrime = counters.LPrime

	# unpack parameter info
	thetaIn = globalParams.θHat
	sigma = globalParams.σHat
	baseIndex = globalParams.baseIndex
	counterType = globalParams.counterType
	D = globalParams.D
	UoModel = globalParams.UoModel

	UoModel_factor = UoModel == 1 ? 0 : 1

	W = size(U_init, 1)

	# Step 1: Estimate thetaHat via gravity or prespecified
	if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
		deltaLambda = doubleDiff(lambda)
		deltaTau = doubleDiff(tau)
		meanTau = mean(deltaTau)
		thetaHat = -sum(deltaLambda .* (deltaTau .- meanTau)) / sum(deltaTau .* deltaTau .- meanTau^2)
	elseif thetaIn > 0
		thetaHat = thetaIn
	end

	# Step 2: Estimate baseline wages
	wHat = ones(D)
	iterWagesPreStep!(wHat, L, lambda)
	wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

	# Step 3: Estimate cHat

	UPow_init = zeros(W, D * (UoModel == 1 ? 1 : D))
	UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)

	# use Frechet solution as the initial guess
	AHat_Frechet = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
	cHat_Frechet = AHat_Frechet .^ (-1) # we define c as 1/A 

	lambdaPrimeFrechet = Array{Float64, 2}(undef, D, D)  # initialise
	wPrimeHatFrechet = ones(D)

	if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
		iterWagesTheory!(wPrimeHatFrechet, LPrime, AHat_Frechet, tauPrime, thetaHat, lambdaPrimeFrechet)
		wPrimeHatFrechet[:] = wPrimeHatFrechet ./ wPrimeHatFrechet[baseIndex] # normalise 
	end

	gammaHatFrechet = computeGamma(AHat_Frechet, tau, wHat, thetaHat, sigma, L)
	gammaPrimeHatFrechet = computeGamma(AHat_Frechet, tauPrime, wPrimeHatFrechet, thetaHat, sigma, LPrime)

	κ_Frechet = (gammaHatFrechet[baseIndex] / gammaPrimeHatFrechet[baseIndex])^(sigma / (sigma - 1)) - 1


	cHat = ones(D, D)
	RN = ones(W)
	# force cHat[1,:] =1
	normalizeAs = true
	upper_lower = [-1, 1]

	gammaHat = ones(D, 2)
	gammaPrimeHat = ones(D, 2)
	Aod_initial = ones(D, D, 2)

	δ_max = globalParams.δ_ref

	for i_upper_lower in 1:2
		# We restrict to Gains from Trade Counterfactual. For generalization, we need to solve for all ds at the same time.

		# First find the RN derivative and the A_ods for d = baseIndex

		function cHat_baseIndex_solver_func!(cHat_x)
			return gmFunction_baseIndexBB!(sigma, tau, UPow_init, wHat, cHat_Frechet[:, baseIndex] .^ (1.0 / thetaHat), cHat_x[1], cHat_x[2], lambda, baseIndex, UoModel_factor, upper_lower[i_upper_lower], δ_max, RN, κ_Frechet, thetaHat) # we force the cHat[1,:] =1
		end
		#cHat_solver_results = nlsolve(cHat_baseIndex_solver_func!, vcat((cHat_Frechet[baseIndex, baseIndex]) .^ (1.0 / thetaHat), 0.5), ftol = 1e-3 ) #, autodiff = :forward

		cHat_solver_results = bboptimize(cHat_baseIndex_solver_func!, vcat((cHat_Frechet[baseIndex, baseIndex]) .^ (1.0 / thetaHat), 0.5); Method = :borg_moea,
			FitnessScheme = ParetoFitnessScheme{3}(is_minimizing = true),
			SearchRange = [(0.8, 2), (0, 1.0)], NumDimensions = 2, ϵ = 0.05,
			MaxSteps = 100, TraceInterval = 1.0)

		#@show cHat_solver_results
		#@show cHat_solver_results.residual_norm ## check it convergedqw2wa
		#@show cHat_baseIndex_solver_func!(cHat_solver_results.zero)
		cHat[:, baseIndex] .= cHat_Frechet[:, baseIndex] .^ (1.0 / thetaHat)

#		pf = pareto_frontier(cHat_solver_results)
#		best_obj_κ, idx_obj_κ = findmin(map(elm -> fitness(elm)[3], pf))
#		best_κ_solution = BlackBoxOptim.params(pf[idx_obj_κ])

#		cHat[baseIndex, baseIndex] = abs.(best_κ_solution)[1]
#		@show cHat_baseIndex_solver_func!(best_κ_solution)
		cHat[baseIndex, baseIndex] = abs.(best_candidate(cHat_solver_results))[1]
		@show cHat_baseIndex_solver_func!(best_candidate(cHat_solver_results))

#α_RN = abs.(cHat_solver_results.zero)[D:D+2]
		#K_RN = abs.(cHat_solver_results.zero)[D+3]
		@show cHat[:, baseIndex]
		#@show α_RN
		#@show K_RN

		# Actually this is needed only for Uo model
		for d ∈ 1:D
			if d != baseIndex
				function cHat_solver_func!(cHat_x)
					return gmFunction_d!(sigma, tau, UPow_init, wHat, cHat_Frechet[:, d] .^ (1.0 / thetaHat), cHat_x[1], lambda, d, UoModel_factor, RN, baseIndex) # we force the cHat[1,:] =1
				end
				cHat_solver_results = nlsolve(cHat_solver_func!, vcat((cHat_Frechet[baseIndex, d])^(1.0 / thetaHat))) #, autodiff = :forward
				@show cHat_solver_results.residual_norm ## check it converged
				cHat[:, d] .= cHat_Frechet[:, d] .^ (1.0 / thetaHat)
				cHat[baseIndex, d] = abs.(cHat_solver_results.zero)[1]
				@show cHat[:, d]
			end
		end

		GC.gc()


		# Step 4: Estimate wPrimeHat
		#lambdaPrime = Array{Float64,2}(undef, D, D)  # initialise
		lambdaPrime = zeros(D, D)
		wPrimeHat = ones(D)


		if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
			function wPrimeHat_solver_func!(wHatPrime_x)
				func_output = gmFunction!(sigma, tauPrime, UPow_init, wHatPrime_x, cHat, lambda, UoModel_factor, RN)

				wage_moment = zeros(Float64, D)
				for o ∈ 1:D
					wage_moment[o] = sum(func_output.expenditure_ratio[o, :] .* wHatPrime_x .* LPrime) - wHatPrime_x[o] * LPrime[o]
				end

				return wage_moment
			end

			wPrimeHat_solver_results = nlsolve(wPrimeHat_solver_func!, wHat)

			@show wPrimeHat_solver_results.residual_norm ## check it converged

			wPrimeHat = abs.(wPrimeHat_solver_results.zero)
			wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 

			lambdaPrime = (gmFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambda, UoModel_factor, RN)).expenditure_ratio
		else
			for o ∈ 1:D
				lambdaPrime[o, o] = 1
			end
		end

		price_index = ones(D)
		price_index_Prime = ones(D)

		price_index[:] .= ((gmFunction!(sigma, tau, UPow_init, wHat, cHat, lambda, UoModel_factor, RN)).price_index)[:]
		price_index_Prime[:] .= ((gmFunction!(sigma, tauPrime, UPow_init, wPrimeHat, cHat, lambdaPrime, UoModel_factor, RN)).price_index)[:]

		# Step 5: Compute gammaHat and gammaPrimeHat


		gammaHat[:, i_upper_lower] = deepcopy((price_index ./ (wHat .* L)) .^ (1 / sigma))
		gammaPrimeHat[:, i_upper_lower] = deepcopy((price_index_Prime ./ (wPrimeHat .* LPrime)) .^ (1 / sigma))

		# we were using above cHat^(1/theta), for numerical efficiency. We adjust the power before returning the values.
		@. cHat[:] = cHat[:] .^ (thetaHat)
		Aod_initial[:, :, i_upper_lower] = deepcopy(cHat_Frechet[:, :] ./ cHat[:, :])
	end
	@show (gammaHat[baseIndex, 1]/gammaPrimeHat[baseIndex,1])^(sigma / (sigma - 1)) - 1
	@show (gammaHat[baseIndex, 2]/gammaPrimeHat[baseIndex,2])^(sigma / (sigma - 1)) - 1
	
	@show gammaHat[:, 1]
	@show gammaHat[:, 2]
	@show Aod_initial[:, :, 1]
	@show Aod_initial[:, :, 2]

	output = (μHat = 1 ./ thetaHat,
		wHat = wHat[:],
		cHat = cHat_Frechet,
		λPrime = lambdaPrimeFrechet,
		wPrimeHat = wPrimeHatFrechet[:],
		γHat = gammaHatFrechet[:],
		γPrimeHat = gammaPrimeHatFrechet[:],
		Aod_initial = ones(D^2),
		wPrimeHat_upper = copy(wPrimeHatFrechet[:]),
		γHat_upper = copy(gammaHat[:, 2]),
		γPrimeHat_upper = copy(gammaPrimeHat[:, 2]),
		Aod_initial_upper = reshape(Aod_initial[:, :, 2], D^2)[:],
		wPrimeHat_lower = copy(wPrimeHatFrechet[:]),
		γHat_lower = copy(gammaHat[:, 1]),
		γPrimeHat_lower = copy(gammaPrimeHat[:, 1]),
		Aod_initial_lower = reshape(Aod_initial[:, :, 1], D^2)[:],
	)


	return output
end

function gmFunction_baseIndexBB!(σ, τ, UPow_init, w_, cPowFrechet, cPow_, α_m_, lambda, d, UoModel_factor, lower_bound, δ_max, m, κ_Frechet, thetaHat)
	# encountered issues as solver was trying negative values so used abs. maybe there is a better solution
	w = abs.(w_)
	D = size(τ, 1) # num countries 
	W = size(UPow_init, 1) # num draws (or goods)

	cPow = zeros(D)
	cPow[:] = cPowFrechet[:]
	cPow[d] = abs.(cPow_)

	α_m = abs.(α_m_)

	bilateral_prices_pow = zeros(eltype(cPow), D)
	bilateral_prices = zeros(eltype(cPow), D)
	prices_pow = 0.0
	expenditure = zeros(eltype(cPow), D)
	expenditure_ratio_residual = zeros(eltype(cPow), D)

	price_index = 0.0
	exporter_idx = 0.0
	autarky_price_index = 0
	m_expectation = 0
	DiscreteImplementation = true
	pricesIdx = zeros(eltype(cPow), D)

	phi_total = 0
	for o ∈ 1:D
		phi_total += (w[o] * τ[o, d] * cPow[o])^(-thetaHat)
	end
	phi_d = (w[d] * τ[d, d] * cPow[d])^(-thetaHat)

	phi_rw = phi_total - phi_d

	dirac_expectation = thetaHat * (phi_rw * phi_d) / phi_total

	phi_rw_pow = phi_rw^(-1 / thetaHat)

	shift_factor_expectation = phi_rw / phi_total




	α_m = min(1 - 0.0001, α_m)

	dirac_exp2 = 0
	dirac_exp3 = 0
	shift_factor_expectation2 = 0
	for ω ∈ 1:W

		for o ∈ 1:D
			o1 = o + UoModel_factor * (d - 1) * D
			bilateral_prices_pow[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))^(1 - σ)
			bilateral_prices[o] = ((w[o] * τ[o, d] * cPow[o]) / (UPow_init[ω, o1]))
		end

		#if DiscreteImplementation
		prices_pow, exporter_idx = findmax(bilateral_prices_pow[:]) # this assumes σ >1
		prices_rw_pow, exporter_rw_idx = findmax(bilateral_prices_pow[Not(d)])
		dom_rw_price_ratio = bilateral_prices_pow[d] / prices_rw_pow
		shift_factor = dom_rw_price_ratio < 1 ? 1 : 0
		#m[ω] = α_m[1] * SmoothDirac(0.001, dom_rw_price_ratio - 1) + (dom_rw_price_ratio > K ? α_m[2] : α_m[3])
		#UPow_init[:] = U_init[:] .^ (-1.0 / thetaHat)
		d1 = d + UoModel_factor * (d - 1) * D
		U = UPow_init[ω, d1]^(-thetaHat)
		U_signed = lower_bound == -1 ? U : -log(1 - exp(-U))
		U_pow = U_signed^(1.0 / thetaHat)
		V = exp(-U_signed)
		#U_rw_pow = (bilateral_prices[Not(d)])[exporter_rw_idx]/(w[d] * τ[d, d] * cPow[d])
		U_rw_pow = (bilateral_prices[Not(d)])[exporter_rw_idx] / phi_rw_pow
		V_rw = exp(-U_rw_pow^(thetaHat))
		#m[ω] = α_m[1] * SmoothDirac(0.001, U_pow-U_rw_pow)/dirac_expectation + 1- α_m[1]
		dirac_term = SmoothDirac(0.01, -(V - V_rw))
		m[ω] = α_m * dirac_term + (1 - α_m)
		dirac_exp3 += dirac_term / W
		m_expectation += m[ω] / W
		expenditure[exporter_idx] += prices_pow * m[ω] / W
		price_index += prices_pow * m[ω] / W
		autarky_price_index += bilateral_prices_pow[d] * m[ω] / W
		shift_factor_expectation2 += shift_factor / W
	end

	@. expenditure[:] = expenditure[:] / m_expectation
	price_index = price_index / m_expectation
	autarky_price_index = autarky_price_index / m_expectation
	@. m[:] = m[:] ./ m_expectation
	δ = sum(@.phi_x.(m[:])) / W



	gamma_gamma_prime = price_index / autarky_price_index
	kappa = gamma_gamma_prime^(1 / (σ - 1)) - 1

	@show δ
	@show α_m
	@show kappa
	

	#delta constraint
	#expenditure_ratio_residual[D] = (δ < δ_max ? 0 : 10000)
	#expenditure_ratio_residual[D] = (δ - δ_max)
	#maximize/minimize counterfactual 
	#expenditure_ratio_residual[D+1] = exp(lower_bound*(κ_Frechet - kappa)/κ_Frechet)

	for o ∈ 2:D
		relative_error = lambda[o, d] / lambda[1, d] - expenditure[o] / expenditure[1]
		absolute_error = lambda[o, d] - expenditure[o] / price_index
		#expenditure_ratio_residual[o-1] = lambda[o,d] / lambda[1,d] - expenditure[o] / expenditure[1] # we could have some cases where expenditure on country 1 goods is zero...
		expenditure_ratio_residual[o-1] = absolute_error # this version works as well, sum of lambdas = 1, probably better as this is the moment condition in the algo
	end
	#=expenditure_ratio_residual_baseIndex = vcat(
		expenditure_ratio_residual[d-1]^2,
		(δ < δ_max ? 0 : 10000),
		exp(lower_bound * (κ_Frechet - kappa) / κ_Frechet),
	)
	return expenditure_ratio_residual_baseIndex =#

	return (abs(expenditure_ratio_residual[d-1]) < 0.01 ? abs(expenditure_ratio_residual[d-1]) : 1000* abs(expenditure_ratio_residual[d-1]),
		δ < δ_max ? 0.0 : 10000.0,
		lower_bound * (κ_Frechet - kappa) > 0 ? 100.0 : lower_bound * (κ_Frechet - kappa)
		)

end
