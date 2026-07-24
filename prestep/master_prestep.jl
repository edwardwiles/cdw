function master_prestep(data, counters, globalParams)

	# unpack data 
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

	# row_idx: destination to exclude from the estimation/moment sample (Part A, 2026-07-23
	# omit-ROW-destination release). nothing (default, absent from globalParams for every
	# pre-existing caller) reproduces today's square D x D behavior exactly -- named_dest==1:D,
	# and within_transform_rect(lambda)==withinTransform(lambda) bit-exact on a square input, so
	# this branch is dimension-agnostic, not a fork.
	row_idx = get(globalParams, :row_idx, nothing)
	named_dest = row_idx === nothing ? (1:D) : filter(!=(row_idx), 1:D)
	Ddest = length(named_dest)

	# Step 1: Estimate thetaHat via gravity or prespecified.
	# Gravity = OLS of ln λ on ln τ with origin + destination fixed effects; by FWL this is the
	# two-way "within" transform (matches the gravity-moment constraint used in the outer loop).
	# (The previous version used a cell-referenced double-difference, which does NOT equal the
	#  two-way-FE coefficient — see gravity_check.jl.) Restricted to named_dest columns so that
	# excluding ROW as a destination re-estimates theta on the correct rectangular sample.
	if thetaIn == 0 # use gravity to estimate theta if no theta prespecified
		Wlambda = within_transform_rect(lambda[:, named_dest])
		Wtau = within_transform_rect(tau[:, named_dest])
		thetaHat = -sum(Wlambda .* Wtau) / sum(Wtau .* Wtau)
	elseif thetaIn > 0
		thetaHat = thetaIn
	end

	# Step 2: Estimate baseline wages
	wHat = ones(D)
	iterWagesPreStep!(wHat, L, lambda)
	wHat = wHat ./ wHat[baseIndex] # normalise so wage is 1 for specified base country 

	# Step 3: Estimate cHat
	AHat = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (thetaHat)) .* (lambda ./ lambda[1, :]')
	cHat = AHat .^ (-1) # we define c as 1/A 

	# Step 4: Estimate wPrimeHat
	lambdaPrime = Array{Float64, 2}(undef, D, D)  # initialise

	wPrimeHat = ones(D)

	if counterType != 1 # only solve if not autarky, as w undetermined in autarky 
		iterWagesTheory!(wPrimeHat, LPrime, AHat, tauPrime, thetaHat, lambdaPrime)
		wPrimeHat[:] = wPrimeHat ./ wPrimeHat[baseIndex] # normalise 
	end

	# Step 5: compute the A_od (outerscaling), if we want to start from a decentered theta_init != thetaHat
	additional_theta_low = 0
	additional_theta_up = 0

	if globalParams.theta_init == -1
		additional_theta_low = get_theta_from_delta(globalParams.δ_ref, globalParams.UoModel == 1 ? D : D^2, -1) * thetaHat
		additional_theta_up = get_theta_from_delta(globalParams.δ_ref, globalParams.UoModel == 1 ? D : D^2, 1) * thetaHat
	elseif globalParams.theta_init == 0
		additional_theta_low = thetaHat
		additional_theta_up = thetaHat
	else
		additional_theta_low = globalParams.theta_init
		additional_theta_up = globalParams.theta_init
	end

	additional_theta_low = max(additional_theta_low, sigma - 1 + 1)
	additional_theta_up = max(additional_theta_up, sigma - 1 + 1)

	AHat_additional_low = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (additional_theta_low)) .* (lambda ./ lambda[1, :]')
	cHat_additional_low = AHat_additional_low .^ (-1) # we define c as 1/A 

	Aod_initial_low = cHat ./ ((cHat_additional_low) .^ (thetaHat / additional_theta_low))



	# Step 5: Compute gammaHat and gammaPrimeHat
	gammaHat_low = computeGamma(AHat_additional_low, tau, wHat, additional_theta_low, sigma, L)
	gammaPrimeHat_low = computeGamma(AHat_additional_low, tauPrime, wPrimeHat, additional_theta_low, sigma, LPrime)


	AHat_additional_up = (((wHat .* tau) ./ (wHat[1, 1] .* tau[1, :]')) .^ (additional_theta_up)) .* (lambda ./ lambda[1, :]')
	cHat_additional_up = AHat_additional_up .^ (-1) # we define c as 1/A 

	Aod_initial_up = cHat ./ ((cHat_additional_up) .^ (thetaHat / additional_theta_up))



	# Step 5: Compute gammaHat and gammaPrimeHat
	gammaHat_up = computeGamma(AHat_additional_up, tau, wHat, additional_theta_up, sigma, L)
	gammaPrimeHat_up = computeGamma(AHat_additional_up, tauPrime, wPrimeHat, additional_theta_up, sigma, LPrime)


	gammaHat = computeGamma(AHat, tau, wHat, thetaHat, sigma, L)
	gammaPrimeHat = computeGamma(AHat, tauPrime, wPrimeHat, thetaHat, sigma, LPrime)



	# named_dest-restricted A_od initial guesses (true dimension shrink: D*Ddest elements, not
	# D^2 -- the ROW column of these one-time calibration matrices is simply never flattened
	# into the outer free-parameter vector). γHat/γPrimeHat/cHat are left at their full-D
	# destination extent below: γHat/γPrimeHat feed only the inert, unread "old γ_θ" θ-slots
	# (moments_gammanorm.jl: "θ[3:2+D] intentionally NOT READ") and the baseIndex-indexed κ
	# printout above (a per-destination-column quantity, unaffected by other columns'
	# presence/absence) -- cHat is genuinely destination-indexed data consumed downstream
	# (Aod = Aod_θ .* cHat .* ...) and IS sliced here to match Aod_θ's (D, Ddest) shape.
	output = (μHat = 1 ./ thetaHat,
		wHat = wHat[:],
		cHat = cHat[:, named_dest],
		λPrime = lambdaPrime,
		wPrimeHat = wPrimeHat[:],
		γHat = gammaHat[:],
		γPrimeHat = gammaPrimeHat[:],
		Aod_initial = ones(D * Ddest),
		wPrimeHat_upper = copy(wPrimeHat[:]),
		γHat_upper = copy(gammaHat_up[:]),
		γPrimeHat_upper = copy(gammaPrimeHat_up[:]),
		Aod_initial_upper = reshape(Aod_initial_up[:, named_dest], D * Ddest)[:],
		wPrimeHat_lower = copy(wPrimeHat[:]),
		γHat_lower = copy(gammaHat_low[:]),
		γPrimeHat_lower = copy(gammaPrimeHat_low[:]),
		Aod_initial_lower = reshape(Aod_initial_low[:, named_dest], D * Ddest)[:])

	# GT defined baseline -> autarky: 1 - (γ'/γ)^{σ/(σ-1)} (matches counterVal in moments!.jl)
	κ = 1 - (output.γPrimeHat[globalParams.baseIndex] / output.γHat[globalParams.baseIndex])^(globalParams.σHat / (globalParams.σHat - 1))
	κ_upper = 1 - (output.γPrimeHat_upper[globalParams.baseIndex] / output.γHat_upper[globalParams.baseIndex])^(globalParams.σHat / (globalParams.σHat - 1))
	κ_lower = 1 - (output.γPrimeHat_lower[globalParams.baseIndex] / output.γHat_lower[globalParams.baseIndex])^(globalParams.σHat / (globalParams.σHat - 1))

	@show κ
	@show κ_upper
	@show κ_lower

	return output
end
#=
function get_delta_from_theta(theta)
@show theta
	if theta < exp(-1) && theta > 0.0001
		return exp(-1 - theta / (-1 + theta)) * theta^(-1 - theta / (-1 + theta)) *
			   (
				   -2 * exp(2) * theta + 6 * exp(2) * theta^2 - 6 * exp(2) * theta^3 + 2 * exp(2) * theta^4 - exp(theta / (-1 + theta)) * theta^(theta / (-1 + theta)) + 2 * exp(3 + 1 / (-1 + theta)) * theta^(1 + theta / (-1 + theta)) -
				   4 * exp(1 + theta / (-1 + theta)) * theta^(1 + theta / (-1 + theta)) - exp(3 + 1 / (-1 + theta)) * theta^(2 + theta / (-1 + theta)) + 2 * exp(1 + theta / (-1 + theta)) * theta^(2 + theta / (-1 + theta))
			   ) / (2 * (-2 + theta))
	elseif theta <= 1+0.001 && theta >= exp(-1)
		return -1 + theta - log(theta)
	elseif theta < 2-0.0000001 && theta > 1+0.001
		return exp(-(theta / (-1 + theta))) * theta^(-(theta / (-1 + theta))) *
			   (
				   exp(1) - 3 * exp(1) * theta + 3 * exp(1) * theta^2 - exp(1) * theta^3 + 2 * exp(1)^(theta / (-1 + theta)) * theta^(theta / (-1 + theta)) - 3 * exp(theta / (-1 + theta)) * theta^(1 + theta / (-1 + theta)) +
				   exp(theta / (-1 + theta)) * theta^(2 + theta / (-1 + theta)) + 2 * exp(theta / (-1 + theta)) * theta^(theta / (-1 + theta)) * log(theta) - exp(theta / (-1 + theta)) * theta^(1 + theta / (-1 + theta)) * log(theta)
			   ) / (-2 + theta)
	else
		return 10000000
	end
end
=#

function get_delta_from_theta(theta, U_dim, up_low)
	if (1-theta) * up_low < 0 || theta == 0
		return 0
	else
		return 1 - (theta^(U_dim - 1)) * ((U_dim + 1) - U_dim * theta + log(theta))
	end
end

function get_theta_from_delta(delta, U_dim, up_low)



	function theta_solver_func!(x)
		delta_x = get_delta_from_theta(abs(x[1]), U_dim, up_low) - delta
		#delta_x = get_delta_from_theta(abs(x[1])) - delta
		@show x
		@show delta_x
		return delta_x^2
	end

	theta_solver_results = nlsolve(theta_solver_func!, ones(1).- up_low .* 0.0001)
	@show theta_solver_results.residual_norm
	@show theta_solver_results.zero
	theta = abs((theta_solver_results.zero)[1])
	return theta
end

