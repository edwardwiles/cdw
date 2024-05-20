function master_cc_algo(prep_output, params)

	@unpack δ_grid, file_name, θ_initial = prep_output

   

	Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(prep_output, params) # run the outer loop 
	writedlm(file_name, [δ_grid κ_lower κ_upper], ',')

	# store the parameters
	writedlm(string("Theta_initial_Frechet", "_", file_name), θ_initial, ',')

	for i ∈ 1:length(δ_grid)
		writedlm(string("Theta_upper_", δ_grid[i], "_", file_name), Θ_upper[:, i], ',')
		writedlm(string("Theta_lower_", δ_grid[i], "_", file_name), Θ_lower[:, i], ',')
	end

	#= To test stuff with specific theta
	δ_grid  = [1] 
	Θ_upper = copy(θ_initial)
	Θ_lower = copy(θ_initial)
	Θ_upper[:,1] = readdlm("Theta_upper_1_Counter_1_countries_4_baseI2_sGrav0_lGrav0_Marg0_NoSc1_ind0_order5_baseOrder50useCDF_1ForceFrechet_0stratify_0IndCDF_0IndMO_5ISampling_0ISF_2_Frechet_4-3-14.csv", ',')
	Θ_lower[:,1] = readdlm("Theta_lower_1_Counter_1_countries_4_baseI2_sGrav0_lGrav0_Marg0_NoSc1_ind0_order5_baseOrder50useCDF_1ForceFrechet_0stratify_0IndCDF_0IndMO_5ISampling_0ISF_2_Frechet_4-3-14.csv", ',')
	=#

	if params.calculateLFD == 1
		LFD_upper = zeros(W, length(δ_grid))
		LFD_lower = zeros(W, length(δ_grid))
		for i ∈ 1:length(δ_grid)
			LFD_upper[:, i] = LFD(Θ_upper[:, i], prep_output, params)
			LFD_lower[:, i] = LFD(Θ_lower[:, i], prep_output, params)
		end
		writedlm(string("LFD_up_", file_name), LFD_upper, ',')
		writedlm(string("LFD_low_", file_name), LFD_lower, ',')
	end
	   
end
