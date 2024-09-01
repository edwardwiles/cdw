function master_cc_algo(prep_output, params)

	@unpack δ_grid, file_name = prep_output
	if params.OuterLoop == 1

		Θ_upper, κ_upper, Θ_lower, κ_lower = ccOuter(prep_output, params) # run the outer loop 
		writedlm(file_name, [δ_grid κ_lower κ_upper], ',')

		cc_output = (
			Θ_upper = Θ_upper,
			Θ_lower = Θ_lower,
			κ_upper = κ_upper,
			κ_lower = κ_lower,
		)
		save_object(string("cc_output_", file_name, ".jld2"),
		cc_output)

		return cc_output
	else
		Θ, κ, δ_star, nStatus = master_cc_inner_algo(prep_output, params)

		cc_output = (
			Θ_upper = Θ,
			Θ_lower = Θ,
			κ_upper = κ,
			κ_lower = κ,
		)
		save_object(string("cc_output_", file_name, ".jld2"),
		cc_output)

		return cc_output
	end

end

function master_cc_inner_algo(prep_output, params)

	Θ, κ, val, x, nStatus = ccInner(prep_output, params) # run the outer loop 
	δ_star = -val 
	return (Θ, κ, δ_star, nStatus)
end
