
include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")

using Distributions, Statistics, Plots, .CounterfactualSensitivity, JLD2

function main_post_cc(PostCCParams)
	@unpack cc_run_file_name, runLFD, runLFDCounterFactual = PostCCParams
	# load cc outputs 
	cc_outputs = load_object(cc_run_file_name)

	@unpack prep_output, params, Θ_upper, κ_upper, Θ_lower, κ_lower, LFD_upper, LFD_lower = cc_outputs


	@show Dates.format(now(), "HH:MM") # print time  
	if runLFD == 1
		runLFD(Θ_upper, Θ_lower, LFD_upper, LFD_lower, prep_output, params)
	end

	if runLFDCounterFactual == 1
		LFDCounterFactual(Θ_upper, Θ_lower, LFD_upper, LFD_lower, prestep_output, prep_output, params)
	end
	@show Dates.format(now(), "HH:MM") # print time 
end

## Define parameters for the post run
params = (
	cc_run_file_name = "test",
	# Post CC Optimization tests
	runLFD = 1, # caclulates the correlation between pro
	runLFDCounterFactual = 1,
)

main_post_cc(params)
