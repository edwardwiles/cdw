
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
	# load cc run outputs 
	cc_lfd_outputs = load_object(cc_run_file_name)

	@unpack setup_output, prestep_output, prep_output, params, cc_output, lfd_output = cc_lfd_outputs

	@show Dates.format(now(), "HH:MM") # print time  
	master_post_cc_lfd(params, setup_output, prestep_output, prep_output, cc_output, lfd_output)
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
