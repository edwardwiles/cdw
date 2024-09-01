
include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")

using Distributions, Statistics, Plots, .CounterfactualSensitivity, JLD2

function main(globalParams)
    
    @unpack runs_files, file_name =  globalParams
    @show Dates.format(now(), "HH:MM") # print time  
    Kappa = zeros(length(runs_files))
    for i = 1:length(runs_files)
        run_outputs = load_object(runs_files[i])
        Kappa[i] =  first(run_outputs)
    end
    writedlm(file_name, [Kappa], ',')
	@show Dates.format(now(), "HH:MM") # print time 
end 

## Define global parameters
params = (runs_files= ["cc_output_upper_1_FD_1_Count_1_NC_4_bI2_sG1_lG0_Marg1_ind0_O10_bO50FF_1IndMO_2IS_0ISF_2Aod_1Fmu_1Uo_1MN1P0CS0_4-8-3.csv.jld2",
    "cc_output_upper_2_FD_0_Count_1_NC_17_bI2_sG0_lG0_Marg1_ind0_O5_bO50FF_0IndMO_5IS_0ISF_2Aod_1Fmu_1Uo_1MN1P0CS0Cor0.1_4-8-14.csv.jld2",
],
file_name = file_name

    
)

main_kappas_From_runs(params)
