from fitnessFunction_vehicle import fitnessFunction_vehicle
import pickle
import numpy as np
import scipy.io as sio

with open("best_individual", "rb") as f:
    best_individual = pickle.load(f) 

params = best_individual["params"]
lesion_fitness = np.zeros(len(params))
# A start to the ablation analysis
for i in range(len(params)):
    modified_params = np.copy(best_individual["params"])
    # Set param to 0 (set to 0.4 becuase it is between 1/3 and 2/3)
    modified_params[i] = 0.4

    lesion_fitness[i] =fitnessFunction_vehicle(        
        modified_params,
        best_individual["ctrnn_size"],
        best_individual["ctrnn_step_size"],
        best_individual["bv_duration"],
        best_individual["bv_distance"],
        best_individual["bv_step_size"],
        best_individual["transient_steps"],
        best_individual["discrete"],
        )
    print(i)
    print("of")
    print(len(params))
    print(lesion_fitness[i])  


# Checking robustness. Solutions are not robust at all to changes in sign, but they are robust to variations
# of about +-0.5

range = np.arange(-10, 10, 1)
robustness = np.zeros(len(range))
num = np.arange(0,len(range),1)
for multiplier in range:
    index = num[multiplier]
    robustness[index] = fitnessFunction_vehicle(        
        best_individual["params"],
        best_individual["ctrnn_size"],
        best_individual["ctrnn_step_size"],
        best_individual["bv_duration"],
        best_individual["bv_distance"],
        best_individual["bv_step_size"],
        best_individual["transient_steps"],
        best_individual["discrete"],
        multiplier
        )
        
    print(robustness[index])




sio.savemat('parameters_for_matlab/lesion_parameters.mat', mdict={'lesion_matrix': lesion_fitness})

sio.savemat('parameters_for_matlab/robustness.mat', mdict={'robust': robustness})