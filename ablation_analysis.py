from fitnessFunction_vehicle import fitnessFunction_vehicle
import pickle
import numpy as np
from input_output_difference import input_output_difference
import scipy.io as sio

with open("best_individual_robust", "rb") as f:
    best_individual = pickle.load(f) 

from os import listdir
from os.path import isfile, join

create_output_diff_map = True

#files_path = "10_neuron_sparse_scan"

#files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

#file_index = np.random.randint(low=0, high=200)
#with open(files_path + "/" + files[file_index], "rb") as f:
#    best_individual = pickle.load(f) 

num_neurons = 6
params = best_individual["params"]
lesion_fitness = np.zeros(num_neurons**2)
output_diff_lesion = np.zeros((num_neurons**2,10,10))
# A start to the ablation analysis
for i in range(num_neurons**2):
    modified_params = np.copy(best_individual["params"])
    # Set param to 0 (set to 0.4 becuase it is between 1/3 and 2/3)
    if (modified_params[i] > 2/3) or (modified_params[i] < 1/3):
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
        print(num_neurons**2)
        print(lesion_fitness[i])  

        if create_output_diff_map:
            output_diff_lesion[i,:,:] = input_output_difference(modified_params, best_individual)
    else:
        print("skip no edge")
        if create_output_diff_map:
            output_diff_lesion[i,:,:] = input_output_difference(modified_params, best_individual)
# Checking robustness. Solutions are not robust at all to changes in sign, but they are robust to variations
# of about +-0.5

range = np.arange(-5, 5, 0.1)
robustness = np.zeros(len(range))
i = 0
for multiplier in range:
    
    robustness[i] = fitnessFunction_vehicle(        
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

    print(i) 
    print("mult")
    print(multiplier)
    print(robustness[i])
    i +=1







#dirr = '10_neuron_sparse_scan/edge_lesion_sparse'
#post = str(file_index)+".mat"

#name_it = dirr+post

#sio.savemat(name_it, mdict={'lesion_edge': lesion_fitness})

#dirr = '10_neuron_sparse_scan/robustness_sparse'
#post = str(file_index)+".mat"

#name_it = dirr+post

#sio.savemat(name_it, mdict={'robustness': robustness})

dirr = 'parameters_for_matlab/edge_lesion_robust'
post = ".mat"

name_it = dirr+post

sio.savemat(name_it, mdict={'lesion_edge': lesion_fitness})

dirr = 'parameters_for_matlab/robustness_robust'
post = ".mat"

name_it = dirr+post

sio.savemat(name_it, mdict={'robustness': robustness})

dirr = 'parameters_for_matlab/output_diff_lesion'
post = ".mat"

name_it = dirr+post

sio.savemat(name_it, mdict={'output_diff_lesion': output_diff_lesion})