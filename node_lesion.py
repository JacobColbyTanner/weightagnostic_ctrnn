from fitnessFunction_vehicle import fitnessFunction_vehicle
import pickle
import numpy as np
import scipy.io as sio

num_neurons = 6


with open("best_individual_robust", "rb") as f:
    best_individual = pickle.load(f) 

#params = best_individual["params"]




lesion_fitness = np.zeros(num_neurons)

# A start to the ablation analysis
for i in range(num_neurons):

    params = np.array(best_individual["params"])
    index = num_neurons**2
    
    weights = params[0:index]
    weights = weights.reshape((num_neurons,num_neurons))

    weights[:,i] = 0.4
    weights[i,:] = 0.4
    weights = weights.reshape(num_neurons**2)
    
    modified_params = np.append(weights,params[index:len(params)])


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
    print(num_neurons)
    print(lesion_fitness[i])  



sio.savemat('parameters_for_matlab/lesion_node_parameters_robust6.mat', mdict={'lesion_matrix': lesion_fitness})




