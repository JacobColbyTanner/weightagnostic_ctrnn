
import scipy.io as sio
import pickle
import numpy as np
from fitnessFunction_vehicle import fitnessFunction_vehicle

#with open("best_individual_robust", "rb") as f:
#        best_individual_robust = pickle.load(f) 

#sio.savemat('parameters_for_matlab/weights_6_robust.mat', mdict={'parameters': best_individual_robust["params"]})

from os import listdir
from os.path import isfile, join

files_path = "10_neuron_sparse_scan"

files = [f for f in listdir(files_path) if isfile(join(files_path, f))]
i = 0
fitness = np.zeros(400)


for file in files:
        with open(files_path + "/" + file, "rb") as f:
                best_individual = pickle.load(f) 

        
                dirr = 'parameters_for_matlab/sparse_agents/'
                post = file + ".mat"

                name_it = dirr+post

                sio.savemat(name_it, mdict={'file': best_individual["params"]})

                fitness[i] = fitnessFunction_vehicle(        
                best_individual["params"],
                best_individual["ctrnn_size"],
                best_individual["ctrnn_step_size"],
                best_individual["bv_duration"],
                best_individual["bv_distance"],
                best_individual["bv_step_size"],
                best_individual["transient_steps"],
                best_individual["discrete"],
                )
                i += 1

                print(i)


dirr = 'parameters_for_matlab/sparse_agents/'
post = "fitness" + ".mat"

name_it = dirr+post

sio.savemat(name_it, mdict={'fitness': fitness})