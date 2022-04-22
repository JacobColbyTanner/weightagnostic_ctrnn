import matplotlib.pyplot as plt
import numpy as np

from CTRNN import CTRNN
from EvolSearch import EvolSearch
from fitnessFunction_vehicle import fitnessFunction_vehicle

import pickle

# WARNING I AM FILTERING WARNINGS BECUASE PATHOS DOESN'T LIKE THEM
import warnings

warnings.filterwarnings("ignore")

########################
# Parameters
########################

ctrnn_size = 10
ctrnn_step_size = 0.01
transient_steps = 100
discrete = True

bv_step_size = 0.05
bv_duration = 50
bv_distance = 5


########################
# Evolve Solutions
########################

pop_size = 500
genotype_size = ctrnn_size ** 2 + 2 * ctrnn_size


evol_params = {
    "num_processes": 10,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": lambda x: fitnessFunction_vehicle(
        x, ctrnn_size, ctrnn_step_size, bv_duration, bv_distance, bv_step_size, transient_steps, discrete=discrete
    ),  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generation
    "mutation_variance": 0.1,  # mutation noise added to offspring.
}
percent_zeros = round(np.random.uniform(), 5)
initial_pop = np.random.choice([0.0, 0.5, 1.0], p=[(1-percent_zeros)/2, percent_zeros, (1-percent_zeros)/2], size=(pop_size, genotype_size))
initial_pop[:, ctrnn_size**2:] = np.random.uniform(size=np.shape(initial_pop[:, ctrnn_size**2:]))

evolution = EvolSearch(evol_params, initial_pop)

save_best_individual = {
   "params": None,
   "discrete": discrete,
   "ctrnn_size": ctrnn_size,
   "ctrnn_step_size": ctrnn_step_size,
   "bv_step_size": bv_step_size,
   "bv_duration": bv_duration,
   "bv_distance": bv_distance,
   "transient_steps": transient_steps,
   "best_fitness": [0],
   "mean_fitness": [],
}

num_its = 0
while save_best_individual["best_fitness"][-1] < 0.4 and num_its < 100:
    evolution.step_generation()
    
    save_best_individual["params"] = evolution.get_best_individual()
    
    save_best_individual["best_fitness"].append(evolution.get_best_individual_fitness())
    save_best_individual["mean_fitness"].append(evolution.get_mean_fitness())

    #print(
    #    len(save_best_individual["best_fitness"]), 
    #    save_best_individual["best_fitness"][-1], 
    #    save_best_individual["mean_fitness"][-1]
    #)

    with open("10_neuron_sparse_scan/best_individual_" + str(percent_zeros), "wb") as f:
        pickle.dump(save_best_individual, f)
        
    num_its += 1
