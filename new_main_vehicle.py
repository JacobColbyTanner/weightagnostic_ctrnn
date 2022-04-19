import matplotlib.pyplot as plt
import numpy as np

from CTRNN import CTRNN
from EvolSearch import EvolSearch
from fitnessFunction_vehicle import fitnessFunction_vehicle
from simulate import simulate 
from simulate_cont import simulate_cont 

import pickle

# WARNING I AM FILTERING WARNINGS BECUASE PATHOS DOESN'T LIKE THEM
import warnings

warnings.filterwarnings("ignore")

# use_best_individual = True
# with open("best_individual", "rb") as f:
#    best_individual = pickle.load(f)

ctrnn_size = 10
pop_size = 50
step_size = 0.05


########################
# Evolve Solutions
########################

genotype_size = ctrnn_size ** 2 + 2 * ctrnn_size


evol_params = {
    "num_processes": 100,
    "pop_size": pop_size,  # population size
    "genotype_size": genotype_size,  # dimensionality of solution
    "fitness_function": lambda x: fitnessFunction_vehicle(
        x, ctrnn_size, step_size
    ),  # custom function defined to evaluate fitness of a solution
    "elitist_fraction": 0.1,  # fraction of population retained as is between generation
    "mutation_variance": 0.1,  # mutation noise added to offspring.
}
initial_pop = np.random.uniform(size=(pop_size, genotype_size))

evolution = EvolSearch(evol_params, initial_pop)
#evolution.step_generation()

best_fitness = []
#best_fitness.append(evolution.get_best_individual_fitness())
#print("best fitness: ", best_fitness)

#print(evolution.get_best_individual())

mean_fitness = []
#mean_fitness.append(evolution.get_mean_fitness())

#save_best_individual = {
#    "params": params,
#    "included_worlds": included_worlds,
#    "num_updates": num_updates,
#    "best_fitness": [],
#    "mean_fitness": [],
# }

for i in range(5):
    evolution.step_generation()
    best_individual = evolution.get_best_individual()
    best_individual_fitness = fitnessFunction_vehicle(
        best_individual, ctrnn_size, step_size
    )

    best_fitness.append(best_individual_fitness)
    mean_fitness.append(evolution.get_mean_fitness())

    print(len(best_fitness), best_fitness[-1], evolution.get_mean_fitness())
#    with open("best_individual11", "wb") as f:
#        pickle.dump(save_best_individual, f)

simulate(evolution.get_best_individual(),ctrnn_size, step_size)