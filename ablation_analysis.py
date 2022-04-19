from fitnessFunction_vehicle import fitnessFunction_vehicle
import pickle
import numpy as np


with open("best_individual3", "rb") as f:
    best_individual = pickle.load(f) 

# A start to the ablation analysis
for i in range(len(params)):
    modified_params = np.copy(best_individual["params"])
    # Set param to 0 (set to 0.4 becuase it is between 1/3 and 2/3)
    modified_params[i] = 0.4
    print(
        fitnessFunction_vehicle(        
            modified_params,
            best_individual["ctrnn_size"],
            best_individual["ctrnn_step_size"],
            best_individual["bv_duration"],
            best_individual["bv_distance"],
            best_individual["bv_step_size"],
            best_individual["transient_steps"],
            ),
            best_individual["params"][i],
            modified_params[i]
        )

# Checking robustness. Solutions are not robust at all to changes in sign, but they are robust to variations
# of about +-0.5
for multiplier in np.arange(-10, 10, 1):
    print(
        fitnessFunction_vehicle(        
            best_individual["params"],
            best_individual["ctrnn_size"],
            best_individual["ctrnn_step_size"],
            best_individual["bv_duration"],
            best_individual["bv_distance"],
            best_individual["bv_step_size"],
            best_individual["transient_steps"],
            multiplier
            ),
            multiplier
        )