from CTRNN import CTRNN
from fitnessFunction_vehicle import fitnessFunction_vehicle
import numpy as np
import braitenberg as bv
import matplotlib.pyplot as plt
import random

import pickle

def simulate(ctrnn_parameters, ctrnn_size, ctrnn_step_size, duration, distance, bv_step_size, transient_steps, discrete=True):
    time = np.arange(0.0, duration, bv_step_size)

    ctrnn = CTRNN(size=ctrnn_size, step_size=ctrnn_step_size)

    ctrnn.set_params(ctrnn_parameters, discrete=discrete)

    print("CTRNN Network weights")
    print(ctrnn.weights)
    print(np.sum(np.abs(ctrnn.weights)))

    #Run until transient dynamics are gone
    ctrnn_input = np.zeros(ctrnn_size)

    for i in range(transient_steps):
        ctrnn.euler_step(ctrnn_input)

    bearing = np.arange(0.0, 2 * np.pi, np.pi / 4)

    # Create stimuli in environment
    steps = 0
    finaldistance = 0
    ii = 0
    agentpos = np.zeros((len(bearing),len(time),2))
    foodpos = np.zeros((len(bearing),2))
    for angle in bearing:
        # Create the agent body
        body = bv.Agent()
        food = bv.Food(distance, angle)
        foodpos[ii] = food.pos()
        ctrnn_input = np.zeros(ctrnn_size)


        j = 0
        for t in time:
            # Set neuron input as the sensor activation levels
            ctrnn_input[-2:] = body.sensor_state()
            agentpos[ii][j] = body.pos().squeeze()
            # Update the nervous system based on inputs
            for i in range(int(bv_step_size/ctrnn_step_size)):
                ctrnn.euler_step(ctrnn_input)
            # Update the body based on nervous system activity
            states = ctrnn.outputs[0:2]
            if np.isnan(np.sum(states)):
                return 0.0

            # print(states.shape)
            motorneuron_outputs = states
            body.step(food, motorneuron_outputs, bv_step_size)
            # Store current body position
            
            j += 1
        ii += 1
        

    for i in range(len(bearing)):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        x = agentpos[i,0,0]
        y = agentpos[i,0,1]

        plt.plot(agentpos[i,:,0],agentpos[i,:,1], color=color)
        plt.plot(foodpos[i, 0], foodpos[i, 1],'o', color=color)

    plt.savefig(f"{ctrnn_size}_neuron_all4.png")
    plt.clf()


if __name__ == "__main__":
    with open("best_individual4", "rb") as f:
        best_individual = pickle.load(f) 
        
    print(
        fitnessFunction_vehicle(        
            best_individual["params"],
            best_individual["ctrnn_size"],
            best_individual["ctrnn_step_size"],
            best_individual["bv_duration"],
            best_individual["bv_distance"],
            best_individual["bv_step_size"],
            best_individual["transient_steps"],
            best_individual["discrete"],
            ),
        )

    simulate(
        best_individual["params"],
        best_individual["ctrnn_size"],
        best_individual["ctrnn_step_size"],
        best_individual["bv_duration"],
        best_individual["bv_distance"],
        best_individual["bv_step_size"],
        best_individual["transient_steps"],
        best_individual["discrete"],
        )



