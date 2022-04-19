from CTRNN import CTRNN
import numpy as np
import braitenberg as bv
import matplotlib.pyplot as plt
import random

def simulate(ctrnn_parameters, ctrnn_size, step_size):
    duration = 20
    time = np.arange(0.0, duration, step_size)

    ctrnn = CTRNN(size=ctrnn_size, step_size=step_size)

    ctrnn.set_params(ctrnn_parameters)

    print("CTRNN Network weights")
    print(ctrnn.weights)

    #Run until transient dynamics are gone
    ctrnn_input = np.zeros(ctrnn_size)

    for i in range(100):
        ctrnn.euler_step(ctrnn_input)

    distance = 5
    bearing = np.arange(0.0, 2 * np.pi, np.pi / 4)

    # Create stimuli in environment
    steps = 0
    finaldistance = 0
    ii = 0
    agentpos = np.zeros((len(bearing),len(time),2))
    foodpos = np.zeros((len(bearing),2))
    for angle in bearing:
        # Create the agent body
        body = bv.Agent(ctrnn_size)
        food = bv.Food(distance, angle)
        foodpos[ii] = food.pos()
        ctrnn_input = np.zeros(ctrnn_size)


        j = 0
        for t in time:
            # Set neuron input as the sensor activation levels
            ctrnn_input[-2:] = body.sensor_state()
            agentpos[ii][j] = body.pos().squeeze()
            # Update the nervous system based on inputs
            for i in range(5):
                ctrnn.euler_step(ctrnn_input)
            # Update the body based on nervous system activity
            states = ctrnn.outputs[0:2]
            if np.isnan(np.sum(states)):
                return 0.0

            # print(states.shape)
            motorneuron_outputs = states
            body.step(food, motorneuron_outputs, step_size)
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
        print("X")
        print(x)
        print("Y")
        print(y)

        plt.plot(agentpos[i,:,0],agentpos[i,:,1], color=color)
        plt.plot(foodpos[i, 0], foodpos[i, 1],'o', color=color)
        print("In plots")

    plt.savefig(f"{ctrnn_size}_neuron_all.png")
    print("After save")
    plt.clf()





