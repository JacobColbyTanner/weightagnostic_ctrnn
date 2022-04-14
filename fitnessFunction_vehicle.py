from CTRNN import CTRNN
import numpy as np
import braitenberg as bv


def fitnessFunction_vehicle(ctrnn_parameters, ctrnn_size, step_size):
    duration = 50
    time = np.arange(0.0, duration, step_size)

    ctrnn = CTRNN(size=ctrnn_size, step_size=step_size)

    num_weights = ctrnn_size ** 2

    new_weights = ctrnn_parameters[:num_weights]
    new = new_weights

    # translate genome into ctrnn parameters
    new = np.zeros(num_weights)

    # force same weights, positive and negative across the network
    # for i in range(num_weights):
    #    if new_weights[i] > 1/3:
    #        new[i] = 1
    #    elif new_weights[i] < -1/3:
    #        new[i] = -1
    #    else:
    #        new[i] = 0

    ctrnn.weights = 2 * (new.reshape((ctrnn_size, ctrnn_size)) - 0.5)
    ctrnn.taus = ctrnn_parameters[num_weights : (num_weights + ctrnn_size)] + 0.0001
    ctrnn.biases = 2 * (
        ctrnn_parameters[(num_weights + ctrnn_size) : (num_weights + 2 * ctrnn_size)]
        - 0.5
    )

    # Create the agent body
    body = bv.Agent(ctrnn_size)

    distance = 5
    bearing = np.arange(0.0, 2 * np.pi, np.pi / 4)

    # Create stimuli in environment
    steps = 0
    finaldistance = 0
    for angle in bearing:
        food = bv.Food(distance, angle)

        ctrnn_input = np.zeros(ctrnn_size)

        for t in time:

            # Set neuron input as the sensor activation levels
            ctrnn_input[-2:] = body.sensor_state()
            # Update the nervous system based on inputs

            ctrnn.euler_step(ctrnn_input)

            # Update the body based on nervous system activity
            states = ctrnn.outputs[0:2]

            # print(states.shape)
            motorneuron_outputs = states
            body.step(food, motorneuron_outputs, step_size)
            finaldistance += body.distance(food)
            steps += 1

    fitness = np.clip(1 - ((finaldistance / steps) / distance), 0, 1)

    return fitness

