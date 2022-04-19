from CTRNN import CTRNN
import numpy as np
import braitenberg as bv


def fitnessFunction_vehicle(ctrnn_parameters, ctrnn, duration, distance, bv_step_size):
    time = np.arange(0.0, duration, bv_step_size)

    ctrnn.set_params(ctrnn_parameters, discrete=True)

    # Run to remove transient dynamics
    ctrnn_input = np.zeros(ctrnn.size)

    for i in range(150):
        ctrnn.euler_step(ctrnn_input)

    distance = 5
    bearing = np.arange(0.0, 2 * np.pi, np.pi / 4)

    # Create stimuli in environment
    steps = 0
    finaldistance = 0
    for angle in bearing:
        food = bv.Food(distance, angle)

        # Create the agent body
        body = bv.Agent()

        for t in time:

            # Set neuron input as the sensor activation levels
            ctrnn_input[-2:] = body.sensor_state()
            # Update the nervous system based on inputs

            for i in range(int(bv_step_size/ctrnn.step_size)):
                ctrnn.euler_step(ctrnn_input)

            # Update the body based on nervous system activity
            states = ctrnn.outputs[0:2]

            if np.isnan(np.sum(states)):
                return 0.0

            # print(states.shape)
            motorneuron_outputs = states
            body.step(food, motorneuron_outputs, bv_step_size)
            finaldistance += body.distance(food)
            steps += 1

    fitness = np.clip(1 - ((finaldistance / steps) / distance), 0, 1)

    return fitness

