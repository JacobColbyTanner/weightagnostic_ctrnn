from CTRNN import CTRNN
import numpy as np
import braitenberg as bv


def fitnessFunction_vehicle(ctrnn_parameters, ctrnn_size, ctrnn_step_size, duration, distance, bv_step_size, transient_steps):
    time = np.arange(0.0, duration, bv_step_size)

    ctrnn = CTRNN(size=ctrnn_size, step_size=ctrnn_step_size)
    ctrnn.set_params(ctrnn_parameters, discrete=True)

    # Run to remove transient dynamics
    ctrnn_input = np.zeros(ctrnn.size)

    for i in range(transient_steps):
        ctrnn.euler_step(ctrnn_input)

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

    fitness = np.clip(1 - ((finaldistance / steps) / distance), 0, 1) - 0.2*np.sum(np.abs(ctrnn.weights))/(ctrnn.size**2)

    return fitness

