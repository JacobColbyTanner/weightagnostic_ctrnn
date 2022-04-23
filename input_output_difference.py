


import pickle
import numpy as np
from CTRNN import CTRNN
import scipy.io as sio




def input_output_difference(params, best_individual):
        
    ctrnn_parameters = params
    ctrnn_size = best_individual["ctrnn_size"]
    ctrnn_step_size = best_individual["ctrnn_step_size"]
    duration = best_individual["bv_duration"]
    distance = best_individual["bv_distance"]
    bv_step_size = best_individual["bv_step_size"]
    transient_steps = best_individual["transient_steps"]
    discrete = best_individual["discrete"]

    print(duration)  
    print(bv_step_size)    
    duration = 0.5
    transient_steps = 25
    range2 = np.arange(0,5,0.5)
    ii = -1

    output_diff = np.zeros((len(range2),len(range2)))

    for input1 in range2:
        ii +=1
        print(ii)
        jj = -1
        for input2 in range2:
            jj +=1
            
            time = np.arange(0.0, duration, bv_step_size)

            ctrnn = CTRNN(size=ctrnn_size, step_size=ctrnn_step_size)
            ctrnn.set_params(ctrnn_parameters, discrete=discrete)
            ctrnn.weights = ctrnn.weights

            # Run to remove transient dynamics
            ctrnn_input = np.zeros(ctrnn.size)

            for i in range(transient_steps):
                ctrnn.euler_step(ctrnn_input)

            all = np.zeros(len(time))
            tt = 0
            for t in time:

                # Set neuron input as the sensor activation levels
                ctrnn_input[-2:] = [input1, input2]
                # Update the nervous system based on inputs

                for i in range(int(bv_step_size/ctrnn.step_size)):
                    ctrnn.euler_step(ctrnn_input)
                
                # Update the body based on nervous system activity
                x = ctrnn.outputs[0]
                y = ctrnn.outputs[1]
                z = x-y
        
                all[tt] = z
                tt+=1

            output_diff[ii,jj] = np.mean(all)


    #sio.savemat('Analysis/output_diff5.mat', mdict={'output_diff': output_diff})
    return output_diff

if __name__ == "__main__":
    with open("best_individual5", "rb") as f:
        best_individual = pickle.load(f) 