
import scipy.io as sio
import pickle


with open("best_individual", "rb") as f:
        best_individual = pickle.load(f) 

sio.savemat('parameters_for_matlab/weights_10_n.mat', mdict={'parameters': best_individual["params"]})