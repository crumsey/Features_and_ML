import sys
import numpy as np
sys.path.append("./pyModelAugmentationUM/")
from plotting import *
sys.path.append("./pyModelAugmentationUM/Neural_Network/")
import nn

restart = int(sys.argv[1])
n_iter = int(sys.argv[2])

f = np.loadtxt("features_shuffled.dat").T  # Features
b = np.loadtxt("beta_target_shuffled.dat") # Beta target

n_features = np.shape(f)[0]	  # Automatically calculates the number of features

n_neurons_hidden_layers=[20,20]	  # Two hidden layers with 20 neurons each

nn_params = {}

####### Sets up the network ###############

network    = [n_features]
network.extend(n_neurons_hidden_layers)
network.append(1)
network    = np.asfortranarray(np.array(network))

###########################################


nn_params["network"] = network
nn_params["weights"] = np.asfortranarray(np.random.random((sum((network[0:-1]+1)*network[1:]))))
if restart>0:
	nn_params["weights"] = np.asfortranarray(np.loadtxt("weights_%d.dat"%restart))

print("Number of datapoints : %d"%(np.shape(f)[1]))
print("Number of weights    : %d"%(np.shape(nn_params["weights"])[0]))

nn_params["opt_params"] = {"alpha":1e-3, "beta_1":0.9, "beta_2":0.999, "eps":1e-8, "beta_1t":1.0, "beta_2t":1.0}
    
nn_params["opt_params_array"] = np.asfortranarray(np.array([nn_params["opt_params"]["alpha"],
                                                            nn_params["opt_params"]["beta_1"],
                                                            nn_params["opt_params"]["beta_2"],
                                                            nn_params["opt_params"]["eps"],
                                                            nn_params["opt_params"]["beta_1t"],
                                                            nn_params["opt_params"]["beta_2t"]]))

nn.nn.nn_train(np.asfortranarray(nn_params["network"]),
                        
                       "sigmoid",		# "relu", "sigmoid"
                       "mse",			# "mse"
                       "adam",			# "adam"
                       
                       nn_params["weights"],
                       np.asfortranarray(f),
                       np.asfortranarray(b),

                       10000, 			# batch size
                       n_iter,   		# epochs
                       1.0,			# Training fraction

                       np.asfortranarray(nn_params["opt_params_array"]),
                       1)			# Verbose
            
beta = nn.nn.nn_predict(np.asfortranarray(nn_params["network"]),
                            
                                          "sigmoid",
                                          "mse",
                                          "adam",
                                          
                                    	  nn_params["weights"],
                                          np.asfortranarray(f),

                                          np.asfortranarray(nn_params["opt_params_array"]))

########## Plotting section ######################

np.savetxt("beta_ML.dat", beta)
np.savetxt("weights_%d.dat"%(n_iter+restart), nn_params["weights"])

with open("nn_config.dat","w+") as f:
    f.write("%d\n"%len(n_neurons_hidden_layers))
    for i in range(len(n_neurons_hidden_layers)):
        f.write("%d\n"%n_neurons_hidden_layers[i])
    f.write("%d\n"%np.shape(nn_params["weights"])[0])
    for i in range(6):
       f.write("%E\n"%nn_params["opt_params_array"][i])

myplot(1, b, beta, 'xr', 1.0, None)
myplot(1, [min(b), max(b)], [min(b), max(b)], '-g', 1.0, None)
myfigshow()

##################################################
