import sys
import numpy as np
from plotting import *
from Neural_Network import nn


#-----------------------------------------------------------------------------------------------------------
# Get command line inputs
#-------------------------

print("")
restart = int(input("Enter the iteration to restart the training from    (Example: 15000)                           : "))
maxiter = int(input("Enter the iteration you want to run the training to (Example: 15010 to run 10 more iterations) : "))
n_iters = maxiter - restart


#-----------------------------------------------------------------------------------------------------------
# Get features and beta for training
#------------------------------------

f = np.loadtxt("input_files/features_shuffled.dat").T
b = np.loadtxt("input_files/beta_target_shuffled.dat")

n_features = np.shape(f)[0]


#-----------------------------------------------------------------------------------------------------------
# Setup the neural network
#--------------------------

network    = [n_features]
network.extend(n_neurons_hidden_layers)
network.append(1)
network    = np.asfortranarray(np.array(network))


nn_params["network"] = network
nn_params["weights"] = np.asfortranarray(np.random.random((sum((network[0:-1]+1)*network[1:]))))
if restart>0:
	nn_params["weights"] = np.asfortranarray(np.loadtxt("output_files/weights/weights_%d.dat"%restart))

print("")
print("Number of datapoints : %d"%(np.shape(f)[1]))
print("Number of weights    : %d"%(np.shape(nn_params["weights"])[0]))
print("")

nn_params["opt_params_array"] = np.asfortranarray(np.array([nn_params["opt_params"]["alpha"],
                                                            nn_params["opt_params"]["beta_1"],
                                                            nn_params["opt_params"]["beta_2"],
                                                            nn_params["opt_params"]["eps"],
                                                            nn_params["opt_params"]["beta_1t"],
                                                            nn_params["opt_params"]["beta_2t"]]))


#-----------------------------------------------------------------------------------------------------------
# Train the neural network
#--------------------------

nn.nn.nn_train(np.asfortranarray(nn_params["network"]),
               
               activation_function,
               loss_function,
               optimizer,
               
               nn_params["weights"],
               np.asfortranarray(f),
               np.asfortranarray(b),

               batch_size,
               n_iters,
               training_fraction,

               np.asfortranarray(nn_params["opt_params_array"]),
               verbose)
            

#-----------------------------------------------------------------------------------------------------------
# Predict using the trained neural network
#------------------------------------------

beta = nn.nn.nn_predict(np.asfortranarray(nn_params["network"]),
                            
                        activation_function,
                        loss_function,
                        optimizer,
                        
                        nn_params["weights"],
                        np.asfortranarray(f),

                        np.asfortranarray(nn_params["opt_params_array"]))


#-----------------------------------------------------------------------------------------------------------
# Save the weights and predictions to the output_files folder
#-------------------------------------------------------------

np.savetxt("output_files/beta_ML.dat", beta)
np.savetxt("output_files/weights/weights_%d.dat"%(n_iters+restart), nn_params["weights"])


#-----------------------------------------------------------------------------------------------------------
# Plot the predictions w.r.t. inverse
#-------------------------------------

myplot("training_quality_%d"%maxiter, b, beta, 'xr', 1.0, "training")
myplot("training_quality_%d"%maxiter, [min(b), max(b)], [min(b), max(b)], '-g', 1.0, "ideal")
myfig("training_quality_%d"%maxiter, "$$\\beta_{inv}$$", "$$\\beta_{ML}$$", "Training quality", legend=True)
myfigsave(".", "training_quality_%d"%maxiter)
myfigshow()

one=1

#-----------------------------------------------------------------------------------------------------------
# Complete the injection library for Fortran
#--------------------------------------------

with open("output_files/.ml_injection_1", "r") as f:
    
    data1 = f.read()

with open("output_files/.ml_injection_2", "r") as g:
    
    data2 = g.read()

with open("output_files/ml_injection.f90", "w+") as h:
    
    h.write(data1)

    h.write("        \n")
    h.write("        real(dp), intent(in)  :: mu, mach, reynolds\n")
    h.write("        real(dp)              :: kappa, ct3, ct4, cb1, cb2, sigma\n")
    h.write("        integer, parameter     :: n_weights=%d\n"%(len(nn_params["weights"])))
    h.write("        integer, dimension(%d)  :: n_neurons=(/ %d,%d"%(np.shape(n_neurons_hidden_layers)[0]+2,n_features,n_neurons_hidden_layers[0]))
    for i in range(1, np.shape(n_neurons_hidden_layers)[0]):
        h.write(", %d"%(n_neurons_hidden_layers[i]))
    h.write(",%d"%(one))
    h.write(" /)\n")
    h.write("        character(len=10)      :: act_fn_name='%s', loss_fn_name='%s', opt_name='%s'\n"%(activation_function, loss_function, optimizer))
    h.write("\n")
    h.write("        real(dp), dimension(6) :: opt_params=(/ %.15f,&\n"%(nn_params["opt_params_array"][0]))
    for i in range(1,5):
        h.write("                                              %.15f,&\n"%(nn_params["opt_params_array"][i]))
    h.write("                                              %.15f /)\n"%(nn_params["opt_params_array"][5]))
    h.write("\n")
    h.write("        real(dp), dimension(n_weights) :: weights = (/ %.15f,&\n"%(nn_params["weights"][0]))
    for i in range(1, len(nn_params["weights"])-1):
        h.write("                                                     %.15f,&\n"%(nn_params["weights"][i]))
    h.write("                                                     %.15f /)\n"%(nn_params["weights"][len(nn_params["weights"])-1]))
    
    h.write("  kappa=0.41_dp\n")
    h.write("  ct3=1.2_dp\n")
    h.write("  ct4=0.5_dp\n")
    h.write("  cb1=0.1355_dp\n")
    h.write("  cb2=0.622_dp\n")
    h.write("  sigma=2._dp/3._dp\n")
    h.write("\n\n")
    h.write(data2)
    h.write("\n")
    
    h.write("        call nn_predict(n_neurons, act_fn_name, loss_fn_name, opt_name, n_weights, weights, n_features, &\n")
    h.write("                        features, beta, opt_params)\n\n")
    h.write("    end subroutine eval_beta\n\n")
    h.write("end module ML_injection")

    h.close()

print("")
print("Fortran module (with subroutine to evaluate features inside flow solver during injection process) completed")

print("")
