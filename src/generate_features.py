import numpy as np
import sys
from sklearn.utils import shuffle
import os




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a function to get confirmation for different questions
#---------------------------------------------------------------

def get_confirmation(question):
    
    while(1):
        
        choice = input(question)
        
        if choice=='y' or choice=='Y':
            print("")
            break
        
        elif choice=='n' or choice=='N':
            print("")
            sys.exit(0)
        
        else:
            print("")
            print("INVALID INPUT: Please select 'y' or 'n'")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get the name of the files containing variables generated by the flow solver
#-----------------------------------------------------------------------------

var_file_names  = []
beta_file_names = []

question = "\nPlease check if these are the files you want to import data from\n\n"

for i in range(len(case_names)):
    
    var_file_names.append("input_files/%s_%s.dat"%(var_file_name, case_names[i]))
    beta_file_names.append("input_files/%s_%s.dat"%(beta_file_name, case_names[i]))
    
    question = question + "\t%50s\t\t%50s\n"%(var_file_names[i], beta_file_names[i])

question = question + "\nPlease confirm (y/n) : "

get_confirmation(question)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get the variables from the file as a matrix and beta from its file as a vector
#--------------------------------------------------------------------------------
var  = []
beta = []

for i in range(len(case_names)):
	var.append(np.loadtxt(var_file_names[i]))
	beta.append(np.loadtxt(beta_file_names[i]))

var = np.vstack(var)
beta = np.hstack(beta)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Randomize the order of data points
#------------------------------------

sys.stdout.write("Randomizing variables and beta ... ")
var, beta = shuffle(var, beta, random_state=0)
sys.stdout.write("Done\n\n")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Extract individual columns as the respective variables from the matrix
#------------------------------------------------------------------------

question = "\nPlease check if these are the variables (in this order) provided by the variables file from the solver\n\n"

for column_id in range(len(var_names)):

    exec("%s = var[:,%d]"%(var_names[column_id], column_id))

    question = question + "\t\t%s\n"%var_names[column_id]

question = question + "\nPlease confirm (y/n) : "

get_confirmation(question)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get indices satisfying filter conditions
#------------------------------------------

sys.stdout.write("Filtering the data such that %s %s %s ... "%(filter_var, filter_type, filter_value))
exec("filter_indices = %s%s%s"%(filter_var, filter_type, filter_value))


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Apply filter conditions to the data and beta
#----------------------------------------------

for column_id in range(len(var_names)):

    exec("%s = %s[filter_indices]"%(var_names[column_id], var_names[column_id]))

beta = beta[filter_indices]
sys.stdout.write("Done\n\n")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate and store derived variables
#--------------------------------------

for key in na_vars:
    exec("%s=%s"%(key, na_vars[key]))

for key in derived_vars:
    exec("%s=%s"%(key, derived_vars[key]))


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate and store features alongwith their mean and standard deviations
#--------------------------------------------------------------------------

question = "\nPlease check if these are the features you wanted\n"

feature_list  = []
feature_mean  = []
feature_stdev = []

for feature_id in range(len(feature_defs)):
    
    feature_list.append(eval(feature_defs[feature_id]))
    
    question = question + "\n\t\t%s\n"%feature_defs[feature_id]
    
    feature_mean.append(np.mean(feature_list[-1]))
    feature_stdev.append(np.sqrt(np.mean(feature_list[-1]**2)-feature_mean[-1]**2))
    
    feature_list[-1] = (feature_list[-1] - feature_mean[-1]) / (feature_stdev[-1])

question = question + "\nPlease confirm (y/n) : "

get_confirmation(question)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a matrix of features
#-----------------------------

features = np.vstack(feature_list)
features = features.T


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Store features and beta in a dat file
#---------------------------------------

sys.stdout.write("Saving randomized and filtered features and corresponding beta files to output_files folder ... ")
np.savetxt("input_files/features_shuffled.dat", features)
np.savetxt("input_files/beta_target_shuffled.dat", beta)
sys.stdout.write("Done\n\n")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a fortran module to generate features later for injection
#------------------------------------------------------------------

with open("output_files/.ml_injection_1", "w+") as f:

    f.write("module ML_injection\n")
    
    f.write("\n")
    
    f.write("    use nn")
    f.write("    implicit none\n")
    f.write("    contains\n")
    
    f.write("\n")
    
    f.write("    subroutine eval_beta(n_data,&\n")
    for i in range(len(var_names)):
        f.write("                         %s,&\n"%var_names[i])
    f.write("                         beta)\n")
    
    f.write("        \n")

    f.write("        implicit none\n")

    f.write("        \n")

    f.write("        integer, parameter                     :: n_features = %d\n"%np.shape(features)[1])

    f.write("        integer, intent(in)                    :: n_data\n")

    f.write("        real*8, dimension(n_data), intent(in)  :: ")
    for i in range(len(var_names)):
        f.write("%s"%var_names[i])
        if (i < len(var_names)-1):
            f.write(",&\n                                                  ")
    f.write("\n\n")

    f.write("        real*8, dimension(n_data), intent(out) :: beta\n")
    
    f.write("        \n")
    
    f.write("        real*8, dimension(n_features,n_data)   :: features\n")
    
    f.write("        \n")
    
    f.write("        real*8, dimension(n_data)              :: ")
    counter = 0
    for key in derived_vars:
        if counter==1:
            f.write(",&\n                                                  ")
            f.write("%s"%key)
        else:
            f.write("%s"%key)
            counter = 1
    f.write("\n\n")

    f.close()

with open("output_files/.ml_injection_2", "w+") as g:
    
    for key in derived_vars:
        g.write("        %s = %s\n\n"%(key, derived_vars[key]))

    for feature_id in range(len(feature_defs)):
        g.write("        features(%d,:) = %s\n"%(feature_id+1, feature_defs[feature_id]))
        g.write("        features(%d,:) = (features(%d,:) - dble(%.15e))/dble(%.15e)\n"%(feature_id+1, feature_id+1, feature_mean[feature_id], feature_stdev[feature_id]))
        g.write("\n")

    g.close()

print("Fortran module (with subroutine to evaluate features inside flow solver during injection process) written partially (Run Neural Network to complete the injection module)")

print("")
