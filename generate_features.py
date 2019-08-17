import numpy as np
from sklearn.utils import shuffle
import os

mu = 1.0

print("")
print("Please have at least two variables in the variable files from all cases")
print("")


# Get the name of the file containing variables from the flow solver
#--------------------------------------------------------------------

filename     = input("Enter the name of the file_group (.dat extension assumed already) containing these variables - order does not matter - (rho, strain_mag, vort_mag, wall_dist, mu, mu_T, nu_SA, upvp) : ")
beta_file    = input("Enter the name of the file_group (.dat extension assumed already) containing target beta distribution : ")
casenames    = list(input("Enter the different case IDs separated by blank spaces : "))

# Final filenames
# <filename>_<casenames[i]>.dat for the i-th case
# <beta_file>_<casenames[i]>.dat for the i-th case

filenames = []
beta_files = []

for i in range(len(casenames)):
	filenames.append("%s_%s.dat"%(filename, casenames[i]))
	beta_files.append("%s_%s.dat"%(beta_file, casenames[i]))


# Get the column indices corresponding to the individual variables in the file
#------------------------------------------------------------------------------

list_indices = list(eval(input("Enter the respective column indices for features in (rho, strain_mag, vort_mag, wall_dist, mu, mu_T, nu_SA)\n\n\
                                        ----------------------------------------------------------------------------------------------\n\
                                                 For example if the file is structured such that        rho is in column 3,\n\
                                                                                                 strain_mag is in column 0,\n\
                                                                                                   vort_mag is in column 5,\n\
                                                                                                  wall_dist is in column 6,\n\
                                                                                                         mu is in column 2,\n\
                                                                                                       mu_T is in column 4,\n\
                                                                                                      nu_SA is in column 1\n\
                                                                                                   and upvp is in column 7\n\
                                                 The input here will be: 3, 0, 5, 6, 2, 4, 1, 7\n\
                                        ----------------------------------------------------------------------------------------------\n\nPlease enter the column indices : ")))




# Get the variables from the file as a matrix and beta from its file as a vector
#--------------------------------------------------------------------------------
var  = []
beta = []
for i in range(len(casenames)):
	var.append(np.loadtxt(filenames[i]))
	beta.append(np.loadtxt(beta_files[i]))
var = np.vstack(var)
beta = np.hstack(beta)




# Randomize the order of data points
#------------------------------------

var, beta = shuffle(var, beta, random_state=0)




# Extract individual columns as the respective variables from the matrix
#------------------------------------------------------------------------

if list_indices[0]>=0:
	rho        = var[:,list_indices[0]]
if list_indices[1]>=0:
	strain_mag = var[:,list_indices[1]]
if list_indices[2]>=0:
	vort_mag   = var[:,list_indices[2]]
if list_indices[3]>=0:
	wall_dist  = var[:,list_indices[3]]
if list_indices[4]>=0:
	mu         = var[:,list_indices[4]]
if list_indices[5]>=0:
	mu_T       = var[:,list_indices[5]]
if list_indices[6]>=0:
	nu_SA      = var[:,list_indices[6]]
if list_indices[7]>=0:
	upvp       = var[:,list_indices[7]]
################################################################
# Introduce new FLOW variables here ----
################################################################


# Filter data based on wall distance if required
#------------------------------------------------

dist_filter = float(input("What wall distance should be the cut-off for the training dataset : "))
filter_indices = wall_dist<dist_filter

if list_indices[0]>=0:
	rho        = var[filter_indices,list_indices[0]]
if list_indices[1]>=0:
	strain_mag = var[filter_indices,list_indices[1]]
if list_indices[2]>=0:
	vort_mag   = var[filter_indices,list_indices[2]]
if list_indices[3]>=0:
	wall_dist  = var[filter_indices,list_indices[3]]
if list_indices[4]>=0:
	mu         = var[filter_indices,list_indices[4]]
if list_indices[5]>=0:
	mu_T       = var[filter_indices,list_indices[5]]
if list_indices[6]>=0:
	nu_SA      = var[filter_indices,list_indices[6]]
if list_indices[7]>=0:
	upvp       = var[filter_indices,list_indices[7]]
################################################################
# Introduce new FLOW variables here ----
################################################################

beta = beta[filter_indices]




# Calculate some more variables based on the variables extracted from file which could be helpful in defining features
#----------------------------------------------------------------------------------------------------------------------

chi_SA     = nu_SA*rho/mu
fv1_SA     = mu_T / (nu_SA+1e-10) / rho
fv2_SA     = 1.0 - chi_SA/(1.0 + chi_SA*fv1_SA)
vort_SA    = vort_mag + fv2_SA * nu_SA/(0.41*wall_dist+1e-10)**2
r_SA       = nu_SA / (vort_SA+1e-10) / (0.41*wall_dist+1e-10)**2
g_SA       = 0.3*r_SA**6 + 0.7*r_SA
fw_SA      = g_SA * (65.0/(g_SA**6 + 64.0))**(1./6.)

production  = 0.1355 * nu_SA * vort_SA
destruction = (0.1355/0.41**2 + 2.622*1.5) * fw_SA * mu_T**2/(wall_dist+1e-10)**2/rho**2

################################################################
# Introduce new DERIVED variables here ----
################################################################



# Set the number of features
#----------------------------
n_features = int(input("Enter the number of features : "))




# Ask for expressions of features one-by-one
#--------------------------------------------

print("Create %d features (indices 0-%d) for use in Machine Learning (will be written as \"features.dat\" in the folder %s)"%(n_features, n_features-1, os.getcwd()))
feature_list  = []
feature_mean  = []
feature_stdev = []
cmd = []

print("")
print("Variable library (subject to availability): rho, strain_mag, vort_mag, wall_dist, mu, mu_T, nu_SA, production, destruction, nu_SA, chi_SA, r_SA, fv1_SA, fv2_SA, vort_SA, g_SA, fw_SA, upvp")
print("")

for i_feature in range(n_features):
    cmd.append(input("Enter the expression for feature %d in terms of the variable library mentioned above : "%i_feature))
    feature_list.append(eval(cmd[-1]))
    feature_mean.append(np.mean(feature_list[-1]))
    feature_stdev.append(np.sqrt(np.mean(feature_list[-1]**2)-feature_mean[-1]**2))
    feature_list[-1] = (feature_list[-1] - feature_mean[-1]) / (feature_stdev[-1])





# Create a matrix of features
#-----------------------------
features = np.vstack(feature_list)
features = features.T




# Store features in features.dat file
#-------------------------------------
np.savetxt("features_shuffled.dat", features)
np.savetxt("beta_target_shuffled.dat", beta)
print("")
print("Features saved to file")




# Create a fortran module to generate features later for injection
#------------------------------------------------------------------
with open("feature_module.f90", "w+") as f:

    f.write("module ML_injection\n")
    f.write("\n")
    f.write("    use nn")
    f.write("    implicit none\n")
    f.write("    contains\n")
    f.write("\n")
    ###########################################################
    # Introduce new FLOW variables in the arguments if needed
    ###########################################################
    f.write("    subroutine eval_beta(rho,strain_mag,vort_mag,mu,nu_SA,wall_dist,upvp,beta)\n")
    f.write("        \n")
    f.write("        implicit none\n")
    f.write("        \n")
    f.write("        integer, parameter                                       :: nfeatures = %d\n"%np.shape(features)[1])
    f.write("        real*8, dimension(:), intent(in)                         :: rho, strain_mag, vort_mag, mu, nu_SA, wall_dist, upvp\n")
    f.write("        real*8, dimension(:), intent(out)                        :: beta\n")
    f.write("        \n")
    f.write("        real*8, dimension(nfeatures,size(rho)) :: features\n")
    f.write("        real*8, dimension(6) :: opt_params\n")
    f.write("        integer :: n_layers, n_weights\n")
    f.write("        integer, dimension(:), allocatable :: n_neurons")
    f.write("        character(len=10) :: act_fn_name\n")
    f.write("        \n")
    f.write("        real*8, dimension(size(rho))      :: upvp, chi_SA, fv1_SA, fv2_SA, vort_SA, r_SA, g_SA, fw_SA, production, destruction\n")
    f.write("        \n")
    f.write("        chi_SA     = nu_SA*rho/mu\n")
    f.write("        fv1_SA     = chi_SA**3/(chi_SA**3+357.911D0)\n")
    f.write("        fv2_SA     = 1.0D0 - chi_SA/(1.0D0 + chi_SA*fv1_SA)\n")
    f.write("        vort_SA    = vort_mag + fv2_SA * nu_SA/(0.41D0*wall_dist)**2\n")
    f.write("        r_SA       = nu_SA / vort_SA / (0.41*wall_dist)**2\n")
    f.write("        g_SA       = 0.3D0*r_SA**6 + 0.7D0*r_SA\n")
    f.write("        fw_SA      = g_SA * (65.0D0/(g_SA**6 + 64.0D0))**(1.0D0/6.0D0)\n")
    f.write("        production  = 0.1355D0 * nu_SA * vort_SA\n")
    f.write("        destruction = (0.1355D0/0.41D0**2 + 2.622D0*1.5D0) * fw_SA * mu_T**2/wall_dist**2/rho**2\n")
    ############################################################
    # Provide expressions for new DERIVED variables here
    ############################################################
    for i_feature in range(len(cmd)):
        f.write("        features(%d,:) = %s\n"%(i_feature+1, cmd[i_feature]))
        f.write("        features(%d,:) = (features(%d,:) - dble(%.15e))/dble(%.15e)\n"%(i_feature+1, i_feature+1, feature_mean[i_feature], feature_stdev[i_feature]))
    f.write("        \n")
    f.write("        open(10, file='nn_config.dat', form='formatted', status='old')\n")
    f.write("        read(10, *) n_layers\n")
    f.write("        allocate(n_neurons(n_layers))\n")
    f.write("        do i=1,n_layers\n")
    f.write("            read(10, *) n_neurons(i)\n")
    f.write("        end do\n")
    f.write("        read(10, *) act_fn_name\n")
    f.write("        read(10, *) n_weights\n")
    f.write("        read(10, *) opt_params(1)\n")
    f.write("        read(10, *) opt_params(2)\n")
    f.write("        read(10, *) opt_params(3)\n")
    f.write("        read(10, *) opt_params(4)\n")
    f.write("        read(10, *) opt_params(5)\n")
    f.write("        read(10, *) opt_params(6)\n")
    f.write("        close(10)\n")
    f.write("        open(20, file='weights.dat', form='formatted', status='old')\n")
    f.write("        allocate(weights(n_weights))\n")
    f.write("        do i=1,n_weights\n")
    f.write("            read(10, *) weights(i)\n")
    f.write("        end do\n")
    f.write("        close(20)\n")
    f.write("        call nn_predict(n_neurons, act_fn_name, 'mse', 'adam', n_weights, weights, nfeatures, size(rho), features, beta, opt_params)\n")
    f.write("        deallocate(n_neurons)\n")
    f.write("        deallocate(weights)\n")
    f.write("    end subroutine eval_features\n")
    f.write("\n")
    f.write("end module ML_injection")

print("Fortran module (with subroutine to evaluate features inside flow solver during injection process) written")


print("")
