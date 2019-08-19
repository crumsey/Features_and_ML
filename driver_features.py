## INPUT SECTION #########################################################################################################################################################

#Please have all files containing variables and beta named using this format : <file_group_name>_<case_name>.dat

#For example, if the file_group_name for variables is decided as \"var\" and the different cases are for differet angles of attack, e.g., 0, 5, 10, ...
#The files containing variables for different angles of attack should be named as var_0.dat, var_5.dat, var_10.dat and so on

#Similarly, if the file_group_name for beta is decided as "beta" for the same cases mentioned above
#The files containing corresponding beta values for different angles of attack should be named as beta_0.dat, beta_5.dat, beta_10.dat and so on

var_file_name  = "var"
beta_file_name = "beta"
case_names     = ["0"]




# SET THE ORDER OF VARIABLES IN WHICH THE VARIABLES ARE STORED BY THE SOLVER AS A LIST

var_names = ["rho", "wall_dist", "nu_SA", "mu_T", "vort_mag", "upvp"]




# SET VARIABLES NOT AVAILABLE FROM SOLVER (NEEDED TO EVALUATE DERIVED VARIABLES) TO SOME DEFAULT VALUES

na_vars   = {'mu' : 1.0}




# SET DERIVED VARIABLES HERE (STORED AS STRINGS, WILL BE EXECUTED AS COMMANDS LATER) AS A DICTIONARY

derived_vars = {'chi_SA'      : 'nu_SA*rho/mu',
                
                'fv1_SA'      : 'mu_T / (nu_SA+1e-10) / rho',

                'fv2_SA'      : '1.0 - chi_SA/(1.0 + chi_SA*fv1_SA)',

                'vort_SA'     : 'vort_mag + fv2_SA * nu_SA/(0.41*wall_dist+1e-10)**2',

                'r_SA'        : 'nu_SA / (vort_SA+1e-10) / (0.41*wall_dist+1e-10)**2',

                'g_SA'        : '0.3*r_SA**6 + 0.7*r_SA',

                'fw_SA'       : 'g_SA * (65.0/(g_SA**6 + 64.0))**(1./6.)',

                'production'  : '0.1355 * nu_SA * vort_SA',

                'destruction' : '(0.1355/0.41**2 + 2.622*1.5) * fw_SA * mu_T**2/(wall_dist+1e-10)**2/rho**2'}




# SET FEATURES HERE (STORED AS STRINGS, WILL BE EXECUTED AS COMMANDS LATER) AS A LIST

feature_defs = ['rho * vort_mag * wall_dist**2 / (mu_T + mu)',
                
                'chi_SA',
                
                'destruction/(production+1e-10)',
                
                'upvp',
                
                'wall_dist']




# FILTER CONDITIONS
# (Only a single filter condition can be applied for now)

filter_var   = 'wall_dist'                        # Can be an available or a derived variable
filter_type  = '<='                               # Valid options are '>=', '<=', '>', '<', '==', '!='
filter_value = 1.0

#########################################################################################################################################################################

exec(compile(open("src/generate_features.py").read(), 'gen_features', 'exec'))
