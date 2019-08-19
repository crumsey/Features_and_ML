n_neurons_hidden_layers = [20, 20]  # CAUTION !! DO NOT CHANGE WHEN RESTARTING FROM A NON-ZERO ITERATION
    
nn_params               = {}

nn_params["opt_params"] = {"alpha"   : 1e-3,
                           "beta_1"  : 0.9,
                           "beta_2"  : 0.999,
                           "eps"     : 1e-8,
                           "beta_1t" : 1.0,
                           "beta_2t" : 1.0}

activation_function     = "sigmoid"  # sigmoid or relu
loss_function           = "mse"      # mean squared error
optimizer               = "adam"     # adam is the only option for now

batch_size              = 10000
training_fraction       = 1.0

verbose                 = 1

exec(compile(open("src/train_neural_network.py").read(), 'train_nn', 'exec'))
