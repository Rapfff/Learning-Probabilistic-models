import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from runner import run_experiment
from examples.examples_models import *
from hypo_search import *

nb_states=[2, 3, 4, 5, 6]
nb_iterations= 100;
size_training_set, len_training_set, size_test_set, len_test_set = 100, 5, 50, 5;


original_models =[modelMCGT1(), modelMCGT2(), modelMCGT3(), modelMCGT4()]


# Experiment 0
run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, random_model, result_file= "result_log")

nb_iterations=2;

# Experiment 1 
#   Random model

# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, random_model, result_file= "result_random_model")

# Expriment 2
#   Random search

# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, random_search, result_file= "result_random_search")

# Experiment 3
#   Equal prop model

# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, equiprobable_model, result_file= "result_equiprobable_model")


# Experiments 4
# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, smart_random_search, result_file= "result_smart_random_search")

