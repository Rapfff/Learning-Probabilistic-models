import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from runner import run_experiment
from examples.examples_models import *
from hypo_search import *

# Original models
original_models =[modelMCGT1(), modelMCGT2(), modelMCGT3(), modelMCGT4()]

# # First experimental category
# nb_states=[2, 3, 4, 5, 6]
# nb_iterations= 100;
# size_training_set, len_training_set, size_test_set, len_test_set = 100, 5, 50, 5;
#     # Experiment 0 -  Uniform model
# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, 1, equiprobable_model, result_file= "result_experiment3", output_folder="results/experiments_1/experiment_uniform_model")
#     # Experiment 1 - Random model
# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, random_model, result_file= "result_experiment1", output_folder="results/experiments_1/experiment_random_model")
#     # Experiment 2 - Random search
# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, random_search, result_file= "result_experiment2", output_folder="results/experiments_1/experiment_random_search")
#     # Experiment 3 - Smart random Search
# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, smart_random_search, result_file= "result_experiment4", output_folder="results/experiments_1/experiment_smart_random_search")

# # Second experimental category
nb_states=[2, 3, 4, 5, 6, 7, 8, 9]
nb_iterations= 100;
size_training_set, len_training_set, size_test_set, len_test_set = 1000, 5, 1000, 5;
split_training_set = 0.2;
experiment_folder = "results/experiments_2"
    # Experiment 1 - Random Search with split training dataset
run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, random_search, split_training_set, result_file= "result_experiment1", output_folder=experiment_folder+"/experiment1")
    # Experiment 2 - Smart Random Search with split training dataset
# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, nb_iterations, smart_random_search, split_training_set, result_file= "result_experiment2", output_folder=experiment_folder+"/experiment2")
    # Experiment 3 - Smart Random Search with different value for λ
    # Experiment 4 - Smart Random Search with modified λ
