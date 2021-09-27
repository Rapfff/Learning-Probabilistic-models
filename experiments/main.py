import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from runner import run_experiment
from examples.examples_models import *
from hypo_search import *

nb_states=[2,2,3,3,4,4,5,5,6,6]
size_training_set, len_training_set, size_test_set, len_test_set = 100, 6, 50, 5;


original_models =[modelMCGT1(), modelMCGT2(), modelMCGT3(), modelMCGT4()]

# Experiments with random models ?
# original_models = [modelMCGT_random(5, ['a', 'b', 'c', 'd', 'e', 'f']), modelMCGT_random(5, ['a', 'b', 'c', 'd', 'e', 'f']), modelMCGT_random(5, ['a', 'b', 'c', 'd', 'e', 'f'])]


# Experiment 1 
#   Random model

# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, random_model, 'Estimation_algorithm_MCGT', None, 'results')

# Expriment 2
#   Random search

# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, random_search, 'Estimation_algorithm_MCGT', None, 'results')

# Experiment 3
#   Equal prop model

# run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, eq_model, 'Estimation_algorithm_MCGT', None, 'results')


# Experiments 4
run_experiment(original_models, size_training_set, len_training_set, size_test_set, len_test_set, nb_states, smart_random_search, 'Estimation_algorithm_MCGT', None, 'results')

