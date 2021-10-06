import os, sys, copy, math, inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import *
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_MCGT import *
from EM_and_BW_algorithms.src.models.MCGT import *
from EM_and_BW_algorithms.src.tools import *

from statistics import fmean, stdev

# UTILITY
base_alphabet = "abcdefghijklmnopqrstuvwxyz"   # Base alphabet, will be sliced to needs.  

# PARAMETERS OF EXPERIMENT
num_states = 5
alphabet = base_alphabet[:num_states]
num_sequenes = 4
len_sequence = int( math.sqrt(num_states) ) * len( alphabet )
num_tests = 1
learn_algo = Estimation_algorithm_MCGT
model_generator = modelMCGT_random

# GENERATE MODELS AND TRAININGSET
generating_model = model_generator(num_states, alphabet)
hypo_model = model_generator(num_states, alphabet)
training_set = generateSet( generating_model, num_sequenes, len_sequence )

# LEARN WHOLE TRAININGSET 
algo = learn_algo( hypo_model, hypo_model.observations() )
set_trained_model = algo.learn( training_set )
set_trained_logLiklihood = set_trained_model.logLikelihood( training_set )

# LEARN TRAININGSET SEQUENTIALLY
current_model = hypo_model
for i in range(num_sequenes):
    seq_set = []
    print( 1 )
    seq_set.append( [training_set[0][i]] )
    print( 2 )
    seq_set.append([1])             # Needs to conform to format of training_set
    print( 3 )
    print( seq_set )
    algo = learn_algo( current_model, current_model.observations() )
    print( 4 )
    current_model = algo.learn( seq_set )
    print( 5 )
seq_trained_logLikelihood = current_model.logLikelihood( training_set )

print( "Trained on Set: ", set_trained_logLiklihood )
print( "Trained on Seq: ", seq_trained_logLikelihood )
