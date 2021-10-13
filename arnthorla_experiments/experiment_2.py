import os, sys, copy, math, inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import *
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_MCGT import *
from EM_and_BW_algorithms.src.models.MCGT import *
from EM_and_BW_algorithms.src.tools import *

from statistics import fmean, stdev

def sumAlphas( alpha_matrix ):
    sum = 0
    for i in range(len(alpha_matrix)):
        sum += alpha_matrix[i][-1]
    return sum

def proba_seq( algo, sequence ):
    beta_matrix = algo.computeBetas( sequence )
    return beta_matrix[algo.h.initial_state][0]

# UTILITY
base_alphabet = "abcdefghijklmnopqrstuvwxyz"   # Base alphabet, will be sliced to needs.  

# PARAMETERS OF EXPERIMENT
num_states = 10
num_sequenes = 2
len_sequence = 5 #int( math.sqrt(num_states) ) * len( alphabet )
alphabet = base_alphabet[:len_sequence] #base_alphabet[:num_states]
num_tests = 1
learn_algo = Estimation_algorithm_MCGT
model_generator = modelMCGT_random

# GENERATE MODELS AND TRAININGSET
generating_model = model_generator(num_states, alphabet)
hypo_model = model_generator(num_states, alphabet)
training_set = generateSet( generating_model, num_sequenes, len_sequence )

untrained_logLikelihood = hypo_model.logLikelihood( training_set )

# LEARN WHOLE TRAININGSET 
algo = learn_algo( hypo_model, hypo_model.observations() )
set_trained_model = algo.learn( training_set )
set_trained_logLiklihood = set_trained_model.logLikelihood( training_set )

# LEARN TRAININGSET SEQUENTIALLY
current_model = hypo_model
for i in range(num_sequenes):
    seq_set = []
    seq_set.append( [training_set[0][i]] )
    seq_set.append([1])             # Needs to conform to format of training_set
    algo = learn_algo( current_model, current_model.observations() )

    alpha = algo.computeAlphas( seq_set[0][0] )
    alpha_sum = sumAlphas( alpha )
    #proba_seq = proba_seq( algo, seq_set[0][0] )
    current_model.pprint()
    print( "i: ", i )
    print( "seq: ", seq_set[0][0] )
    print( "alpha: ", alpha )
    print( "alpha_sum: ", alpha_sum )
    #print( "proba_seq: ", proba_seq )
    #print( "log(proba_seq): ", log(proba_seq) )
    if alpha_sum <= 0:
        current_model.pprint()
        print( "seq: ", seq_set[0][0] )
        print( "Sequence cant be learned: i = ", i )
        break
    current_model = algo.learn2( seq_set )
seq_trained_logLikelihood = current_model.logLikelihood( training_set )

print( "Untrained: ", untrained_logLikelihood )
print( "Trained on Set: ", set_trained_logLiklihood )
print( "Trained on Seq: ", seq_trained_logLikelihood )
