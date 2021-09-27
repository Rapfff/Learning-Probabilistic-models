import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples.examples_models import *

def random_model(nb_states, alphabet, model_type, training_set, log_like_org=0):
    if model_type == 'MCGT':
        return modelMCGT_random(nb_states,alphabet)

def random_search(nb_states, alphabet, model_type, training_set, log_like_org=0):
    if model_type== 'MCGT':
        best_model = modelMCGT_random(nb_states,alphabet)
        best_like = abs(log_like_org- best_model.logLikelihood(training_set))
        for i in range(100):
            temp_model = modelMCGT_random(nb_states,alphabet)
            temp_like = abs(log_like_org- temp_model.logLikelihood(training_set))
            if temp_like<best_like:
                best_model, best_like=temp_model, temp_like
        return best_model

def eq_model(nb_states, alphabet, model_type, training_set, log_like_org=0):
    if model_type == 'MCGT':
        return modelMCGT_equiprobable(nb_states, alphabet)