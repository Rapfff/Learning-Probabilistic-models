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

def smart_random_search(nb_states, alphabet, model_type, training_set, log_like_org=0):
    if model_type== 'MCGT':
        lambda_ = 0.5
        best_model = modelMCGT_random(nb_states,alphabet)
        best_like = abs(log_like_org- best_model.logLikelihood(training_set))
        for i in range(100):
            temp_model = modelMCGT_random(nb_states,alphabet)
            temp_model = mearge_MCGT(lambda_, temp_model, best_model)
            temp_like = abs(log_like_org- temp_model.logLikelihood(training_set))
            if temp_like<best_like:
                best_model, best_like=temp_model, temp_like
        return best_model

def eq_model(nb_states, alphabet, model_type, training_set, log_like_org=0):
    if model_type == 'MCGT':
        return modelMCGT_equiprobable(nb_states, alphabet)

# Helper function
def mearge_MCGT(lambda_, model1, model2):
    '''
    Function that takes two MCGT and calculates: lambda* model1 + (1- lambda)* model2
    
    assumes that model 1 and model 2 have the same number of states 
    and for each state the state array is the same, and the sympol array is the same.
    '''
    new_states=[]
    states1=model1.states
    states2=model2.states
    for i in range(len(states1)):
        state1=states1[i].next_matrix
        state2=states2[i].next_matrix
        proba_transition= [state1[1][i]*lambda_+(1-lambda_)*state2[1][i] for i in range(len(state1[1]))]
        if (state1[1] != state2[1]) or (state1[2] != state2[2]):
            raise ValueError('merge_MCGT assumes that for every state the symbol array and state array are the same')
        new_states.append(MCGT_state([proba_transition, state1[1], state2[2]]))
    return MCGT(new_states,model1.initial_state)