from experiments.hypo_search import random_model
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append('../EM_and_BW_algorithms')

from src.learning.Estimation_algorithm_MCGT import *

def experiment(
        training_set, 
        test_set, 
        model_type, 
        log_like_org, 
        alphabet, 
        nb_states = [4,5,6], 
        hypo_generator = random_model, 
        learning_algorithm_type = 'BW', 
        learning_algorithm_epsilon = 0.01, 
        output_folder = 'results',
        result_file= 'result'):
    '''
    Experiment runs experiment on the original model
        information about the original model we need is 
            model_type,
            training_set,
            testing_set and
            log_like_org
            alphabet
        To generate the hypothesis model we need
            nb_states and
            hypo_generator
        Then we learn the hypothesis model with the learning algorithm
            learning_algorithm_type and
            learning_algorithm_epsilon

    saves results to <output_folder> 

    retuns None;
    '''
    f = open(output_folder+"/"+result_file+".txt",'a')

    for nb in nb_states:
        # Get Hypothisis model 
        hypo_model = hypo_generator(nb, alphabet, model_type, training_set)
        # hypo_model.pprint()
        log_like_hypo = hypo_model.logLikelihood(test_set);
        # Save hypo 

        f.write(str(nb)+', ')
        # f.write("Hypothesis model: ")
        # f.write(str(hypo_model)+'\n') # TODO: meaby get more info about model then just the name
        # f.write("logLikelihood= "+str(log_like_hypo)+"\n")
        f.write(str(log_like_hypo)+', ')
        # Learn the hypo model
        if learning_algorithm_type == 'BW':
            if model_type== 'MCGT':
                # f.write('Algorithm: '+ 'Estimation algorithm MCGT\n')
                algorithm=Estimation_algorithm_MCGT(hypo_model, alphabet)

        else: #TODO
            raise TypeError('Invalid type of algorithm, or this algorithm has not been implemented in the experiment function :D <3')

        algorithm.learn(training_set, output_folder+'/output_model.txt', learning_algorithm_epsilon)
        log_like_hypo_learned = algorithm.h.logLikelihood(test_set);
        # Save learnd model
        # f.write("Learned hypothesis model: ")
        # f.write(str(algorithm.h)+'\n') # TODO: meaby get more info about model then just the name
        # f.write("logLikelihood= "+str(log_like_hypo_learned)+"\n\n")
        f.write(str(log_like_hypo_learned)+',')
        f.write(str(log_like_org)+'\n')

    f.close()
    return