import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append('../EM_and_BW_algorithms')

from src.learning.Estimation_algorithm_MCGT import *

# TODO: Add epsilon

def experiment(training_set, test_set, model_type, log_like_org, alphabet, nb_states, hypo_generator, learning_algorithm_type, learning_algorithm_epsilon, output_folder):
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
    f = open(output_folder+"/results.txt",'a')

    for nb in nb_states:
        # Get Hypothisis model 
        hypo_model = hypo_generator(nb, alphabet, model_type, training_set)
        hypo_model.pprint()
        log_like_hypo = hypo_model.logLikelihood(test_set);
        # Save hypo 

        f.write(str(nb)+', ')
        # f.write("Hypothesis model: ")
        # f.write(str(hypo_model)+'\n') # TODO: meaby get more info about model then just the name
        # f.write("logLikelihood= "+str(log_like_hypo)+"\n")
        f.write(str(log_like_hypo)+', ')
        # Learn the hypo model
        if learning_algorithm_type == 'Estimation_algorithm_MCGT':
            if model_type== 'MCGT':
                # f.write('Algorithm: '+ 'Estimation algorithm MCGT\n')
                algorithm=Estimation_algorithm_MCGT(hypo_model, alphabet)
            else: # TODO: other algorithms
                raise TypeError(learning_algorithm_type+' is for MCGT not '+ model_type)

        else: #TODO
            raise TypeError('Invalid type of algorithm, or this algorithm has not been implemented in the experiment function :D <3')

        algorithm.learn(training_set)
        log_like_hypo_learned = algorithm.h.logLikelihood(test_set);
        # Save learnd model
        # f.write("Learned hypothesis model: ")
        # f.write(str(algorithm.h)+'\n') # TODO: meaby get more info about model then just the name
        # f.write("logLikelihood= "+str(log_like_hypo_learned)+"\n\n")
        f.write(str(log_like_hypo_learned)+',')
        f.write(str(log_like_org)+'\n')

    f.close()
    return