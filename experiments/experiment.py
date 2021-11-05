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
        model_name,
        log_like_org, 
        alphabet, 
        nb_states = {4,5,6}, 
        iterations = 1,
        hypo_generator = random_model,
        hypo_training_set= False, 
        hypo_generator_args= dict(),
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
            Optional:
                hypo_training_set (Use a separate set while generating the hypothesis model)
                hypo_generator_args (Extra arguments for the hypothesis generator)
        Then we learn the hypothesis model with the learning algorithm
            learning_algorithm_type and
            learning_algorithm_epsilon

    saves results to <output_folder>/<result_file>

    retuns None;
    '''
    if hypo_training_set== False:
        hypo_training_set= training_set
    f = open(output_folder+"/"+result_file+".txt",'a')
    for nb in nb_states:
        # Keep track of best model so far
        best_log= float('inf');
    
        for i in range(iterations):
            # Get Hypothisis model 
            hypo_model = hypo_generator(nb, alphabet, model_type, hypo_training_set, **hypo_generator_args)
            log_like_hypo = hypo_model.logLikelihood(test_set);

            # Learn the hypo model
            if learning_algorithm_type == 'BW':
                if model_type== 'MCGT':
                    # f.write('Algorithm: '+ 'Estimation algorithm MCGT\n')
                    algorithm=Estimation_algorithm_MCGT(hypo_model, alphabet)

            else: #TODO
                raise TypeError('Invalid type of algorithm, or this algorithm has not been implemented in the experiment function :D <3')

            # algorithm.learn(training_set, output_folder+'/output_model_'+str(nb_states)+'_'+str(i)+'.txt', learning_algorithm_epsilon)
            algorithm.learn(training_set, output_folder+'/output_model.txt', learning_algorithm_epsilon)
            log_like_hypo_learned = algorithm.h.logLikelihood(test_set);
            if abs(log_like_hypo_learned-log_like_org)<best_log:
                best_log= log_like_hypo_learned;
                os.rename(output_folder+'/output_model.txt', output_folder+"/"+result_file+"_best_learrned_model_"+str(model_name)+"_"+str(nb)+".txt")

            # Save results
            f.write(str(nb)+', '+str(hypo_model.logLikelihood(training_set))+', '+str(log_like_hypo)+', '+str(log_like_hypo_learned)+'\n')
    f.close()
    return