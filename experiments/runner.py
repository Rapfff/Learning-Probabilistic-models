from experiments.hypo_search import random_model
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append('../EM_and_BW_algorithms')

from experiment import experiment
from src.models.MDP import *
from src.models.MCGT import *
from src.models.HMM import *
from src.tools import *

model_types= {MCGT: 'MCGT', HMM: 'HMM', MDP: 'MDP'}


def run_experiment(original_models, 
    size_training_set= 100, 
    len_training_set=7, 
    size_test_set=100, 
    len_test_set= 7, 
    nb_states= [4,5,6], 
    iterations= 1,
    hypo_generator= random_model, 
    split_training_set = 0,
    hypo_generator_args = dict(),
    learning_algorithm_type= 'BW', 
    learning_algorithm_epsilon= 0.01, 
    output_folder= 'results',
    result_file= 'result'
    ):
    os.makedirs(output_folder, exist_ok=True)
    f = open(output_folder+"/"+result_file+".txt",'w')
    for original_model in original_models:
        alphabet= original_model.observations();

        # Get training and test set
        training_set_hypo= False
        if split_training_set > 0:
            if split_training_set>1:
                raise ValueError("split_training set should be between 0 and 1.")
            training_set= list()
            training_set = generateSet(original_model, int(round(size_training_set*(1-split_training_set), 0)), len_training_set)
            training_set_hypo = generateSet(original_model, int(round(size_training_set*(split_training_set), 0)), len_training_set)
        else:
            training_set = generateSet(original_model, size_training_set, len_training_set)
        # saveSet(training_set,output_folder+"/training_set.txt")
        test_set = generateSet(original_model, size_test_set, len_test_set)
        # saveSet(training_set,output_folder+"/test_set.txt")

        model_type=model_types.get(type(original_model));

        # TODO create a case for each model type
        log_like_org= original_model.logLikelihood(test_set);

        # write to result file
        f = open(output_folder+"/"+result_file+".txt",'a')
        f.write("Model to learn: ")
        f.write(original_model.name+'\n')
        f.write("Training_set: "+str(size_training_set)+" sequences of "+str(len_training_set)+" observations\n")
        f.write("Testing_set: "+str(size_test_set)+" sequences of "+str(len_test_set)+" observations\n")
        f.write("logLikelihood of original model: "+ str(log_like_org)+"\n")
        f.write("Observation alphabet: "+ str(alphabet)+"\n")
        f.write("Learning algorithm "+ learning_algorithm_type+ ", epsilon: "+ str(learning_algorithm_epsilon)+'\n\n')
        # TODO: add search algorithm for hypo

        f.close();



        experiment(training_set, test_set, model_type, original_model.name, log_like_org, alphabet, nb_states, iterations, hypo_generator, training_set_hypo, hypo_generator_args, learning_algorithm_type, learning_algorithm_epsilon, output_folder, result_file)