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
    datasets= dict(),
    splitdatasets= False,
    size_training_set= 1000, 
    len_training_set=5, 
    size_test_set=1000, 
    len_test_set= 5, 
    nb_states= [4,5,6], 
    iterations= 1,
    hypo_generator= random_model, 
    hypo_generator_args = dict(),
    learning_algorithm_type= 'BW', 
    learning_algorithm_epsilon= 0.01, 
    output_folder= 'results',
    result_file= 'result'
    ):
    os.makedirs(output_folder, exist_ok=True)
    f = open(output_folder+"/"+result_file+".txt",'a')
    f.close()
    for original_model in original_models:
        alphabet= original_model.observations();

        # Get training and test set
        if original_model.name not in datasets:
            training_set = generateSet(original_model, size_training_set, len_training_set)
            hypo_training_set= False
            test_set = generateSet(original_model, size_test_set, len_test_set)
        else:
            if splitdatasets== False:
                training_set= loadSet(datasets[original_model.name]['trainingset'])
                hypo_training_set = False
            else:
                training_set= loadSet(splitdatasets[original_model.name]['trainingset2'])
                hypo_training_set = loadSet(splitdatasets[original_model.name]['trainingset1'])
            
            test_set= loadSet(datasets[original_model.name]['testset'])
        

        model_type=model_types.get(type(original_model));

        # TODO create a case for each model type
        log_like_org= original_model.logLikelihood(test_set);

        # write to result file
        f = open(output_folder+"/"+result_file+".txt",'a')
        f.write("Model to learn: ")
        f.write(original_model.name+'\n')
        if splitdatasets== False:
            f.write("Training_set: "+str(sum(training_set[1]))+" sequences of "+str(len(training_set[0][1]))+" observations\n")
        else:
            f.write("Training_set: "+str(sum(training_set[1]))+"/"+str(sum(hypo_training_set[1]))+" sequences of "+str(len(training_set[0][1]))+" observations\n")
        f.write("Testing_set: "+str(sum(test_set[1]))+" sequences of "+str(len(test_set[0][1]))+" observations\n")
        f.write("logLikelihood of original model: "+ str(log_like_org)+"\n")
        f.write("Observation alphabet: "+ str(alphabet)+"\n")
        f.write("Learning algorithm "+ learning_algorithm_type+ ", epsilon: "+ str(learning_algorithm_epsilon)+'\n')
        f.write("Hypothesis generator: "+ hypo_generator.__name__+"\n")

        f.close();



        experiment(training_set, test_set, model_type, original_model.name, log_like_org, alphabet, nb_states, iterations, hypo_generator, hypo_training_set, hypo_generator_args, learning_algorithm_type, learning_algorithm_epsilon, output_folder, result_file)