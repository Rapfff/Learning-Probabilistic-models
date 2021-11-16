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


def run_experiment(
    original_models=set(), 
    datasets= dict(),
    splitdatasets= False,
    size_training_set= 1000, 
    len_training_set=5, 
    size_test_set=1000, 
    len_test_set= 5, 
    nb_states= {4,5,6}, 
    iterations= 1,
    hypo_generator= random_model, 
    hypo_generator_args = dict(),
    learning_algorithm_type= 'BW', 
    learning_algorithm_epsilon= 0.01, 
    output_folder= 'results',
    result_file= 'result'
    ):
    """
        Get the necessary information from the original models and runs experiments on them
        Input:
            original_models: Set of models to learn (MCGT)
            datasets: Two options
                Option 1: dictionary with path to the sets
                    datasets= {<Model_name>:  {'trainingset': <path to trainingset>, 'testset': <path to testset>}}
                    If you want to use separated training set you can have the dictionary 
                        splitdatasets= {<Model_name>:  {'trainingset1': <path to trainingset1>, 'trainingset2': <path to trainingset2>}}
                        # trainingset1: hypo_trainingset, trainingset2: learn_trainingset
                Option 2: Generate training and testing set
                    Generates a training set of size: size_training_set and length: len_training_set
                    Generates a test set of size: size_test_set and length: len_test_set
            nb_states: set of a number of states in the hypothesis model to try
            iterations: Number of times to run the experiment (This is useful when the hypothesis generator is random)
            hypo_generator: a function that generates a hypothesis model we input to the learning algorithm
            hypo_generator_args: extra arguments to put to the hypotesis generator
            learning_algorithm_type: the name of the learning algoritm used
            learning_algorithm_epsilon: the epsilon value for the learning algorithm
            output_folder: a path to the folder we want to store the results
            result_file: a file that we store our results in
    """

    # create the result file (in folder output_folder)
    os.makedirs(output_folder, exist_ok=True)
    f = open(output_folder+"/"+result_file+".txt",'a')
    f.close()

    # run experimet for each original model
    for original_model in original_models:
        # Get the alphabet
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
        
        # Get model type
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

        # Call the experiment function
        experiment(
            training_set=training_set, 
            test_set=test_set, 
            model_type=model_type, 
            model_name=original_model.name, 
            log_like_org=log_like_org, 
            alphabet=alphabet, 
            nb_states=nb_states, 
            iterations=iterations, 
            hypo_generator=hypo_generator,
            hypo_training_set=hypo_training_set, 
            hypo_generator_args=hypo_generator_args, 
            learning_algorithm_type=learning_algorithm_type, 
            learning_algorithm_epsilon=learning_algorithm_epsilon, 
            output_folder=output_folder, 
            result_file=result_file )