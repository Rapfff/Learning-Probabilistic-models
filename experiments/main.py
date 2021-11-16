import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from runner import run_experiment
from examples.examples_models import *
from hypo_search import *


nb_states={2, 3, 4, 5, 6, 7, 8, 9}
nb_iterations= 100;
experiment_folder = "results/experiments_5"

original_models2= {modelMCGT_game(), modelMCGT_REBER(), modelMCGT5(), modelMCGT6(), modelMCGT7(), modelMCGT8(), modelMCGT9(), modelMCGT10(), modelMCGT11(), modelMCGT12()}

datasets= {'MCGT_games': {'trainingset': 'datasets/MCGT_gamestrainingset.txt', 'testset': 'datasets/MCGT_gamestestset.txt'}, 
    'MCGT_REBER': {'trainingset': 'datasets/MCGT_REBERtrainingset.txt', 'testset': 'datasets/MCGT_REBERtestset.txt'}, 
    'MCGT5': {'trainingset': 'datasets/MCGT5trainingset.txt', 'testset': 'datasets/MCGT5testset.txt'},
    'MCGT6': {'trainingset': 'datasets/MCGT6trainingset.txt', 'testset': 'datasets/MCGT6testset.txt'},
    'MCGT7': {'trainingset': 'datasets/MCGT7trainingset.txt', 'testset': 'datasets/MCGT7testset.txt'},
    'MCGT8': {'trainingset': 'datasets/MCGT8trainingset.txt', 'testset': 'datasets/MCGT8testset.txt'},
    'MCGT9': {'trainingset': 'datasets/MCGT9trainingset.txt', 'testset': 'datasets/MCGT9testset.txt'},
    'MCGT10': {'trainingset': 'datasets/MCGT10trainingset.txt', 'testset': 'datasets/MCGT10testset.txt'},
    'MCGT11': {'trainingset': 'datasets/MCGT11trainingset.txt', 'testset': 'datasets/MCGT11testset.txt'},
    'MCGT12': {'trainingset': 'datasets/MCGT12trainingset.txt', 'testset': 'datasets/MCGT12testset.txt'},
    }
datasets_split10= {
    'MCGT_games': {'trainingset1': 'datasets/split_10/MCGT_gamestrainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT_gamestrainingset2.txt'}, 
    'MCGT_REBER': {'trainingset1': 'datasets/split_10/MCGT_REBERtrainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT_REBERtrainingset2.txt'}, 
    'MCGT5': {'trainingset1': 'datasets/split_10/MCGT5trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT5trainingset2.txt'}, 
    'MCGT6': {'trainingset1': 'datasets/split_10/MCGT6trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT6trainingset2.txt'}, 
    'MCGT7': {'trainingset1': 'datasets/split_10/MCGT7trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT7trainingset2.txt'}, 
    'MCGT8': {'trainingset1': 'datasets/split_10/MCGT8trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT8trainingset2.txt'}, 
    'MCGT9': {'trainingset1': 'datasets/split_10/MCGT9trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT9trainingset2.txt'}, 
    'MCGT10': {'trainingset1': 'datasets/split_10/MCGT10trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT10trainingset2.txt'}, 
    'MCGT11': {'trainingset1': 'datasets/split_10/MCGT11trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT11trainingset2.txt'}, 
    'MCGT12': {'trainingset1': 'datasets/split_10/MCGT12trainingset1.txt', 'trainingset2': 'datasets/split_10/MCGT12trainingset2.txt'}}
datasets_split20= {
    'MCGT_games': {'trainingset1': 'datasets/split_20/MCGT_gamestrainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT_gamestrainingset2.txt'}, 
    'MCGT_REBER': {'trainingset1': 'datasets/split_20/MCGT_REBERtrainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT_REBERtrainingset2.txt'}, 
    'MCGT5': {'trainingset1': 'datasets/split_20/MCGT5trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT5trainingset2.txt'}, 
    'MCGT6': {'trainingset1': 'datasets/split_20/MCGT6trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT6trainingset2.txt'}, 
    'MCGT7': {'trainingset1': 'datasets/split_20/MCGT7trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT7trainingset2.txt'}, 
    'MCGT8': {'trainingset1': 'datasets/split_20/MCGT8trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT8trainingset2.txt'}, 
    'MCGT9': {'trainingset1': 'datasets/split_20/MCGT9trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT9trainingset2.txt'}, 
    'MCGT10': {'trainingset1': 'datasets/split_20/MCGT10trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT10trainingset2.txt'}, 
    'MCGT11': {'trainingset1': 'datasets/split_20/MCGT11trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT11trainingset2.txt'}, 
    'MCGT12': {'trainingset1': 'datasets/split_20/MCGT12trainingset1.txt', 'trainingset2': 'datasets/split_20/MCGT12trainingset2.txt'}}
datasets_split50= {
    'MCGT_games': {'trainingset1': 'datasets/split_50/MCGT_gamestrainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT_gamestrainingset2.txt'}, 
    'MCGT_REBER': {'trainingset1': 'datasets/split_50/MCGT_REBERtrainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT_REBERtrainingset2.txt'}, 
    'MCGT5': {'trainingset1': 'datasets/split_50/MCGT5trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT5trainingset2.txt'}, 
    'MCGT6': {'trainingset1': 'datasets/split_50/MCGT6trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT6trainingset2.txt'}, 
    'MCGT7': {'trainingset1': 'datasets/split_50/MCGT7trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT7trainingset2.txt'}, 
    'MCGT8': {'trainingset1': 'datasets/split_50/MCGT8trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT8trainingset2.txt'}, 
    'MCGT9': {'trainingset1': 'datasets/split_50/MCGT9trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT9trainingset2.txt'}, 
    'MCGT10': {'trainingset1': 'datasets/split_50/MCGT10trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT10trainingset2.txt'}, 
    'MCGT11': {'trainingset1': 'datasets/split_50/MCGT11trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT11trainingset2.txt'}, 
    'MCGT12': {'trainingset1': 'datasets/split_50/MCGT12trainingset1.txt', 'trainingset2': 'datasets/split_50/MCGT12trainingset2.txt'}}

if __name__ == "__main__":
    if sys.argv[1]=='1': # Split with random search
        run_experiment(
            original_models=original_models2, 
            datasets= datasets, splitdatasets=datasets_split50, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=random_search, 
            output_folder=experiment_folder+'/random_search_split', 
            result_file='random_split_50')

        run_experiment(
            original_models=original_models2, 
            datasets= datasets, splitdatasets=datasets_split20, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=random_search, 
            output_folder=experiment_folder+'/random_search_split', 
            result_file='random_split_20')

        run_experiment(
            original_models=original_models2, 
            datasets= datasets, splitdatasets=datasets_split10, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=random_search, 
            output_folder=experiment_folder+'/random_search_split', 
            result_file='random_split_10')
    elif sys.argv[1]=='2': # Split with smart random search
        run_experiment(
            original_models=original_models2, 
            datasets= datasets, splitdatasets=datasets_split50, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            output_folder=experiment_folder+'/smart_random_search_split', 
            result_file='smart_random_split_50')

        run_experiment(
            original_models=original_models2, 
            datasets= datasets, splitdatasets=datasets_split20, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            output_folder=experiment_folder+'/smart_random_search_split', 
            result_file='smart_random_split_20')

        run_experiment(
            original_models=original_models2, 
            datasets= datasets, splitdatasets=datasets_split10, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            output_folder=experiment_folder+'/smart_random_search_split', 
            result_file='smart_random_split_10')
    elif sys.argv[1]=='3': # smart random search w. dynamic lambda
        run_experiment(
            original_models=original_models2, 
            datasets= datasets, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            hypo_generator_args={'modify': True},
            output_folder=experiment_folder+'/smart_random_search_dy', 
            result_file='smart_random_dy')
    elif sys.argv[1]=='4': # smart random search w. different lambda values (noise)
        run_experiment(
            original_models={modelMCGT6()}, 
            datasets= datasets, 
            nb_states={9},iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            hypo_generator_args={'lambda_': 0.25},
            output_folder=experiment_folder+'/smart_random_search_25', 
            result_file='smart_random_search_25')
        run_experiment(
            original_models= {modelMCGT_game(), modelMCGT_REBER(), modelMCGT5(), modelMCGT7(), modelMCGT8(), modelMCGT9(), modelMCGT12()}, 
            datasets= datasets, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            hypo_generator_args={'lambda_': 0.25},
            output_folder=experiment_folder+'/smart_random_search_25', 
            result_file='smart_random_search_25')
        
        run_experiment(
            original_models= original_models2, 
            datasets= datasets, 
            nb_states=nb_states,iterations=nb_iterations, 
            hypo_generator=smart_random_search, 
            hypo_generator_args={'lambda_': 0.9},
            output_folder=experiment_folder+'/smart_random_search_90', 
            result_file='smart_random_search_90')
        
        
        


    else:
        print('Run: main.py <1: split with random search 2: split with smart random search 3: smart random search w. dynamic lambda>')
