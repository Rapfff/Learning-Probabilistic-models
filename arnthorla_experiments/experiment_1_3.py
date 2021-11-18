import os, sys, copy, math, inspect, datetime
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import *
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_MCGT import *
from EM_and_BW_algorithms.src.models.MCGT import *
from EM_and_BW_algorithms.src.tools import *

from statistics import fmean, stdev
from time import process_time_ns


# UTILITY DEFINITIONS
MIN_COLUMN_WIDTH = 25                                           # Minimum width of result columns
RESULTS_FOLDER_SUFFIX = "_results"                              # Results folder name ends with suffix
num_res_results = 3                                             # Number of columns in res (results)
num_summary_results = 4                                         # Number of colums in summary (results)
base_alphabet = "abcdefghijklmnopqrstuvwxyz"                    # Base alphabet, will be sliced to needs.
res_headers = ["#States", "#Sequences", "Learned - Original" ]  # Header of logged results
summary_headers = ["#States", "#Sequences", "Mean(Learned - Original)", "Std.Dev(Learned - Original)"]


# PARAMETERS OF EXPERIMENT
num_states = [2,3] #[2,3,4]
num_sequences = [10,100] #[10, 100, 1000, 10000]
num_tests = 1 #30 #100                             # How many repeats of equivalent tests, for statistics
model_generator = modelMCGT_random              # Function that generates model
name_of_model = "Random MCGT"
learn_algo = Estimation_algorithm_MCGT          # Learning algorithm to be used
name_of_learning_algo = "Estimation Algorithm MCGT"
res = []                                        # Datastructure to hold results
running_time = 0                                # Running time in minutes 


## UTILITY FUNCTIONS
#def fullResultsFolderName():
#    current_stack_index = len(inspect.stack())-1    # index of current running script (not just this file)
#    filename = inspect.stack()[current_stack_index].filename    # full filename (with path)
#    filename = os.path.splitext( filename )[0]                  # remove file extension
#    filename = filename + RESULTS_FOLDER_SUFFIX                 # add "_results" suffix
#    return filename
#
#def createResultsFolderIfNotExists():
#    results_folder = fullResultsFolderName()                    # Full folder name of results folder
#    if not os.path.isdir( results_folder ):
#        os.mkdir( results_folder )
#
#def createFullTargetFilename():
#    createResultsFolderIfNotExists()
#    now = datetime.datetime.now()
#    full_folder_path = fullResultsFolderName()
#    filename = ("result_" +str(now.year)+"_"+str(now.month)+"_"+str(now.day)+
#            "_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+".txt")
#    return os.path.join( full_folder_path, filename )
#
#
#
#def summary_stats( res ):
#    summary_total = []
#    for i in range(len(num_states)):
#        for j in range(len(num_sequences)):
#            results, summary_row = [], []
#            for k in range( num_tests ):
#                results.append(res[i][j][k][2])
#            if len(results) > 1:
#                summary_row = [ res[i][j][0][0],res[i][j][0][1], fmean(results), stdev(results) ]
#            else:
#                summary_row = [ res[i][j][0][0],res[i][j][0][1], results[0], 0.0 ]
#            summary_total.append(summary_row)
#    return summary_total
#    
#
#def logResults():
#    #TODO: Don't Repeat Yourself -> Ripe for refactoring
#    # SUMMARY STATISTICS
#    summary = summary_stats( res )
#
#    # EXPERIMENT PARAMETERS
#    experiment_parameters = [["Number of states: ", str(num_states)],
#            ["Number of sequences: ", str(num_sequences)],
#            ["Number of tests: ", str(num_tests)],
#            ["Model type: ", name_of_model],
#            ["Learning algorithm: ", name_of_learning_algo],
#            ["Running time (minutes): ", round((end_time-start_time)/(60*1000000000),2) ]]
#
#    # Column format
#    column_width_res = max(MIN_COLUMN_WIDTH, len(max(res_headers,key=len)))        
#    column_width_summary = max(MIN_COLUMN_WIDTH, len(max(summary_headers,key=len)))
#    column_width = max( column_width_res, column_width_summary )
#
#    # PRINT TO STDOUT
#    # Parameters
#    for pair in experiment_parameters:
#        print( pair[0], pair[1] )
#    print()
#
#    # Results
#    for col in range(num_res_results):
#        print( f'|{res_headers[col]:<{column_width_res}}', end='' )
#    print( "|" )
#    for i in range(len(num_states)):
#        for j in range(len(num_sequences)):
#            for k in range(num_tests):
#                for col in range(num_res_results):
#                    print( f'|{res[i][j][k][col]:<{column_width_res}}', end='' )
#                print( "|" )
#    print()
#    # Summary statistics
#    for col in range(num_summary_results):
#        print( f'|{summary_headers[col]:<{column_width_summary}}', end='' )
#    print( "|" )
#    for i in range(len(num_states)*len(num_sequences)):
#        for col in range(num_summary_results):
#            print( f'|{summary[i][col]:<{column_width_summary}}', end='' )
#        print( "|" )
#
#    # PRINT TO FILE
#    filename = createFullTargetFilename()
#    with open( filename, "w" ) as f:
#    # Parameters
#        for pair in experiment_parameters:
#            print( pair[0], pair[1], file=f )
#        print("",file=f )
#    # Results
#        for col in range(num_res_results):
#            print( f'|{res_headers[col]:<{column_width_res}}', end='', file=f )
#        print( "|", file=f )
#        for i in range(len(num_states)):
#            for j in range(len(num_sequences)):
#                for k in range(num_tests):
#                    for col in range(num_res_results):
#                        print( f'|{res[i][j][k][col]:<{column_width_res}}', end='', file=f )
#                    print( "|", file=f )
#        print("",file=f)
#    # Summary statistics
#        for col in range(num_summary_results):
#            print( f'|{summary_headers[col]:<{column_width_summary}}', end='', file=f )
#        print( "|", file=f )
#        for i in range(len(num_states)*len(num_sequences)):
#            for col in range(num_summary_results):
#                print( f'|{summary[i][col]:<{column_width_summary}}', end='', file=f )
#            print( "|", file=f )

# PREP FOR EXPERIMENT
res = ([[[[None for i in range(num_res_results)] 
    for j in range(num_tests)] 
    for k in range(len(num_sequences))] 
    for m in range(len(num_states))])

# EXPERIMENT
start_time = process_time_ns()
for i in range( len(num_states) ): # Number of States
    alphabet = base_alphabet[:num_states[i]]  # slice alphabet, length equal to number of states
    len_sequence = int( math.sqrt(num_states[i]) ) * num_states[i] # formula suggested by Raphaël 
    for j in range( len(num_sequences) ): # Number of Sequnces 
        for k in range( num_tests ): # Tests with same parameters
            isResultNegative = False
            isSecondTrainingDifferent = False
            print( "States:[",i+1,"/",len(num_states),
                    "] Sequences:[",j+1,"/",len(num_sequences),
                    "] Tests:[",k+1,"/",num_tests,"]")
            res[i][j][k][0] = num_states[i]
            res[i][j][k][1] = num_sequences[j]
            original_model = model_generator( num_states[i], alphabet )
            training_set = generateSet( original_model, num_sequences[j], len_sequence )


#######################################################################################
#                                  PROBLEM CODE:
#######################################################################################
# CORRECT RESULT: 
            ###########################################################################
            # Note: Computing the log likelihood of the original model here           #
            #       gives a correct result.                                           #
            ###########################################################################
            # logLiklihood of original model:
            #original_model_logLikelihood = original_model.logLikelihood( training_set )
            
            # First training:
            algo = learn_algo( original_model, original_model.observations() )
            first_trained_model = algo.learn( training_set ) # note learn3() for debugging
            first_trained_model_logLikelihood = first_trained_model.logLikelihood( training_set )

# INCORRECT RESULT:
            ###########################################################################
            # Note: Computing the log likelihood of the original model here           #
            #       gives an incorrect result.                                        #
            # Description of error:                                                   #
            #       trained_logLikelihoo - original_logLiklihood < 0.0                #
            #       Also, if you train the original model again and get               #
            #       a new trained model, its logLiklihood will be higher              #
            #       and probably the correct one, and all subsequent training         #
            #       will result in that result, and only the first trained model      #
            #       will be different (incorrect).                                    #
            ###########################################################################
            # logLiklihood of original model:
            original_model_logLikelihood = original_model.logLikelihood( training_set )
#######################################################################################


            # Difference between original and trained logLikelihood:
            res[i][j][k][2] = first_trained_model_logLikelihood - original_model_logLikelihood
            if res[i][j][k][2] < 0.0:
                isResultNegative = True
                print( "isResultNegative == True" )
            
            # Second training:
            algo = learn_algo( original_model, original_model.observations() )
            second_trained_model = algo.learn( training_set )
            second_trained_model_logLikelihood = second_trained_model.logLikelihood(training_set)
            if first_trained_model_logLikelihood != second_trained_model_logLikelihood:
                isSecondTrainingDifferent = True
                print( "isSecondTrainingDifferent == True" )

            if isResultNegative or isSecondTrainingDifferent:
                print( "\n############################## BUG ##############################" )
                print("States: ", num_states[i], " Number of sequeces: ", num_sequences[j] )
                if isResultNegative:
                    print( "Problem: Result is negative!" )
                if isSecondTrainingDifferent:
                    print( "Problem: First training gives different logLikelihood, than Second training." )
                    print( "\tIf this happens, all subsequent trainings give the same as second, but different than first." )
                print( "Stats: " )
                print( "Difference trained-origial logLikelihood:", res[i][j][k][2] )
                print( "original_model_logLikelihood: ", original_model_logLikelihood )
                print( "old_trained_logLikelihood: ", first_trained_model_logLikelihood )
                print( "new_trained_logLikelihood: ", second_trained_model_logLikelihood )
                print( "Models:" )
                print( "Original Model:" )
                original_model.pprint()
                print( "First Trained Model:" )
                first_trained_model.pprint()
                print( "Second Trained Model:" )
                second_trained_model.pprint()
                print(   "#################################################################\n" )


end_time = process_time_ns()
#logResults()
