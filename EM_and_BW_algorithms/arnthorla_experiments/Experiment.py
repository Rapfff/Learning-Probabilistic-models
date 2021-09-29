import os, sys, copy, math
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples.examples_models import *
from src.learning.Estimation_algorithm_MCGT import *
from src.models.MCGT import *
from src.tools import *

from statistics import fmean, stdev

class Experiment_MCGT:
    def __init__( self, num_states, num_tests, model_generators, learn_algo ):
        self.base_alphabet = "abcdefghijklmnopqrstuvwxyz"   # Base alphabet, will be sliced to needs.  
        self.num_states = num_states                        # List of integers 
        self.num_tests = num_tests                          # How many repeats of equivalent test (for statistics)
        self.model_generators = model_generators            # List of model generator functionsj
        self.learn_algo = learn_algo                        # Learning algorithim, ie Estimation Algorithm

class Ex_Generator_Model_Trained_On_Own_Output( Experiment_MCGT ):
    def __init__( self, num_states, num_tests, num_sequences ):
        Experiment_MCGT.__init__( self, num_states, num_tests )
        self.num_sequences = num_sequences                  # List of integers
        self.res = [[[None for i in range(3)] for j in range(self.num_tests)] for k in range(len(self.num_states))]

    def run(self):
        for i in range( len(self.num_states) ): # States
            alphabet = self.base_alphabet[:self.num_states[i]]  # slice alphabet, length equal to number of states
            len_sequence = int( math.sqrt(self.num_states[i]) ) * self.num_states[i] # formula suggested by Raphaël 
            #TODO loop for num_sequences training sets
            for j in range( self.num_tests ): # Tests
                print( "[", i+1, "/",len(self.num_states),"][",j+1,"/",num_tests, "]")
                self.res[i][j][0] = self.num_states[i]
                original_model = modelMCGT_equiprobable( self.num_states[i], alphabet )
                #TODO change num_sequences ... is now a list of ints
                training_set = generateSet( original_model, num_sequences, len_per_training_sequence ) 
                self.res[i][j][1] = original_model.logLikelihood( training_set )
                training_model = copy.deepcopy( original_model )                    
                algo = self.learn_algo( training_model, training_model.observations() )
                training_model = algo.learn( training_set )
                self.res[i][j][2] = training_model.logLikelihood( training_set )


#base_alphabet = "abcdefghijklmnopqrstuvwxyz" # Base alphabet, will be sliced to needs.
#
## Get base filename and construct filename to log results
#base_filename = os.path.splitext( os.path.basename( __file__ ) )[0]
#base_logfile_name = base_filename + '_results'
#
#
## Definitions
#size_alphabet = 5
#num_training_sequences = 10
#len_per_training_sequence = 5
#num_states_to_test = [3]
#num_models_per_experiment = 3 
#
## Experiement
#alphabet = base_alphabet[:size_alphabet]
#
#res = [[[None for i in range(3)] for j in range(num_models_per_experiment)] for k in range(len(num_states_to_test))]
#
#for i in range( len(num_states_to_test) ):
#    for j in range( num_models_per_experiment ):
#        print( "[", i+1, "/",len(num_states_to_test),"][",j+1,"/",num_models_per_experiment, "]")
#        res[i][j][0] = num_states_to_test[i]
#        #original_model = modelMCGT_random( num_states_to_test[i], alphabet )
#        original_model = modelMCGT_equiprobable( num_states_to_test[i], alphabet )
#        training_set = generateSet( original_model, num_training_sequences, len_per_training_sequence ) 
#        res[i][j][1] = original_model.logLikelihood( training_set )
#        training_model = copy.deepcopy( original_model )                    
#        algo = Estimation_algorithm_MCGT( training_model, training_model.observations() )
#        training_model = algo.learn( training_set )
#        res[i][j][2] = training_model.logLikelihood( training_set )
#        #res[i][j][3] = res[i][j][2] - res[i][j][1]
#
#model_name = original_model.__str__()
#model_name = model_name.split("_")
#print("original_model: ", "_".join(model_name[:-1]) )
#
## Summary statistics
#summary = [[None for i in range(2)] for j in range(len(num_states_to_test))]
#for i in range( len(num_states_to_test) ):
#    results = []
#    for j in range( num_models_per_experiment ):
#       results.append( res[i][j][2]-res[i][j][1] )
#    summary[i][0] = fmean( results )
#    summary[i][1] = stdev( results )
#
## Print Results
#print()
#for model in res:
#    for test in model: 
#        print( "#states: ", test[0], "\toriginal: ", test[1], "\ttrained: ", test[2], "\timprovement: ", test[2]-test[1] )
#        #print(  test[0], "\t", test[1], "\t ", test[2], "\t",  test[3] )
#
## Print Summary statistics
#print()
#for i in range( len( num_states_to_test ) ):
#    print( "#states: ", num_states_to_test[i], "\tMean impovement: ", summary[i][0], "\tStdev of improvement: ", summary[i][1] )
#
## Log results to file
#with open( base_logfile_name + ".txt", "a" ) as f:
#    print( "Size of observation alphabet: ", size_alphabet, file=f )
#    print( "Number of training sequences: ", num_training_sequences, file=f )
#    print( "Length per training sequence: ", len_per_training_sequence, file=f )
#    print( "Number of states to be tested: ", num_states_to_test, file=f )
#    print( "Number of models that are tested in each experiment: ", num_models_per_experiment, file=f )
#    print( "Observation alphabet: ", alphabet, file=f )
#    print(file=f)
#    for model in res:
#        for test in model: 
#            print( "#states: ", test[0], "\toriginal: ", test[1], "\ttrained: ", test[2], "\timprovement: ", test[2]-test[1], file=f )
#    # Print Summary statistics
#    print(file=f)
#    for i in range( len( num_states_to_test ) ):
#        print( "#states: ", num_states_to_test[i], "\tMean impovement: ", summary[i][0], "\tStdev of improvement: ", summary[i][1], file=f )
#
#
#
#
#
