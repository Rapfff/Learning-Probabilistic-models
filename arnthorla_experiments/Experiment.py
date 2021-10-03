import os, sys, copy, math, inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import *
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_MCGT import *
from EM_and_BW_algorithms.src.models.MCGT import *
from EM_and_BW_algorithms.src.tools import *

from statistics import fmean, stdev

class Experiment_MCGT:
    def __init__( self, num_states, num_tests, model_generators, learn_algo ):
        self.MIN_COLUMN_WIDTH = 20                          # Minimum width of result columns
        self.RESULTS_FOLDER_SUFFIX = "_results"             # Results folder name ends with suffix
        self.base_alphabet = "abcdefghijklmnopqrstuvwxyz"   # Base alphabet, will be sliced to needs.  
        self.num_states = num_states                        # List of integers 
        self.num_tests = num_tests                          # How many repeats of equivalent test (for statistics)
        self.model_generators = model_generators            # List of model generator functions
        self.learn_algo = learn_algo                        # Learning algorithim, ie Estimation Algorithm
        self.res_headers = []                               # To have it defined in parent class, will hold header of results
        self.res = []                                       # To have it defined in parent calss, will hold results in child
        self.results_folder = self._fullResultsFolderName() # Full folder name of results folder

    def _getDimensions(self, li, dimensions=None ):
        if dimensions is None:
            dimensions = []
        if type(li) != list or not li:
            return dimensions
        else:
            dimensions.append(len(li))
            li = li[0]
            return self._getDimensions( li, dimensions )

    def _logList(self, li, dimensions, logFunc ):
        if len(dimensions) < 1:
            return None
        if len(dimensions) == 1:
            logFunc( li )
        else:
            length = dimensions.pop(0)
            for i in range(length):
                self._logList( li[i], copy.copy(dimensions), logFunc )

    def _fullResultsFolderName(self):
        current_stack_index = len(inspect.stack())-1    # index of current running script (not just this file)
        filename = inspect.stack()[current_stack_index].filename    # full filename (with path)
        filename = os.path.splitext( filename )[0]                  # remove file extension
        filename = filename + self.RESULTS_FOLDER_SUFFIX            # add "_results" suffix
        return filename

    def _createResultsFolderIfNotExists(self):
        if not os.path.isdir( self.results_folder ):
            os.mkdir( self.results_folder )

    def _logResults(self):
        column_width = max(self.MIN_COLUMN_WIDTH, len(max(self.res_headers,key=len)))        
        dimemsions = self._getDimensions( self.res  )

        # stdout
        def f( li, w=column_width ):
            length = len(li)
            for i in range(length):
                print( f'|{li[i]:<{w}}', end='' )
            print( '|' )
        f( self.res_headers )
        self._logList( self.res, dimensions, f )

        # ensure that result folder exists
        self._createResultsFolderIfNotExists()

        #TODO: Next: Writing to files



class Ex_Generator_Model_Trained_On_Own_Output( Experiment_MCGT ):
    def __init__( self, num_states, num_tests, model_generators, learn_algo, num_sequences ):
        Experiment_MCGT.__init__( self, num_states, num_tests, model_generators, learn_algo )
        self.num_sequences = num_sequences                      # List of integers
        self.res_headers = ["#States", "#Sequences", "#Tests" ]
        self.res = ([[[[None for i in range(3)] 
            for j in range(self.num_tests)] 
            for k in range(len(self.num_states))] 
            for m in range(len(num_states))])


    def run(self):
        for i in range( len(self.num_states) ): # Number of States
            alphabet = self.base_alphabet[:self.num_states[i]]  # slice alphabet, length equal to number of states
            len_sequence = int( math.sqrt(self.num_states[i]) ) * self.num_states[i] # formula suggested by Raphaël 
            for j in range( len(self.num_sequences) ): # Number of Sequnces 
                for k in range( self.num_tests ): # Tests with same parameters
                    print( "[",i+1,"/",len(self.num_states),"][",j+1,"/",len(self.num_sequences),"][",k+1,"/",self.num_tests,"]")
                    self.res[i][j][k][0] = self.num_states[i]
                    self.res[i][j][k][1] = self.num_sequences[j]
                    original_model = self.model_generators[0]( self.num_states[i], alphabet )
                    training_model = copy.deepcopy( original_model )                    
                    training_set = generateSet( original_model, self.num_sequences[j], len_sequence )
                    algo = self.learn_algo( training_model, training_model.observations() )
                    training_model = algo.learn( training_set )
                    self.res[i][j][k][2] = training_model.logLikelihood( training_set ) - original_model.logLikelihood( training_set )
        super()._logResults()

if __name__ == "__main__":
    exp1 = Ex_Generator_Model_Trained_On_Own_Output([2],2,[modelMCGT_random],Estimation_algorithm_MCGT,[2])
    exp1.run()


