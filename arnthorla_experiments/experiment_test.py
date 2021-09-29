import os, sys, copy
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples.examples_models import *
from src.learning.Estimation_algorithm_HMM import *
from src.learning.Estimation_algorithm_MCGT import *
from src.models.MCGT import *
from src.tools import *

#original_model = modelMCGT1()
original_model = modelMCGT4()
original_model.pprint()

training_set = generateSet( original_model,2,100 )
original_score = original_model.logLikelihood( training_set )

#print( "\n", training_set, "\n" )
training_model = copy.deepcopy(original_model)

algo = Estimation_algorithm_MCGT( training_model, training_model.observations() )
training_model = algo.learn( training_set )

training_model.pprint()
trained_score = training_model.logLikelihood( training_set )

print( "original score: ", original_score )
print( "trainded score: ", trained_score )


#print( training_set )
#observations = model.observations()
#algo = Estimation_algorithm_MCGT(model,observations)


"""
seq = "$bd"
p1 = algo.problem1( seq)
print( p1 )
p2 = algo.problem2(seq)
print(p2)

print()
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
print( model.run(10) )
"""
