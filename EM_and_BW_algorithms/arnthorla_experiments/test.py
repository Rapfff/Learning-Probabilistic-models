import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples.examples_models import *
from src.learning.Estimation_algorithm_HMM import *
#from ..src.models.HMM import *

#model = modelHMM1()
#model = modelHMM1_equiprobable()
#model = modelHMM2()
#model = modelHMM2_random()
#model = modelHMM2_equiprobable()
#model = modelHMM3()
#model = modelHMM4()
model = modelHMM_random(10, "$abcdxy")
model.pprint()
observations = model.observations()
algo = Estimation_algorithms_HMM(model,observations)

seq = "$bd"
p1 = algo.problem1(seq)
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

