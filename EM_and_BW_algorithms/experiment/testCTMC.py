import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples.examples_models import modelCTMC_REBER, modelCTMC1, modelCTMC_random
from src.tools import generateSet
from src.learning.Estimation_algorithm_CTMC import Estimation_algorithm_CTMC

m = modelCTMC1()

tr = generateSet(m,100,8)
ts = generateSet(m,100,8)
#ts_without_time = [[],ts[1]]
#for i in ts[0]:
#	ts_without_time[0].append([i[j] for j in range(1,len(i),2)])

i = modelCTMC_random(3,m.observations())

algo = Estimation_algorithm_CTMC(i)

n = algo.learn(tr)

print(m.logLikelihood_with_time(ts))
print(i.logLikelihood_with_time(ts))
print(n.logLikelihood_with_time(ts))
print()
print("We are learning")
m.pprint()
print("We get")
n.pprint()