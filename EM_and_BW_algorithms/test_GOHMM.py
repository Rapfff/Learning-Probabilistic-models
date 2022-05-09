from examples.examples_models import modelGOHMM_random, modelGOHMM2, modelGOHMM_nox
from src.tools import generateSet
from src.learning.BW_GOHMM import BW_GOHMM
m1 = modelGOHMM2()
m2 = modelGOHMM_nox(min_mu=0.0,max_mu=5.0,min_std=0.1,max_std=5.0)
tr = generateSet(m1,1000,30)
algo = BW_GOHMM(m2)
m3 = algo.learn(tr)
m3.pprint()