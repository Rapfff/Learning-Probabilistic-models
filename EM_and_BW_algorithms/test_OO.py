from examples.examples_models import modelMCGT_REBER, modelMCGT_random,
									 modelHMM_random, modelHMM4, modelCOMC1,
									 modelCOHMM1, modelCOHMM_random
from src.tools import generateSet
from src.models.MCGT import loadMCGT
from src.models.HMM import loadHMM
from src.models.coMC import loadcoMC
from src.models.coHMM import loadcoHMM
from src.learning.BW_HMM import BW_HMM
from src.learning.BW_MCGT import BW_MCGT
from src.learning.BW_coHMM import BW_coHMM
from os import remove
"""
m = modelMCGT_REBER()

m.pprint()
print(m.observations())
print(m.tau(0,1,'B'))
print(m.tau(0,2,'B'))
print(m.tau(0,1,'V'))

s = generateSet(m,10,5)
print(s)

print(m.logLikelihood(s))

m2 = modelMCGT_random(7,m.observations())
algo = BW_MCGT(m2)
m3 = algo.learn(s)
m3.pprint()

###########################################################

m = modelHMM4()

m.pprint()
print(m.observations())
print(m.tau(0,2,'y'))
print(m.tau(0,1,'y'))
print(m.tau(0,1,'V'))

s = generateSet(m,10,5)
print(s)

print(m.logLikelihood(s))

m2 = modelMCGT_random(5,m.observations())
algo = BW_MCGT(m2)
m3 = algo.learn(s)
m3.pprint()

remove("output_model.txt")

###########################################################

m = modelCOMC1()

m.pprint()
print(m.tau(0,1,1.0))
print(m.tau(0,0,1.0))
print(m.tau(0,2,0.0))

s = generateSet(m,10,5)
print(s)

print(m.logLikelihood(s))

m.save("test_save.txt")

m2 = loadcoMC("test_save.txt")
m2.pprint()
###########################################################
"""
m = modelCOHMM1()

m.pprint()
print(m.tau(0,1,1.0))
print(m.tau(0,0,1.0))
print(m.tau(0,2,0.0))

s = generateSet(m,10,5)
print(s)

print(m.logLikelihood(s))

m2 = modelCOHMM_random(2,min_mu=0.0,max_mu=1.0,min_std=0.0,max_std=2.0):
algo = BW_coHMM(m2)
m3 = algo.learn(s)
m3.pprint()
remove("output_model.txt")
