from src.tools import generateSet

from src.learning.BW_MC import BW_MC
from examples.examples_models import modelMC_REBER, modelMC_random

print("\nMC")
m = modelMC_REBER()
s = generateSet(m,100,10)

m2 = modelMC_random(7,m.observations(),True)
print(m2.initial_state)
algo = BW_MC(m2)

m3 = algo.learn(s)

print(m3.initial_state)

print(m2.logLikelihood(s), m3.logLikelihood(s))
print(m.logLikelihood(s))

print("\nHMM")
from src.learning.BW_HMM import BW_HMM
from examples.examples_models import modelHMM4, modelHMM_random

m = modelHMM4()
s = generateSet(m,100,10)

m2 = modelHMM_random(5,m.observations(),True)
print(m2.initial_state)
algo = BW_HMM(m2)

m3 = algo.learn(s)

print(m3.initial_state)

print(m2.logLikelihood(s), m3.logLikelihood(s))
print(m.logLikelihood(s))

print("\ncoMC")
from src.learning.BW_coMC import BW_coMC
from examples.examples_models import modelCOMC1, modelCOMC_random

m = modelCOMC1()
s = generateSet(m,100,10)

m2 = modelCOMC_random(2,True)
print(m2.initial_state)
algo = BW_coMC(m2)

m3 = algo.learn(s)

print(m3.initial_state)

print(m2.logLikelihood(s), m3.logLikelihood(s))
print(m.logLikelihood(s))

print("\ncoHMM")
from src.learning.BW_coHMM import BW_coHMM
from examples.examples_models import modelCOHMM1, modelCOHMM_random

m = modelCOHMM1()
s = generateSet(m,100,10)

m2 = modelCOHMM_random(2,True)
print(m2.initial_state)
algo = BW_coHMM(m2)

m3 = algo.learn(s)

print(m3.initial_state)

print(m2.logLikelihood(s), m3.logLikelihood(s))
print(m.logLikelihood(s))
