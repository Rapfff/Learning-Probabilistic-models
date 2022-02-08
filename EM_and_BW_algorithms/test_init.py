from examples.examples_models import modelMCGT_REBER, modelMCGT_random
from tools import generateSet
from src.learning.BW_MCGT import BW_MCGT

m = modelMCGT_REBER()
s = generateSet(m,100,10)

m2 = modelMCGT_random(7,m.observations(),True)
m2.pprint()
algo = BW_MCGT(m2)

m3 = algo.learn(s)

m3.pprint()

print(m2.logLikelihood(s), m3.logLikelihood(s))
print(m.logLikelihood(s))