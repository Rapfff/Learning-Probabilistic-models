from examples.examples_models import modelCTMC_REBER, modelCTMC1, modelCTMC_random, modelMC_REBER
from src.learning.BW_CTMC import BW_CTMC
from src.learning.BW_CTMC_Composition import BW_CTMC_Composition
from src.tools import generateSet
from src.models.CTMC import *

m = modelMC_REBER()

training_set = generateSet(m,1000,7)
test_set     = generateSet(m,1000,7)

r1 = modelCTMC_random(4,m.observations(),self_loop=False)
r2 = modelCTMC_random(4,m.observations(),self_loop=False)
output_model = BW_CTMC_Composition(r1,r2).learn(training_set,verbose=True)
print(output_model.logLikelihood(test_set))

r3 = modelCTMC_random(16,m.observations(),self_loop=False)
output_model = BW_CTMC(r3).learn(training_set,verbose=True)
print(output_model.logLikelihood(test_set))

