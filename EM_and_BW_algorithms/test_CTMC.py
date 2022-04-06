from examples.examples_models import modelCTMC_REBER, modelCTMC1, modelCTMC_random, modelMC_REBER
from src.learning.BW_CTMC import BW_CTMC
from src.tools import generateSet
from src.models.CTMC import *

m = modelMC_REBER()

training_set = generateSet(m,1000,7)
test_set     = generateSet(m,1000,7)

r1 = modelCTMC_random(3,m.observations())
r2 = modelCTMC_random(3,m.observations())

r1 = BW_CTMC(r).learn(training_set,verbose=True)
output_model.pprint_untimed()
output_model.pprint()

