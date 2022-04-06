from examples.examples_models import modelCTMC_REBER, modelCTMC1, modelCTMC_random
from src.learning.BW_CTMC import BW_CTMC
from src.tools import generateSet
from src.models.CTMC import loadCTMC

m = modelCTMC_REBER()
m.pprint()

training_set = generateSet(m,1000,7,timed=False)
test_set     = generateSet(m,1000,7,timed=False)

r = modelCTMC_random(7,m.observations())

output_model = BW_CTMC(r).learn(training_set,verbose=True)
output_model.pprint_untimed()
output_model.pprint()

#print(output_model.logLikelihood(test_set))

