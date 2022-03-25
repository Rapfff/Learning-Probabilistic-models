from examples.examples_models import modelMCGT_REBER, modelMCGT_random
from src.learning.BW_LTL import BW_LTL
from src.tools import generateSet
from src.models.MCGT import loadMCGT
m = modelMCGT_REBER()

training_set = generateSet(m,1000,7)
test_set     = generateSet(m,1000,7)
formula      = "b & X(t xor p) & GF e"

output_model = BW_LTL().learn(formula,training_set,alphabet=m.observations(),verbose=True,nb_states=7)

output_model.pprint()