from examples.examples_models import modelMCGT_REBER
from src.tools import generateSet
m = modelMCGT_REBER()
m.pprint()
print(m.observations())
print(m.tau(0,1,'B'))
print(m.tau(0,2,'B'))
print(m.tau(0,1,'V'))

s = generateSet(m,10,5)
print(s)

print(m.logLikelihood(s))