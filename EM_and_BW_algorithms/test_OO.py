from examples.examples_models import modelMCGT_REBER, modelHMM4, modelCOMC1
from src.tools import generateSet
from src.models.MCGT import loadMCGT
from src.models.HMM import loadHMM
from src.models.coMC import loadcoMC
from os import remove

m = modelMCGT_REBER()

m.pprint()
print(m.observations())
print(m.tau(0,1,'B'))
print(m.tau(0,2,'B'))
print(m.tau(0,1,'V'))

s = generateSet(m,10,5)
print(s)

print(m.logLikelihood(s))

m.save("test_save.txt")

m2 = loadMCGT("test_save.txt")
m2.pprint()
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

m.save("test_save.txt")

m2 = loadHMM("test_save.txt")
m2.pprint()
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

remove("test_save.txt")


