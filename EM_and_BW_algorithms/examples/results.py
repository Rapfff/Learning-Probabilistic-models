import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from MCGT import loadMCGT
from tools import generateSet
from examples_models import modelMCGT_REBER

mcgt_EM = "REBERtests/"
mcgt_ALERGIA = "REBERtests/alergiamodel.txt"

test_set_size = 1000000
test_set_length = 5
test_set_model = modelMCGT_REBER()

###############################################

mcgt_EM = [ loadMCGT(mcgt_EM+str(i)+".txt") for i in range(1,11) ]
mcgt_ALERGIA = loadMCGT(mcgt_ALERGIA)

test_set = generateSet(test_set_model,test_set_size,test_set_length)


loglikelihood_EM =  [ mcgt_EM[i].logLikelihood(test_set) for i in range(10) ]
loglikelihood_ALERGIA = mcgt_ALERGIA.logLikelihood(test_set)

print("Result for EM(best)    | "+str(max(loglikelihood_EM)))
print("Result for EM(average) | "+str(sum(loglikelihood_EM)/len(loglikelihood_EM)))
print("Result for ALERGIA     | "+str(loglikelihood_ALERGIA))

print("Best EM : "+str(loglikelihood_EM.index(max(loglikelihood_EM))))
