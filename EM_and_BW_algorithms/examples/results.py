import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from MCGT import loadMCGT
from tools import generateSet

mcgt_EM = ?? 
mcgt_ALERGIA = ??

test_set_size = ??
test_set_length = ??
test_set_model = ??

###############################################

mcgt_EM = [ loadMCGT(mcgt_EM+str(i)+".txt") for i in range(1,11) ]
mcgt_ALERGIA = loadMCGT(mcgt_ALERGIA)

test_set = generateSet(test_set_model,test_set_size,test_set_length)


loglikelihood_EM =  [ mcgt_EM[i].logLikelihood(test_set) for i in range(10) ]
loglikelihood_ALERGIA = mcgt_ALERGIA.logLikelihood(test_set)

print("Result for EM      | "+str(max(loglikelihood_EM)))
print("Result for ALERGIA | "+str(loglikelihood_EM))

print("Best EM : "+str(loglikelihood_EM.index(max(loglikelihood_EM))))
