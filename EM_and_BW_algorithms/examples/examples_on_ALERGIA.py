from examples_models import modelMCGT4, modelMCGT_random
from alergia import Alergia
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from time import time
from numpy import mean

alphabet = ['x','y','a','b','c','d','e']

nb_runs = 1000
runs_length = 6

res_outputs = []
res_val = []
h = modelMCGT4()

for i in range(nb_runs):        
	res = h.run(runs_length)
	
	if not res in res_outputs:
		res_outputs.append(res)
		res_val.append(0)

	res_val[res_outputs.index(res)] += 1
trainning_set = [res_outputs,res_val]

nb_runs = 100000
runs_length = 6

res_outputs = []
res_val = []

for i in range(nb_runs):
	res = h.run(runs_length)
	
	if not res in res_outputs:
		res_outputs.append(res)
		res_val.append(0)

	res_val[res_outputs.index(res)] += 1
test_set = [res_outputs,res_val]


start_time = time()
modelAlergia = Alergia(trainning_set,0.05)
modelAlergia = modelAlergia.learn()
modelAlergia.pprint()
resAlergia = modelAlergia.logLikelihood(test_set)
Alergia_duration = time()-start_time

EM_duration = []
resEM = []
for i in range(10):
	h = modelMCGT_random(6,alphabet)
	start_time = time()
	algo = Estimation_algorithm_MCGT(h, alphabet)
	algo.problem3multiple(trainning_set)
	resEM.append( algo.h.logLikelihood(test_set) )
	EM_duration.append( time()-start_time )
EM_duration = mean(EM_duration)
resEM = mean(resEM)


start_time = time()
algo = Estimation_algorithm_MCGT(modelAlergia, alphabet)
algo.problem3multiple(trainning_set)
resEM_Alergia = algo.h.logLikelihood(test_set)
EM_Alergia_duration = time()-start_time + Alergia_duration

print("               |  ALERGIA   |     EM     | ALERGIA+EM |")
print(" logLikelihood | "+str(resAlergia)[:10]+" | "+str(resEM)[:10]+" | "+str(resEM_Alergia)[:10]+" | ")
print("   duration    |    "+str(Alergia_duration)[:4]+"    |    "+str(EM_duration)[:4]+"    |    "+str(EM_Alergia_duration)[:4]+"    | ")
