from alergia import Alergia
from examples.examples_models import modelMCGT_REBER, modelMCGT_random
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from tools import generateSet
import datetime

alphabet = ['B','E','P','S','T','V','X']
#original_model = modelMCGT_REBER()
#training_set = generateSet(original_model, 10000,5)
#print(training_set)
"""
f = open("big_test/training_set.txt",'w')
for i in training_set[0]:
	for j in i:
		f.write(i+' ')
	f.write('\n')
f.write('***\n')
for i in training_set[1]:
	f.write(str(i)+' ')
f.write('\n')
f.close()
"""
"""
nb_states = []
for alpha in [0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.02,0.01]:
	algo = Alergia(training_set,alpha)
	m = algo.learn()
	print(algo.alpha,len(m.states))

	if not len(m.states) in nb_states:
		nb_states.append(len(m.states))
		m.name = "model alergia "+str(alpha)+" "+str(len(m.states))
		m.save("big_test/modelAlergia_"+str(len(m.states))+".txt")
"""
nb_states = [11,12,13,14,15]
training_set = [['BTSXS', 'BPTTT', 'BTXXT', 'BPVPS', 'BTSXX', 'BTXSE', 'BTSSX', 'BTSSS', 'BPTVV', 'BPTVP', 'BPVVE', 'BPTTV', 'BPVPX', 'BTXXV'],
				[582, 1739, 721, 405, 598, 952, 764, 1065, 497, 505, 768, 777, 346, 281]]


for n in nb_states:
	for k in range(10):
		print(datetime.datetime.now(),"states:",n,"iteration:",k)
		m = modelMCGT_random(n,alphabet)
		algo = Estimation_algorithm_MCGT(m,alphabet)
		algo.problem3(training_set)
		algo.h.save("big_test/modelEM_"+str(n)+'_'+str(k+1)+".txt")
