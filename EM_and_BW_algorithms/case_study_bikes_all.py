from examples.examples_models import modelMCGT_random
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from datetime import datetime

alphabet = ['--5']
alphabet += [ str(i) for i in range(-4,10)]
alphabet.append('++10')

f = open("datasets/days377_group.csv",'r')
l = f.readline()
l = f.readline()

training_set = []
test_set = []

while datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2019,1,1,0,0):
	l = l[1:]
	if not l in training_set[0]:
		training_set[0].append(l)
		training_set[1].append(1)
	else:
		training_set[1][training_set[0].index(l)] += 1
	l = f.readline()

while datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2020,1,1,0,0):
	l = l[1:]
	if not l in test_set[0]:
		test_set[0].append(l)
		test_set[1].append(1)
	else:
		test_set[1][test_set[0].index(l)] += 1
	l = f.readline()

f.close()

nb_states = ??
nb_experiments = ??

for k in range(nb_experiments):
	print(datetime.now(),"iteration:",k)
	m = modelMCGT_random(n,alphabet)
	algo = Estimation_algorithm_MCGT(m,alphabet)
	algo.problem3(training_set)
	algo.h.save("case_study_bikes_all/modelEM_"+str(k+1)+".txt")
