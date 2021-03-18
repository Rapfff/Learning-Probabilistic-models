from examples.examples_models import modelMCGT_random
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
import datetime

alphabet = ['--5']
alphabet += [ str(i) for i in range(-4,10)]
alphabet.append('++10')
print(alphabet)

f = open("datasets/bikes/days377_group.csv",'r')
l = f.readline()
l = f.readline()[:-1].split(',')

training_set = [[],[]]
test_set = [[],[]]

day = 2 # we start with wednesday

while datetime.datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2017,1,1,0,0):
	l = f.readline()[:-1].split(',')
	day = (day+1) % 7

while datetime.datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2019,1,1,0,0):
	if day < 5: # we remove the weekends
		l = l[9:-4] #  we keep just 8am-8pm
		if not l in training_set[0]:
			training_set[0].append(l)
			training_set[1].append(1)
		else:
			training_set[1][training_set[0].index(l)] += 1
	
	l = f.readline()[:-1].split(',')
	day = (day+1) % 7

while datetime.datetime.strptime(l[0],'%Y-%m-%d') < datetime.datetime(2020,1,1,0,0):
	if day < 5: # we remove the weekends
		l = l[9:-4] #  we keep just 8am-8pm
		if not l in test_set[0]:
			test_set[0].append(l)
			test_set[1].append(1)
		else:
			test_set[1][test_set[0].index(l)] += 1
	l = f.readline()[:-1].split(',')
	day = (day+1) % 7

f.close()
print(training_set)
print(test_set)
nb_states = 6
nb_experiments = 10

for k in range(nb_experiments):
	print(datetime.datetime.now(),"iteration:",k)
	m = modelMCGT_random(nb_states,alphabet)
	algo = Estimation_algorithm_MCGT(m,alphabet)
	algo.problem3(training_set,k)
	algo.h.save("case_study_bikes_all/modelEM_"+str(k+1)+".txt")
