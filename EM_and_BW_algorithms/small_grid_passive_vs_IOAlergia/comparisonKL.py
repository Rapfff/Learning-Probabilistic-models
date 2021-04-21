import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from examples.examples_models import scheduler_uniform
from tools import generateSetUnique, generateSet
from MDP import loadMDP, KLDivergence, loadPrismMDP
import matplotlib.pyplot as plt
from statistics import mean

nb_states = 9

nb_seq = 50
len_seq= 10

alphas = ["0.15"]

training_set = [[],[]]

f = open("training_set.txt",'r')
l = f.readline()
while l:
	l = l.replace("'",'')
	l = l.replace(' ','')
	training_set[0].append(l[1:-2].split(','))
	l = f.readline()
	training_set[1].append(int(l[:-1]))
	l = f.readline()
f.close()


m = loadPrismMDP("small_grid.prism")
for i in range(len(m.states)):
	for a in m.states[i].actions():
		if len(m.states[i].next_matrix[a][0]) == 1 and m.states[i].next_matrix[a][1][0] == i: #moving toward wall
			m.states[i].next_matrix[a][2][0] = "Wall"

resOriginal_training = m.logLikelihoodTraces(training_set)

#print("Alergia")
#m2 = loadMDP("modelAlergia_"+str(0.3)+".txt")
#m2.pprint()
#resAlergia_training = m2.logLikelihoodTraces(training_set)
#input()
#resAlergia_test_KL  = KLDivergence(m,m2,test_set_KL)
#input()
#resAlergia_test_log = m2.logLikelihoodTraces(test_set)
#input()
#
s = scheduler_uniform(m.actions())
test_set_KL = generateSetUnique(m,nb_seq,len_seq,s)
test_set    = generateSet(m,nb_seq,len_seq,s)

resOriginal_test = m.logLikelihoodTraces(test_set)

t = []
t2= []
t3= []
for k in range(1,11):
	print("EM. Iteration",k)
	m2 = loadMDP("modelEM_"+str(k)+".txt")
	t.append(KLDivergence(m,m2,test_set_KL))
	t2.append(m2.logLikelihoodTraces(test_set))
	t3.append(m2.logLikelihoodTraces(training_set))
print(t)
print(t2)
print(t3)
to_remove = [i for i in range(len(t)) if (t[i] is None or t2[i] is None or t3[i] is None)]
print(to_remove)
t =  [ t[i] for i in range(len(t))  if not i in to_remove]
t2 = [t2[i] for i in range(len(t2)) if not i in to_remove]
t3 = [t3[i] for i in range(len(t3)) if not i in to_remove]
print(len(t))
if len(t) > 0:
	f = open("table_results.txt",'w')
	f.write("Results original model training_set, test_set:"+str(resOriginal_training)+', '+str(resOriginal_test)+"\n")
	#f.write("Results alergia model (training_set, test_set_KL, test_set_log):"+str(resAlergia_training)+','+str(resAlergia_test_KL)+','+str(resAlergia_test_log)+"\n")
	for i in range(len(t)):
		f.write("Results EM model (training_set, test_set, test_set_log):"+str(t3[i])+','+str(t[i])+','+str(t2[i])+"\n")
	f.close()
