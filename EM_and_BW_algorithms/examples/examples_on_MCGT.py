import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples_models import *
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from Estimation_algorithms_MCGT import BW_ON_MCGT
from Estimation_algorithms_MCGT import EM_ON_MCGT
from random import randint

EQUIP = 0
RANDOM = 1

def getAlphabetFromSequences(sequences):
	if type(sequences) == str:
		return list(set(sequences))
	else:
		seq = ""
		sequences = list(set(sequences))
		for i in sequences:
			seq += i
		return list(set(seq))

def test1(seq):
	h = modelMCGT1()
	print(BW_ON_MCGT(h,['x','y','a','b','c','d']).problem1(seq))
	print()

def test3(seq,nb_states,initialization_type):
	alphabet = getAlphabetFromSequences(seq)
	if initialization_type == EQUIP:
		h = modelMCGT_equiprobable(nb_states, alphabet)
	else:
		h = modelMCGT_random(nb_states, alphabet)

	print("With EM :")
	res = EM_ON_MCGT(h, alphabet).problem3(seq)
	#res.UPPAAL_convert("test")
	print()

def test3multiple(nb_runs,runs_length):
	res_outputs = []
	res_val = []
	h = modelMCGT1()

	for i in range(nb_runs):        
		res = h.run(runs_length)
		
		if not res in res_outputs:
			res_outputs.append(res)
			res_val.append(0)

		res_val[res_outputs.index(res)] += 1


	alphabet = getAlphabetFromSequences(res_outputs)
	print(alphabet)
	h = modelMCGT_random(5,alphabet)
	h.pprint()

	#res = []
	print("With EM (same random initialization):")
	algo = Estimation_algorithm_MCGT(h, alphabet)
	res1 = algo.problem3multiple([res_outputs,res_val])
	print(res1)
	#algo.h.UPPAAL_convert("test")

#test1("xbd")
#test3("xad",3,EQUIP)
#test3multiple(100000,5)
#------------------------------------------------------------------------------

fout = open("results.py",'w')


min_state = 1
max_state = 5

runs_length = 6
nb_runs_by_states = 10

training_set_seq = []
training_set_val = []
test_set_seq = []
test_set_val = []


h = modelMCGT4()

for i in range(1000): 
	res = h.run(runs_length)
	
	if not res in training_set_seq:
		training_set_seq.append(res)
		training_set_val.append(0)

	training_set_val[training_set_seq.index(res)] += 1

for i in range(100000): 
	res = h.run(runs_length)
	
	if not res in test_set_seq:
		test_set_seq.append(res)
		test_set_val.append(0)

	test_set_val[test_set_seq.index(res)] += 1


alphabet = getAlphabetFromSequences(test_set_seq)
fout.write("RES = [\n")

for states in range(min_state,max_state+1):
	fout.write("\t  [")
	for iii in range(nb_runs_by_states):
		h = modelMCGT_random(states,alphabet)
		print("estimating with",states," , run nb:", iii)
		algo = Estimation_algorithm_MCGT(h, alphabet)
		algo.problem3multiple([training_set_seq,training_set_val])
		algo.sequences = [test_set_seq,test_set_val]
		fout.write(str(algo.logLikelihood()))
		if iii == nb_runs_by_states-1:
			fout.write("]")
			if states < max_state:
				fout.write(",\n")
			else:
				fout.write("]\n")
		else:
			fout.write(',')
fout.close()

from results import RES
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.violinplot(RES,showmeans=False,showmedians=True)
ax.set_xticks([y + min_state for y in range(len(RES))])
ax.set_xlabel('Number of states')
ax.set_ylabel('loglikelihood')# add x-tick labels
plt.setp(ax,xticks=[y + min_state for y in range(len(RES))], xticklabels=[str(y + min_state) for y in range(len(RES))])
plt.show()