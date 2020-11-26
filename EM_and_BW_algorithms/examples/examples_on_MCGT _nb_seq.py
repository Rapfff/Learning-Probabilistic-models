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

runs_length = 6
res_outputs1 = []
res_val1 = []
res_outputs2 = []
res_val2 = []
res_outputs3 = []
res_val3 = []
res_outputs4 = []
res_val4 = []
res_outputs5 = []
res_val5 = []
res_outputs_ref = []
res_val_ref = []

h = modelMCGT4()

for i in range(10): 
	res = h.run(runs_length)
	
	if not res in res_outputs1:
		res_outputs1.append(res)
		res_val1.append(0)

	res_val1[res_outputs1.index(res)] += 1

for i in range(100): 
	res = h.run(runs_length)
	
	if not res in res_outputs2:
		res_outputs2.append(res)
		res_val2.append(0)

	res_val2[res_outputs2.index(res)] += 1

for i in range(1000): 
	res = h.run(runs_length)
	
	if not res in res_outputs3:
		res_outputs3.append(res)
		res_val3.append(0)

	res_val3[res_outputs3.index(res)] += 1

for i in range(10000): 
	res = h.run(runs_length)
	
	if not res in res_outputs4:
		res_outputs4.append(res)
		res_val4.append(0)

	res_val4[res_outputs4.index(res)] += 1

for i in range(100000): 
	res = h.run(runs_length)
	
	if not res in res_outputs5:
		res_outputs5.append(res)
		res_val5.append(0)

	res_val5[res_outputs5.index(res)] += 1

for i in range(100000): 
	res = h.run(runs_length)
	
	if not res in res_outputs_ref:
		res_outputs_ref.append(res)
		res_val_ref.append(0)

	res_val_ref[res_outputs_ref.index(res)] += 1


alphabet = getAlphabetFromSequences(res_outputs_ref)
h = modelMCGT_random(6,alphabet)
#print("estimating with 10...")
#algo = Estimation_algorithm_MCGT(h, alphabet)
#time1 = algo.problem3multiple([res_outputs1,res_val1])[1]
#algo.sequences = [res_outputs_ref,res_val_ref]
#res1 = algo.logLikelihood()

print("estimating with 100...")
algo = Estimation_algorithm_MCGT(h, alphabet)
time2 = algo.problem3multiple([res_outputs2,res_val2])[1]
algo.sequences = [res_outputs_ref,res_val_ref]
res2 = algo.logLikelihood()


print("estimating with 1000...")
algo = Estimation_algorithm_MCGT(h, alphabet)
time3 = algo.problem3multiple([res_outputs3,res_val3])[1]
algo.sequences = [res_outputs_ref,res_val_ref]
res3 = algo.logLikelihood()


print("estimating with 10000...")
algo = Estimation_algorithm_MCGT(h, alphabet)
time4 = algo.problem3multiple([res_outputs4,res_val4])[1]
algo.sequences = [res_outputs_ref,res_val_ref]
res4 = algo.logLikelihood()

print("estimating with 100000...")
algo = Estimation_algorithm_MCGT(h, alphabet)
time5 = algo.problem3multiple([res_outputs5,res_val5])[1]
algo.sequences = [res_outputs_ref,res_val_ref]
res5 = algo.logLikelihood()

outf = open("resultats.py",'w')
#outf.write("LOGLIKELI=["+str(res1)+","+str(res2)+","+str(res3)+","+str(res4)+"]\n")
#outf.write("TIME=["+str(time1)+","+str(time2)+","+str(time3)+","+str(time4)+"]\n")
outf.write("LOGLIKELI=["+str(res2)+","+str(res3)+","+str(res4)+","+str(res5)+"]\n")
outf.write("TIME=["+str(time2)+","+str(time3)+","+str(time4)+","+str(time5)+"]\n")
outf.close()