from examples_models import *
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from Estimation_algorithms_MCGT import BW_ON_MCGT
from Estimation_algorithms_MCGT import EM_ON_MCGT
from random import randint

EQUIP = 0
RANDOM = 1

def test1(seq):
	h = modelMCGT1()
	print(BW_ON_MCGT(h,['x','y','a','b','c','d']).problem1(seq))
	print()

def test3(seq,nb_states,alphabet,initialization_type):
	if initialization_type == EQUIP:
		h = modelMCGT_equiprobable(nb_states,alphabet)
	else:
		h = modelMCGT_random(nb_states,alphabet)

	print("With BW (random initialization):")
	BW_ON_MCGT(h,alphabet).problem3(seq)
	print()
	print("With EM (same random initialization):")
	res = EM_ON_MCGT(h,alphabet).problem3(seq)
	res.UPPAAL_convert("test")
	print()
	#return res

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

	alphabet = ['x','y','a','b','c','d']
	h = modelMCGT_random(5,alphabet)
	h.pprint()

	#res = []
	print("With EM (same random initialization):")
	algo = Estimation_algorithm_MCGT(h,alphabet)
	res1 = algo.problem3multiple([res_outputs,res_val])
	print(res1)
	algo.h.UPPAAL_convert("test")

#test1("xbd")
#test3("xada",5,['x','y','a','b','c','d'],RANDOM)
test3multiple(10000,4)