import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples_models import *
from Estimation_algorithms_HMM import EMprime_ON_HMM
from Estimation_algorithms_HMM import EM_ON_HMM
from Estimation_algorithms_HMM import BW_ON_HMM

def test1(seq):
	h = modelHMM3()
	print("With BW:")
	print(BW_ON_HMM(h,['$','a','b']).problem1(seq))
	print()
	input()
	print("With EM:")
	print(EM_ON_HMM(h,['$','a','b']).problem1(seq))
	print()

def test2(seq):
	h = modelHMM3()
	print("With BW:")
	print(BW_ON_HMM(h,['$','a','b','c','d']).problem2(seq))
	print()
	input()
	print("With EM:")
	print(EM_ON_HMM(h,['$','a','b','c','d']).problem2(seq))
	print()

def test3(seq):
	h = modelHMM2_random()
	print("With BW (random initialization):")
	EM_ON_HMM(h,['x','y','a','b','d']).problem3(seq)
	print()
	input()
	print("With EM (same random initialization):")
	EM_ON_HMM(h,['x','y','a','b','d']).problem3(seq)
	print()

def test3multiple(nb_runs,runs_length):
	res_outputs = []
	res_val = []
	h = modelHMM2()
	models = [h]

	for i in range(nb_runs):		
		res = models[0].run(runs_length)
		
		if not res in res_outputs:
			res_outputs.append(res)
			res_val.append(0)

		res_val[res_outputs.index(res)] += 1

	h = modelHMM2_random()
	print("With BW (random initialization):")
	BW_ON_HMM(h,['x','y','a','b','d']).probem3multiple([res_outputs,res_val])
	print()
	input()
	print("With EM (same random initialization):")
	EM_ON_HMM(h,['x','y','a','b','d']).probem3multiple([res_outputs,res_val])
	print()
	


#test1("$b")
#test2("$b")
#test3("xad")
test3multiple(100000,4)