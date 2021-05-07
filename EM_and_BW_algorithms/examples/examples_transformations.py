import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.models.MCGT import HMMtoMCGT
from src.models.MCGS import HMMtoMCGS
from examples_models import modelHMM4

def test(nb_runs,runs_length):
	res_outputs = []
	res_val = [[],[],[]]

	h = modelHMM4()
	models = [h, HMMtoMCGT(h), HMMtoMCGS(h)]

	for i in range(nb_runs):
		for m in range(len(models)):
			
			res = models[m].run(runs_length)
			
			if not res in res_outputs:
				res_outputs.append(res)
				for mm in range(len(models)):
					res_val[mm].append(0)

			res_val[m][res_outputs.index(res)] += 1

	for m in range(len(models)):
		for i in range(len(res_outputs)):
			res_val[m][i] = str((res_val[m][i]*100)/nb_runs)

	print("OUTPUT\tHMM\tMCGT\tMCGS",end='')
	for i in range(len(res_outputs)):
		print('\n'+res_outputs[i]+'\t',end='')
		for m in range(len(models)):
			print(res_val[m][i]+'\t',end='')
	
test(1000000,3)