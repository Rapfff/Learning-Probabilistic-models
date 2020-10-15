from examples_models import *
from Estimation_algorithms_MCGT_multiple import EM_ON_MCGT
from Estimation_algorithms_MCGT_multiple import BW_ON_MCGT
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
	print("With EM with TUK (same random initialization):")
	EM_ON_MCGT(h,alphabet).problem3(seq)
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
	res1 = EM_ON_MCGT(h,alphabet).problem3multiple([res_outputs,res_val])
	#res.append(EM_ON_MCGT(h,alphabet).problem3multiple([res_outputs,res_val]))
	print(res1)
	print()
	#print("With BW (random initialization):")
	res1 = BW_ON_MCGT(h,alphabet).problem3multiple([res_outputs,res_val])
	#res.append(BW_ON_MCGT(h,alphabet).problem3multiple([res_outputs,res_val]))
	print(res1)
	#print()
	#return res


"""
min_length = 2
max_length = 3
#min_nb_states = 4
#max_nb_states = 8
alphabet = ['x','y','a','b','c','d']
nb_exp = 10
likelihood = [[],[]]
duration = [[],[]]
axx  = []

for i in range(min_length,max_length+1):
	axx.append(i)
	likelihood[0].append(0)
	likelihood[1].append(0)
	duration[0].append(0)
	duration[1].append(0)

	print("sequence of length",i)
	for j in range(nb_exp):
		print(j,end=" ")
		tt = test3multiple(10000,i)
		likelihood[0][-1] += tt[0][0]
		likelihood[1][-1] += tt[1][0]
		duration[0][-1] += tt[0][1]
		duration[1][-1] += tt[1][1]

	likelihood[0][-1] /= nb_exp
	likelihood[1][-1] /= nb_exp
	duration[0][-1] /= nb_exp
	duration[1][-1] /= nb_exp


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)

ax1.plot(axx, likelihood[0],label="EM")
ax1.plot(axx, likelihood[1],label="BW")

ax2.plot(axx, duration[0],label="EM")
ax2.plot(axx, duration[1],label="BW")

ax1.set(xlabel='length of the sequences', ylabel='average final likelihood')
ax2.set(xlabel='length of the sequences', ylabel='duration (seconds)')
plt.legend()
plt.show()
"""

#test1("xbd")
#test2("$b")
#test3("xada",5,['x','y','a','b','c','d'],RANDOM)
test3multiple(1,4)