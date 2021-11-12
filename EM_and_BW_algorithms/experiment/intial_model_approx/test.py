from Estimation_algorithm_MCGT import Estimation_algorithm_MCGT
from Estimation_algorithm_MDP import Estimation_algorithm_MDP
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)
from examples.examples_models import modelMCGT_REBER, modelMCGT_random, modelMDP_random, modelMDP5, scheduler_uniform
from src.tools import generateSet
from src.models.MDP import loadMDP
from numpy import log,inf
from datetime import datetime
from random import shuffle
from shutil import rmtree
from scipy.optimize import curve_fit

def func(x,a,b,k):
	return -a*x**(-k)+b

def approx(values,knowledge_default,window):
	knowledge = min(knowledge_default,len(values))
	ydata = values[:knowledge]+[values[knowledge-1]]*(window-knowledge)
	xdata = range(1,window+1)
	try:
		popt, pcov = curve_fit(func,xdata,ydata)
		return func(range(1,len(values)), *popt)
	except RuntimeError:
		return False

nb_models = 1000
nb_iterations = 35
nb_to_keep = 10
folder = "exp3/"
os.mkdir(folder[:-1])
knowledge_default = nb_iterations
window = 200
"""
original = modelMCGT_REBER()
training_set = generateSet(original,1000,8)
test_set = generateSet(original,200,8)

algo = Estimation_algorithm_MCGT(modelMCGT_random(7,original.observations()),original.observations())
algo.learn(training_set,"loglikelihoods.csv",test_set)
"""

original = modelMDP5()
sched = scheduler_uniform(original.actions())

training_set = generateSet(original,1000,8,sched)
test_set = generateSet(original,200,8,sched)

algo = Estimation_algorithm_MDP(modelMDP_random(4,original.observations(),original.actions()),original.observations(),original.actions())
algo.learn(training_set,folder+"loglikelihoods",test_set)
fin = open(folder+"loglikelihoods.csv",'r')
l = [float(i) for i in fin.readline()[:-1].split(',')]
fin.close()
rmtree(folder[:-1])
estimated = approx(l,knowledge_default,window)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
t = min(len(l),len(estimated))
ax.scatter(range(1,knowledge_default+1),l[:knowledge_default],alpha=0.5)
ax.scatter(range(knowledge_default+1,t+1),l[knowledge_default:t],alpha=0.5)
ax.plot(range(1,t+1),estimated[:t],'r',label="estimation")
ax.set_xlabel("BW iterations")
ax.set_ylabel("log-likelihood of the test set")
ax.legend()
plt.show()

"""
for m in range(nb_models):
	modelMDP_random(4,original.observations(),original.actions()).save(folder+"init_model"+str(m)+".txt")

###################################################
start_time_approx = datetime.now()
v = []
for i in range(nb_models):
	print("Approx",i+1,"/",nb_models)
	algo = Estimation_algorithm_MDP(loadMDP(folder+"init_model"+str(i)+".txt"),original.observations(),original.actions())
	algo.learn(training_set,folder+"loglikelihoods"+str(i),test_set,nb_iterations=nb_iterations)
	fin = open(folder+"loglikelihoods"+str(i)+".csv",'r')
	l = [float(i) for i in fin.readline()[:-1].split(',')]
	v.append(l)
	fin.close()

knowledge_default = nb_iterations
window = 200

estimated = []
i = 0
while i < len(v):
	print(i)
	t = approx(v[i],knowledge_default,window)
	if not t:
		v = v[:i]+v[i+1:]
		i -= 1
	else:
		estimated.append(func(range(1,len(v[i])), *popt))
	i += 1

v_estimated =   [estimated[i][-1] for i in range(len(estimated))]
v_estimated_sorted = v_estimated[:]
v_estimated_sorted.sort()
kept = [v_estimated.index(v_estimated_sorted[-i]) for i in range(1,nb_to_keep+1)]

v_approx = []
for m in kept:
	print("Approx",kept.index(m)+1,"/",nb_to_keep)
	algo = Estimation_algorithm_MDP(loadMDP(folder+"loglikelihoods"+str(m)+"_after.txt"),original.observations(),original.actions())
	v_approx.append(algo.learn(training_set,folder+"loglikelihoods"+str(m)+".csv",test_set))
approx_time = datetime.now()-start_time_approx
###################################################
start_time = datetime.now()
kept = list(range(nb_models))
shuffle(kept)
k = 0
v = []
while datetime.now() - start_time < approx_time and k < len(kept):
	print("Classic",k)
	algo = Estimation_algorithm_MDP(loadMDP(folder+"init_model"+str(kept[k])+".txt"),original.observations(),original.actions())
	v.append(algo.learn(training_set,folder+"loglikelihoods"+str(kept[k])+".csv",test_set))
	k += 1
classic_time = datetime.now() - start_time
###################################################
print("Approx time :",approx_time)
print("Mean approx :",sum(v_approx)/len(v_approx))
print("Best approx :",max(v_approx))
print("Classic time:",classic_time)
print("Mean classic:",sum(v)/len(v))
print("Best classic:",max(v))

rmtree(folder[:-1])
"""