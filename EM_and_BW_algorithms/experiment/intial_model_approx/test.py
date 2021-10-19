import matplotlib.pyplot as plt
from Estimation_algorithms_MCGT import Estimation_algorithm_MCGT
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)
from examples.examples_models import modelMCGT_REBER
from src.tools import generateSet

original = modelMCGT_REBER()
training_set = generateSet(original,1000,8)
test_set = generateSet(original,200,8)

algo = Estimation_algorithm_MCGT(original,original.observations())
algo.learn(training_set,"loglikelihoods.csv",test_set)

fout = open("loglikelihoods.csv",'r')
l = [float(i) for i in fout.readline().split(',')]

x = range(len(l))

fig, ax = plt.subplots()
ax.plot(x, l)
plt.show()