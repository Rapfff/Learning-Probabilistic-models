import os, sys

from matplotlib import use
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from examples.examples_models import modelCTMC2, modelCTMC3, modelCTMC_random
from src.learning.BW_CTMC_Composition import BW_CTMC_Composition
from src.learning.BW_CTMC import BW_CTMC
from src.models.CTMC import parallelComposition, loadCTMC
from src.tools import generateSet, saveSet, loadSet, setFromList
from statistics import mean, stdev
from datetime import datetime
import matplotlib.pyplot as plt

original1 = modelCTMC2()
original2 = modelCTMC3()
original_model = parallelComposition(original1,original2)

#timed_training_set = generateSet(original_model,1000,10,timed=True)
#saveSet(timed_training_set,"output/training_set.txt")
#random1 = modelCTMC_random(4,list("rgb"),1,5,False)
#random2 = modelCTMC_random(4,list("rgb"),1,5,False)
#random1.save("output/random1.txt")
#random2.save("output/random2.txt")
random1 = loadCTMC("output/random1.txt")
random2 = loadCTMC("output/random2.txt")
timed_training_set = loadSet("output/training_set.txt")
for i,seq in enumerate(timed_training_set[0]):
    for j in range(0,len(seq),2):
        timed_training_set[0][i][j] = float(seq[j])
#timed_model1, timed_model2 = BW_CTMC_Composition(random1,random2).learn(timed_training_set,verbose=True,pp="Timed approach")
#timed_model1.save("output/timed_model1.txt")
#timed_model2.save("output/timed_model2.txt")

#original_model.pprint()
#parallelComposition(timed_model1,timed_model2).pprint()
#input()

untimed_training_set = [[],[]]
for seq,times in zip(timed_training_set[0],timed_training_set[1]):
    useq = [seq[i] for i in range(1,len(seq),2)]
    if useq in untimed_training_set[0]:
        untimed_training_set[1][untimed_training_set[0].index(useq)] += times
    else:
        untimed_training_set[0].append(useq)
        untimed_training_set[1].append(times)


_, untimed_model2 = BW_CTMC_Composition(original1,random2).learn(untimed_training_set,verbose=True,pp="Untimed approach",fixed=1)
untimed_model2.save("output/untimed_model2.txt")

"""
simple = BW_CTMC(parallelComposition(random1,random2)).learn(timed_training_set,output_file="output/simple.txt",verbose=True)


timed_model1 = loadCTMC("output/timed_model1.txt")
timed_model2 = loadCTMC("output/timed_model2.txt")
timed_model  = parallelComposition(timed_model1,timed_model2)

untimed_model2 = loadCTMC("output/untimed_model2.txt")
untimed_model  = parallelComposition(original1,untimed_model2)

untimed_test_set_composition = generateSet(original_model,1000,10,timed=False)

print("Original composition on test set:",original_model.logLikelihood(untimed_test_set_composition))
print("Random   composition on test set:",parallelComposition(random1,random2).logLikelihood(untimed_test_set_composition))
print("Output1  composition on test set:",timed_model.logLikelihood(untimed_test_set_composition))
print("Output2  composition on test set:",untimed_model.logLikelihood(untimed_test_set_composition))
print("Simple               on test set:",simple.logLikelihood(untimed_test_set_composition))
print()
"""
untimed_test_set_2 = generateSet(original2,1000,10,timed=False)
print("Original 2 on test set:",original2.logLikelihood(untimed_test_set_2))
print("Random   2 on test set:",random2.logLikelihood(untimed_test_set_2))
print("Output   2 on test set:",untimed_model2.logLikelihood(untimed_test_set_2))
