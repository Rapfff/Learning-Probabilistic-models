import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from examples.examples_models import modelCTMC2, modelCTMC3, modelCTMC_random
from src.learning.BW_CTMC_Composition import BW_CTMC_Composition
from src.models.CTMC import parallelComposition
from src.tools import generateSet, saveSet
from statistics import mean, stdev
from datetime import datetime
import matplotlib.pyplot as plt

original1 = modelCTMC2()
original2 = modelCTMC3()

original_model = parallelComposition(original1,original2)
timed_training_set = generateSet(original_model,1000,10,timed=True)
saveSet(timed_training_set,"training_set.txt")
random1 = modelCTMC_random(4,list("rgb"),False)
random2 = modelCTMC_random(4,list("rgb"),False)
random1.save("random1.txt")
random2.save("random2.txt")

timed_model1, timed_model2 = BW_CTMC_Composition(random1,random2).learn(timed_training_set,verbose=True,pp="Timed approach")
timed_model1.save("timed_model1.txt")
timed_model2.save("timed_model2.txt")

untimed_sequences = []
for seq in timed_training_set[0]:
    untimed_sequences.append([seq[i] for i in range(0,len(seq),2)])
untimed_training_set = [untimed_sequences, timed_training_set[1]]

_, untimed_model2 = BW_CTMC_Composition(original1,random2).learn(untimed_training_set,verbose=True,pp="Untimed approach",fixed=1)
untimed_model2.save("untimed_model2.txt")

original2.pprint()
print()
timed_model2.pprint()
print()
untimed_model2.pprint()