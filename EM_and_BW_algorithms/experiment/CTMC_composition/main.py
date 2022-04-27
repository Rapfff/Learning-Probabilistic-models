from cProfile import label
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from examples.examples_models import modelCTMC2, modelCTMC3, modelCTMC_random
from src.learning.BW_CTMC_Composition import BW_CTMC_Composition
from src.learning.BW_CTMC import BW_CTMC
from src.models.CTMC import parallelComposition
from src.tools import generateSet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from ast import literal_eval

def generateTestSets(nb_seq=1000,len_seq=10):
    untimed_test_set_composition = generateSet(original_model,nb_seq,len_seq,timed=True)
    untimed_test_set_2 = generateSet(original2,nb_seq,len_seq,timed=True)
    return untimed_test_set_composition, untimed_test_set_2

def generateTrainingSets(nb_seq=1000,len_seq=10):    
    timed_training_set = generateSet(original_model,nb_seq,len_seq,timed=True)
    
    untimed_training_set = [[],[]]
    for seq,times in zip(timed_training_set[0],timed_training_set[1]):
        useq = [seq[i] for i in range(1,len(seq),2)]
        if useq in untimed_training_set[0]:
            untimed_training_set[1][untimed_training_set[0].index(useq)] += times
        else:
            untimed_training_set[0].append(useq)
            untimed_training_set[1].append(times)
    return timed_training_set, untimed_training_set

def generateRandomModels(disjoint=False):
    if not disjoint:
        return modelCTMC_random(4,list("rgb"),1,5,False), modelCTMC_random(4,list("rgb"),1,5,False)
    else:
        return modelCTMC_random(4,['r1','g1','b1'],1,5,False), modelCTMC_random(4,['r2','g2','b2'],1,5,False)



original1 = modelCTMC2()
original2 = modelCTMC3()
original_model = parallelComposition(original1,original2,disjoint=True)
timed_test_set_composition, timed_test_set_2 = generateTestSets()
timed_training_set, untimed_training_set = generateTrainingSets(1,5)
print(untimed_training_set)
random1, random2 = generateRandomModels(disjoint=True)

s = datetime.now()
timed_model1, timed_model2 = BW_CTMC_Composition(random1,random2).learn(timed_training_set,verbose=True)
s = (datetime.now()-s).total_seconds()

timed_model = parallelComposition(timed_model1,timed_model2)
quality_original = original_model.logLikelihood(timed_test_set_composition)
quality_original2= original2.logLikelihood(timed_test_set_2)
ll = abs(quality_original-timed_model.logLikelihood(timed_test_set_composition))
print(s,ll)
ll = max(abs(quality_original2-timed_model2.logLikelihood(timed_test_set_2)),abs(quality_original2-timed_model1.logLikelihood(timed_test_set_2)))
print(s,ll)
input()


NB_EXPERIMENTS = 100

dots_compo_compo  = []
dots_compo_model2  = []
dots_simple_compo = []
dots_model2_compo = []
dots_model2_model2 = []

original1 = modelCTMC2()
original2 = modelCTMC3()
original_model = parallelComposition(original1,original2)
timed_test_set_composition, timed_test_set_2 = generateTestSets()
quality_original = original_model.logLikelihood(timed_test_set_composition)
quality_original2= original2.logLikelihood(timed_test_set_2)

duration_tot = timedelta()
for exp in range(1,NB_EXPERIMENTS+1):
    start_exp = datetime.now()
    print("Experiment",exp,'/',NB_EXPERIMENTS,':',end=" ")
    timed_training_set, untimed_training_set = generateTrainingSets(1000,10)
    random1, random2 = generateRandomModels()


    print("Composition,",end=" ")
    s = datetime.now()
    timed_model1, timed_model2 = BW_CTMC_Composition(random1,random2).learn(timed_training_set,output_file="output/compo_"+str(exp))
    s = (datetime.now()-s).total_seconds()
    timed_model = parallelComposition(timed_model1,timed_model2)
    ll = abs(quality_original-timed_model.logLikelihood(timed_test_set_composition))
    dots_compo_compo.append((s,ll))
    ll = max(abs(quality_original2-timed_model2.logLikelihood(timed_test_set_2)),abs(quality_original2-timed_model1.logLikelihood(timed_test_set_2)))
    dots_compo_model2.append((s,ll))
    
    print("Simple,",end=" ")
    s = datetime.now()
    simple = BW_CTMC(parallelComposition(random1,random2)).learn(timed_training_set,output_file="output/simple_"+str(exp)+".txt")
    s = (datetime.now()-s).total_seconds()
    ll = abs(quality_original-simple.logLikelihood(timed_test_set_composition))
    dots_simple_compo.append((s,ll))

    print("Model2")
    s = datetime.now()
    _, untimed_model2 = BW_CTMC_Composition(original1,random2).learn(untimed_training_set,output_file="output/model2_"+str(exp),to_update=2)
    s = (datetime.now()-s).total_seconds()
    ll = abs(quality_original-parallelComposition(_,untimed_model2).logLikelihood(timed_test_set_composition))
    dots_model2_compo.append((s,ll))
    ll = abs(quality_original2-untimed_model2.logLikelihood(timed_test_set_2))
    dots_model2_model2.append((s,ll))


    duration_tot += datetime.now()-start_exp
    print(datetime.now(),". Expected ETA:",datetime.now()+(NB_EXPERIMENTS-exp)*duration_tot/exp)


f = open("output/summary.txt",'w')
f.write(str(dots_compo_compo)+'\n')
f.write(str(dots_compo_model2)+'\n')
f.write(str(dots_simple_compo)+'\n')
f.write(str(dots_model2_compo)+'\n')
f.write(str(dots_model2_model2)+'\n')
f.close()

f = open("output/summary.txt",'r')
dots_compo_compo = literal_eval(f.readline()[:-1])
dots_compo_model2 = literal_eval(f.readline()[:-1])
dots_simple_compo = literal_eval(f.readline()[:-1])
dots_model2_compo = literal_eval(f.readline()[:-1])
dots_model2_model2 = literal_eval(f.readline()[:-1])
f.close()


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter([i[0] for i in dots_simple_compo], [i[1] for i in dots_simple_compo], c='g', alpha=0.5, label="Exp. 1: U||V")
ax.scatter([i[0] for i in dots_compo_compo],  [i[1] for i in dots_compo_compo],  c='r', alpha=0.5, label="Exp. 2: U and V")
ax.scatter([i[0] for i in dots_model2_compo], [i[1] for i in dots_model2_compo], c='b', alpha=0.5, label="Exp. 3: V")
ax.set_xlabel("Running time (s)")
ax.set_ylabel("Average of the absolute differences of the loglikelihoods")
ax.set_title("Composition")
ax.legend()
plt.show()


fig, ax = plt.subplots(figsize=(5,5))
ax.scatter([i[0] for i in dots_compo_model2], [i[1] for i in dots_compo_model2], c='r', alpha=0.5, label="Exp. 2: U and V")
ax.scatter([i[0] for i in dots_model2_model2],[i[1] for i in dots_model2_model2],c='b', alpha=0.5, label="Exp. 3: V")
ax.set_xlabel("Running time (s)")
ax.set_ylabel("Average of the absolute differences of the loglikelihoods")
ax.set_title("Second component only")
ax.legend()
plt.show()