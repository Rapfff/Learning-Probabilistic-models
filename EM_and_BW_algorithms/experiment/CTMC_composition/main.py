import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from examples.examples_models import modelMC_map, modelMC_REBER, modelMC_random, modelCTMC_random
from src.learning.BW_MC import BW_MC
from src.learning.BW_CTMC_Composition import BW_CTMC_Composition
from src.tools import generateSet
from statistics import mean, stdev
from datetime import datetime
import matplotlib.pyplot as plt

def experiment(original_model, name, nb_states_small, nb_states_big,nb_experiments):
    running_time = [[],[],[]]
    quality = [[],[],[]]

    training_set = generateSet(original_model,1000,7)
    test_set     = generateSet(original_model,1000,7)
    
    f = open("points_Composition_"+name+".txt",'a')
    for c in range(nb_experiments):
        print("Composition ",c+1,"/",nb_experiments," ",name,sep="")
        s = datetime.now()
        r1 = modelCTMC_random(nb_states_small,original_model.observations(),self_loop=False)
        r2 = modelCTMC_random(nb_states_small,original_model.observations(),self_loop=False)
        output_model1 = BW_CTMC_Composition(r1,r2).learn(training_set)
        output_model1 = output_model1.toMC(name)
        running_time[0].append((datetime.now()-s).total_seconds())
        quality[0].append(output_model1.logLikelihood(test_set))
        f.write(str(running_time[0][-1])+'\n')
        f.write(str(quality[0][-1])+'\n')
    f.close()
    
    f = open("points_classicalbig_"+name+".txt",'a')
    for c in range(nb_experiments):
        print("Classic big",c+1,"/",nb_experiments," ",name,sep="")
        s = datetime.now()
        r3 = modelMC_random(nb_states_big,original_model.observations())
        output_model2 = BW_MC(r3).learn(training_set)
        running_time[1].append((datetime.now()-s).total_seconds())
        quality[1].append(output_model2.logLikelihood(test_set))
        f.write(str(running_time[1][-1])+'\n')
        f.write(str(quality[1][-1])+'\n')
    f.close()
    
    f = open("points_classicalsmall_"+name+".txt",'a') 
    for c in range(nb_experiments):
        print("Classic equiv",c+1,"/",nb_experiments," ",name,sep="")
        s = datetime.now()
        r3 = modelMC_random(len(original_model.states),original_model.observations())
        output_model2 = BW_MC(r3).learn(training_set)
        running_time[2].append((datetime.now()-s).total_seconds())
        quality[2].append(output_model2.logLikelihood(test_set))
        f.write(str(running_time[2][-1])+'\n')
        f.write(str(quality[2][-1])+'\n')
    f.close()
    
    return (running_time,quality)

nb_experiments = 0
running_time, quality = experiment(modelMC_REBER(),"REBER",4,16,nb_experiments)
"""
string1 = ""
string1 += "Model: REBER"+'\n'
string1 += "Average running time composition: "+str(running_time[0][0])+'\n'
string1 += "Std running time composition: "+str(running_time[0][1])+'\n'
string1 += "\n"
string1 += "Average running time classical big: "+str(running_time[1][0])+'\n'
string1 += "Std running time classical big: "+str(running_time[1][1])+'\n'
string1 += "\n"
string1 += "Average running time classical small: "+str(running_time[2][0])+'\n'
string1 += "Std running time classical small: "+str(running_time[2][1])+'\n'
string1 += "\n"
string1 += "Average loglikelihood composition: "+str(quality[0][0])+'\n'
string1 += "Std loglikelihood composition: "+str(quality[0][1])+'\n'
string1 += "\n"
string1 += "Average loglikelihood classical big: "+str(quality[1][0])+'\n'
string1 += "Std loglikelihood classical big: "+str(quality[1][1])+'\n'
string1 += '\n'
string1 += "Average loglikelihood classical small: "+str(quality[2][0])+'\n'
string1 += "Std loglikelihood classical small: "+str(quality[2][1])+'\n'
string1 += '\n'
running_time, quality = experiment(modelMC_map(),"MAP",5,25,10)
string1 += "Model: MAP"+'\n'
string1 += "Average running time composition: "+str(running_time[0][0])+'\n'
string1 += "Std running time composition: "+str(running_time[0][1])+'\n'
string1 += "\n"
string1 += "Average running time classical big: "+str(running_time[1][0])+'\n'
string1 += "Std running time classical big: "+str(running_time[1][1])+'\n'
string1 += "\n"
string1 += "Average running time classical small: "+str(running_time[2][0])+'\n'
string1 += "Std running time classical small: "+str(running_time[2][1])+'\n'
string1 += "\n"
string1 += "Average loglikelihood composition: "+str(quality[0][0])+'\n'
string1 += "Std loglikelihood composition: "+str(quality[0][1])+'\n'
string1 += "\n"
string1 += "Average loglikelihood classical big: "+str(quality[1][0])+'\n'
string1 += "Std loglikelihood classical big: "+str(quality[1][1])+'\n'
string1 += '\n'
string1 += "Average loglikelihood classical small: "+str(quality[2][0])+'\n'
string1 += "Std loglikelihood classical small: "+str(quality[2][1])+'\n'
string1 += '\n'
f = open("report3.txt",'w')
f.write(string2)
f.close()
"""

methods = ["Composition","Classical big","Classical small"]

running_time = [[],[],[]]
quality = [[],[],[]]

for i,n in enumerate(["Composition","classicalbig","classicalsmall"]):
    f = open("points_"+n+"_REBER.txt")
    l = f.readline()
    while l != "":
        running_time[i].append(float(l[:-1]))
        l = f.readline()
        quality[i].append(float(l[:-1]))
        l = f.readline()
    f.close()

# size and color:
colors = ['r','b','g']
fig, ax = plt.subplots()

for i in range(3):
    ax.scatter(running_time[i], quality[i],c=colors[i],label=methods[i], alpha=0.3,s=45)
ax.legend()
ax.set(xlabel='running time (s)', ylabel='average loglikelighood')
plt.show()