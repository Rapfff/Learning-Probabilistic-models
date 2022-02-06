from examples.examples_models import modelCOMC_random, modelCOMC1 
from learning.BW_coMC import BW_coMC 
from tools import generateSet 
 
original = modelCOMC1() 
 
training_set = generateSet(original,1,4) 
 
 
m = modelCOMC_random(2,min_mu=0.0) 
algo = BW_coMC(m) 
 
mout = algo.learn(training_set) 
 
mout.pprint()