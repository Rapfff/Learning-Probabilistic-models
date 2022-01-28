from examples.examples_models import modelCOMC_random, modelCOMC1
from learning.Estimation_algorithm_coMC import Estimation_algorithm_coMC
from tools import generateSet

original = modelCOMC1()

training_set = generateSet(original,100,10)


m = modelCOMC_random(2)
algo = Estimation_algorithm_coMC(m)

mout = algo.learn(training_set)

mout.pprint()