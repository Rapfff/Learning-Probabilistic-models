from Active_Learning_MDP import Active_Learning_MDP
from examples.examples_models import modelMDP5, scheduler_uniform, modelMDP_random
from tools import generateSet

m = modelMDP5()
s = scheduler_uniform(m.actions())
t = generateSet(m,50,5,s)


algo = Active_Learning_MDP(m, m.observations(),m.actions())
algo.learn(t,0.2,0.25,10,50)