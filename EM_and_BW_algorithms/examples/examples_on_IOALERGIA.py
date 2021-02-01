from examples_models import modelMDP4, scheduler_random
from IOalergia import IOAlergia

alphabet = ['A','B','C']
actions = ['a','b']
m = modelMDP4()
s = scheduler_random(actions)

training_set_seq = []
training_set_val = []

for i in range(100):
	trace = m.run(5, s)

	if not trace in training_set_seq:
		training_set_seq.append(trace)
		training_set_val.append(0)

	training_set_val[training_set_seq.index(trace)] += 1

training_set = [training_set_seq,training_set_val]
algo = IOAlergia(training_set, 0.05)
m = algo.learn()
m.pprint()
print(m.logLikelihoodTraces(training_set))