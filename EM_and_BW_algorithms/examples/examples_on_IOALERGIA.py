from examples_models import modelMDP5, scheduler_random, modelMDP_random
from IOalergia import IOAlergia
from Estimation_algorithms_MDP import Estimation_algorithm_MDP_sequences


def generateSet(set_size,sequence_size,scheduler,model):
	seq = []
	val = []
	for i in range(set_size):
		trace = model.run(sequence_size, scheduler)

		if not trace in seq:
			seq.append(trace)
			val.append(0)

		val[seq.index(trace)] += 1

	return [seq,val]

alphabet = ['A','B','C']
actions = ['a','b']
m = modelMDP5()
s = scheduler_random(actions)

training_set_size = 2000
test_set_size = 200
sequence_size = 6

training_set = generateSet(training_set_size, sequence_size, s,m)
test_set = generateSet(test_set_size, sequence_size, s,m)
algo = IOAlergia(training_set, 0.5)
m = algo.learn()
resAlergia = m.logLikelihoodTraces(test_set)
while resAlergia == -256:
	training_set = generateSet(training_set_size, sequence_size, s,m)
	test_set = generateSet(test_set_size, sequence_size, s,m)
	algo = IOAlergia(training_set, 0.5)
	m = algo.learn()
	resAlergia = m.logLikelihoodTraces(test_set)

print("Alergia",resAlergia)

resEM = []
runEM = 10
nbstates = len(m.states)
for i in range(runEM):
	h = modelMDP_random(nbstates,alphabet,actions)
	algo = Estimation_algorithm_MDP_sequences(h,alphabet,actions)
	algo.problem3(training_set)
	resEM.append(algo.h.logLikelihoodTraces(test_set))

f = open("resEMIOAlergia.txt",'w')
f.write(resAlergia)
f.write('\n')
for i in resEM:
	f.write(i)
	f.write(',')
f.close()