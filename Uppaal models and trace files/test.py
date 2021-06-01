import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import modelMCGT_random
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_MCGT import Estimation_algorithm_MCGT

nb_states = 1
seq_len = 6
alphabet = ['1','2','3','4','5','6']
# training_set = [[sequence1,sequence2,...],[number_of_sequence1,number_of_sequence2,...]]
# Example:
training_file = open("DiceModels/Dice1/dice1_trace", "r")
training_set = [[],[]]
l = training_file.readline()
stripped = l.strip('][\n').split(',')
cuttof = len(stripped) % seq_len
stripped = stripped[:-cuttof]
c = 1
seq = []
while c < len(stripped)+1:
	if len(seq) == seq_len :

		if seq in training_set[0]:
			training_set[1][training_set[0].index(seq)] += 1
		else:
			training_set[0].append(seq)
			training_set[1].append(1)
		seq = []

	seq.append(stripped[c-1][0])
	c += 1

print(training_set)


#while l:
#	stripped = l.strip('][\n').split(',')
#	seq =  [i[0] for i in stripped]
#	training_set[0].append(seq)
#	training_set[1].append(1)
#	l = training_file.readline()
training_file.close()

# modelMCGT_random(nb_states, alphabet)
m = modelMCGT_random(nb_states,alphabet)
m.pprint()


# Estimation_algorithms_MCGT_multiple.Estimation_algorithm_MCGT(initial_model,alphabet)
EM_algo = Estimation_algorithm_MCGT(m,alphabet)

loglikelihood_on_training_set, running_time = EM_algo.learn(training_set)
print("log: ", loglikelihood_on_training_set, "\n======\nruntime: ", running_time)
output_model = EM_algo.h

output_model.pprint() 