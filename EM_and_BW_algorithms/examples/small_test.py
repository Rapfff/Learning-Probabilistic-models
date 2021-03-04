from examples.examples_models import modelMCGT_random
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT

nb_states = 3
seq_len = 2
alphabet = ['1','2','3','4','5','6']
# training_set = [[sequence1,sequence2,...],[number_of_sequence1,number_of_sequence2,...]]
# Example:
training_file = open("datasets/dice1_trace", "r")
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

loglikelihood_on_training_set, running_time = EM_algo.problem3(training_set)
print("log: ", loglikelihood_on_training_set, "\n======\nruntime: ", running_time)
output_model = EM_algo.h

output_model.pprint()