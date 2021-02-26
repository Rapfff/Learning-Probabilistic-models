from examples.examples_models import modelMCGT4, modelMCGT_random
from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from tools import generateSet
from MCGT import *

# we suppose we know the observation space, otherwise we can simply extract it from 
# the training_set
alphabet = ['a','b','c','d','e','x','y']
size_alphabet = 7

nb_seq = 1000
length_seq = 5

m = modelMCGT4()
training_set = generateSet(m,nb_seq,length_seq)
test_set = generateSet(m,nb_seq*10,length_seq)


#####################################################
matrix = [[0 for j in range(size_alphabet)] for i in range(size_alphabet+1) ]

for seq in range(len(training_set[0])):
	matrix[0][alphabet.index(training_set[0][seq][0])] += training_set[1][seq]
	for k in range(1,length_seq):
		matrix[alphabet.index(training_set[0][seq][k-1])+1][alphabet.index(training_set[0][seq][k])] += training_set[1][seq]

states = []

for s in matrix:
	states.append(MCGT_state([[p/sum(s) for p in s], list(range(1,size_alphabet+1)) , alphabet ]))

m1 = MCGT(states,0,"initial_model")
#####################################################


print(m1.logLikelihood(training_set))
print()
m2 = modelMCGT_random(size_alphabet+1,alphabet)
print(m2.logLikelihood(training_set))
print()

print("Learning on m1...")
algo = Estimation_algorithm_MCGT(m1,alphabet)
res1 = algo.problem3(training_set)
res1.append(algo.h.logLikelihood(test_set))
print()
print("Learning on m2...")
algo = Estimation_algorithm_MCGT(m2,alphabet)
res2 = algo.problem3(training_set)
res2.append(algo.h.logLikelihood(test_set))
print()

print("With m1:")
print("Running time:                 ",res1[1])
print("loglikelihood on training set:",res1[0])
print("loglikelihood on test set:",res1[2])
print()
print("With m2:")
print("Running time:                 ",res2[1])
print("loglikelihood on training set:",res2[0])
print("loglikelihood on test set:",res2[2])
