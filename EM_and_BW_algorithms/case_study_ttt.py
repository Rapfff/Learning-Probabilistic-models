from examples.examples_models import modelMDP_random, scheduler_uniform
from examples.tictactoeMDP import TicTacToe
from Estimation_algorithms_MDP_multithreading import Estimation_algorithm_MDP
from datetime import datetime
from tools import randomProbabilities, generateSet
from MDP import MDP_state

nb_states = 15
m = TicTacToe()
observations = m.observations()
actions = m.actions()

def createInitialModel():
	m = modelMDP_random(nb_states, observations[:-3], actions)
	#add states
	for obs in range(3):
		d = {}
		for act in actions:
			d[act] = [ [1.0],[nb_states + obs],[observations[10+obs]] ]
		m.states.append(MDP_state(d))

	for state in range(len(m.states)-3):
		for act in actions:
			m.states[state].next_matrix[act][0]  = randomProbabilities(len(m.states[state].next_matrix[act][0])+3)
			m.states[state].next_matrix[act][1] += [nb_states,nb_states + 1,nb_states + 2]
			m.states[state].next_matrix[act][2] += observations[-3:]

	return m

set_size = 100
sequence_size = 7

s = scheduler_uniform(actions)
training_set = generateSet(m,set_size,sequence_size,s,True)

nb_experiments = 10

for k in range(nb_experiments):
	print(datetime.now(),"iteration:",k)
	m = createInitialModel()
	algo = Estimation_algorithm_MDP(m,observations,actions)
	algo.problem3(training_set,"case_study_ttt/modelEM_"+str(k+1)+".txt",0.0001,k)
