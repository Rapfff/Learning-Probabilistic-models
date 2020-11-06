from MDP import *
from tools import correct_proba
from random import randint

def scheduler_random(path, current):
	actions = current.actions()
	return actions[randint(0,len(actions)-1)]

# a MDP is fully observable if the obs in each state is unique (for all s,s' s.t. s != s' then s.obs != s'.obs )
class Estimation_algorithm_fullyobservable_MDP:
	def __init__(self):
		None

	def learnFromSequences(self,sequences):
		actions_id = []
		for seq in sequences:
			for j in seq[1]:
				if not j in actions_id:
					actions_id.append(j)
		
		states_id = []
		for seq in sequences:
			for j in seq[0]:
				if not j in states_id:
					states_id.append(j)

		count_matrix = []
		for i in range(len(states_id)):
			count_matrix.append([])
			for j in range(len(actions_id)):
				count_matrix[-1].append([])
				for k in range(len(states_id)):
					count_matrix[-1][-1].append(0)

		for seq in sequences:
			for i in range(len(seq[1])):
				count_matrix[states_id.index(seq[0][i])][actions_id.index(seq[1][i])][states_id.index(seq[0][i+1])] += 1

		states = []
		for s1 in range(len(states_id)):
			dic_state = {}
			for a in range(len(actions_id)):
				den = sum(count_matrix[s1][a])
				if den > 0:
					list_action = [[],[]]
					for s2 in range(len(states_id)):
						num = count_matrix[s1][a][s2]
						if num > 0:
							list_action[0].append(num/den)
							list_action[1].append(s2)
					list_action[0] = correct_proba(list_action[0],2)
					dic_state[actions_id[a]] = list_action
			states.append(MDP_state(dic_state, states_id[s1]))

		return MDP(states,states_id.index(sequences[0][0][0]))

	def learnFromBlackBox(self,black_box,l,length_exp):
		sequences = []
		for i in range(l):
			sequences.append(black_box.run(length_exp, scheduler_random))
		return self.learnFromSequences(sequences)