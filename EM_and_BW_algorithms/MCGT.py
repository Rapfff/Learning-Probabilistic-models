from tools import resolveRandom
from itertools import combinations_with_replacement

class MCGT_state:

	def __init__(self,next_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_symbol,transition2_symbol,...]]
		"""
		if round(sum(next_matrix[0]),2) < 1.0:
			print("Sum of the probabilies of the next_matrix should be 1.0 here it's ",sum(next_matrix[0]))
			#return False
		self.next_matrix = next_matrix

	def next(self):
		c = resolveRandom(self.next_matrix[0])
		return [self.next_matrix[1][c],self.next_matrix[2][c]]

	def g(self,state,obs):
		for i in range(len(self.next_matrix[0])):
			if self.next_matrix[1][i] == state and self.next_matrix[2][i] == obs:
				return self.next_matrix[0][i]
		return 0.0

class MCGT:

	def __init__(self,states,initial_state):
		self.initial_state = initial_state
		self.states = states

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0
			
	def run(self,number_steps):
		output = ""
		current = self.initial_state

		while len(output) < number_steps:
			[next_state, symbol] = self.states[current].next()
			output += symbol
			current = next_state

		return output

	def pprint(self):
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			for j in range(len(self.states[i].next_matrix[0])):
				if self.states[i].next_matrix[0][j] > 0.0:
					print("s",i," - (",self.states[i].next_matrix[2][j],") -> s",self.states[i].next_matrix[1][j]," : ",self.states[i].next_matrix[0][j],sep='')
		print()
	
	def allStatesPathIterative(self, start, obs_seq):
		"""return all the states path from start that can generate obs_seq"""
		res = []
		for i in range(len(self.states[start].next_matrix[1])):
			if self.states[start].next_matrix[0][i] > 0 and self.states[start].next_matrix[2][i] == obs_seq[0]:
				if len(obs_seq) == 1:
					res.append([start,self.states[start].next_matrix[1][i]])
				else:
					t = self.allStatesPathIterative(self.states[start].next_matrix[1][i],obs_seq[1:])
					for j in t:
						res.append([start]+j)
		return res
	
	def allStatesPath(self,obs_seq):
		"""return all the states path that can generate obs_seq"""
		res = []
		for j in self.allStatesPathIterative(self.initial_state,obs_seq):
			res.append(j)
		return res

	def allStatesPathLength(self,size):
		"""returns all states path of length <size>+1 """
		combi = combinations_with_replacement([x for x in range(len(self.states))], size)
		res = []
		for i in list(combi)[:-1]:
			res.append([self.initial_state]+list(i))
		return res


	def probabilityStatesObservations(self,states_path, obs_seq):
		"""return the probability to get this states_path generating this observations sequence"""
		if states_path[0] != self.initial_state:
			return 0.0
		else:
			res = 1.0
		for i in range(len(states_path)-1):
			if res == 0.0:
				return 0.0
			res *= self.states[states_path[i]].g(states_path[i+1],obs_seq[i])
		return res
		#but sum_k alpha(seq[:k],states_path[k])*beta(seq[k:],states_path[k]) =? proba(states_path,seq)

	def probabilityObservations(self,obs_seq):
		res = 0
		for p in self.allStatesPath(obs_seq):
			res += self.probabilityStatesObservations(p,obs_seq)
		return res

def HMMtoMCGT(h):
	states_g = []
	for sh in h.states:
		transitions = [[],[],[]]
		for sy in range(len(sh.output_matrix[0])):
			for ne in range(len(sh.next_matrix[0])):
				transitions[0].append(sh.output_matrix[0][sy]*sh.next_matrix[0][ne])
				transitions[1].append(sh.next_matrix[1][ne])
				transitions[2].append(sh.output_matrix[1][sy])
		states_g.append(MCGT_state(transitions))
	return MCGT(states_g,h.initial_state)