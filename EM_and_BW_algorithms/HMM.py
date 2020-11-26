from tools import resolveRandom

class HMM_state:

	def __init__(self,output_matrix, next_matrix):
		"""
		output_matrix = [[proba_symbol1,proba_symbol2,...],[symbol1,symbol2,...]]
		next_matrix = [[proba_state1,proba_state2,...],[state1,state2,...]]
		"""
		#if sum(output_matrix[0]) < 1.0:
		#	print("Sum of the probabilies of the output_matrix should be 1.0")
		#	print(output_matrix)
		#	return False
		self.output_matrix = output_matrix
		#if sum(next_matrix[0]) < 1.0:
		#	print("Sum of the probabilies of the next_matrix should be 1.0")
		#	print(next_matrix)
		#	return False
		self.next_matrix = next_matrix

	def a(self,state):
		if state in self.next_matrix[1]:
			return self.next_matrix[0][self.next_matrix[1].index(state)]
		else:
			return 0

	def b(self,sigma):
		if sigma in self.output_matrix[1]:
			return self.output_matrix[0][self.output_matrix[1].index(sigma)]
		else:
			return 0

	def generate(self):
		return self.output_matrix[1][resolveRandom(self.output_matrix[0])]

	def next(self):
		return self.next_matrix[1][resolveRandom(self.next_matrix[0])]


class HMM:

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
			output += self.states[current].generate()
			current = self.states[current].next()

		return output

	def pprint(self):
		for i in range(len(self.states)):
			print("----STATE s",i,"----",sep='')
			for j in range(len(self.states)):
				print(" -> s",j," : ",self.states[i].a(j),sep='')
			print("************")
			for j in self.states[i].output_matrix[1]:
				print(" => ",j," : ",self.states[i].b(j),sep='')

	def allStatesPathIterative(self, start, obs_seq):
		"""return all the states path from start that can generate obs_seq"""
		res = []
		for i in range(len(self.states)):
			if self.states[i].b(obs_seq[0]) > 0 and self.states[start].a(i) > 0:
				if len(obs_seq) == 1:
					res.append([start,i])
				else:
					t = self.allStatesPathIterative(i,obs_seq[1:])
					for j in t:
						res.append([start]+j)
		return res

	def allStatesPath(self,obs_seq):
		"""return all the states path that can generate obs_seq"""
		res = []
		if self.states[self.initial_state].b(obs_seq[0]) > 0.0:
			t = self.allStatesPathIterative(self.initial_state,obs_seq[1:])
			for j in t:
				res.append(j)
		return res

	def prob(self,states_path, obs_seq):
		"""return the probability to get this states_path generating this observations sequence"""
		if states_path[0] != self.initial_state:
			return 0
		res = self.states[states_path[0]].b(obs_seq[0])
		for i in range(1,len(states_path)):
			res *= self.states[states_path[i-1]].a(states_path[i])*self.states[states_path[i]].b(obs_seq[i])
		return res
		#but sum_k alpha(seq[:k],states_path[k])*beta(seq[k:],states_path[k]) =? proba(states_path,seq)
