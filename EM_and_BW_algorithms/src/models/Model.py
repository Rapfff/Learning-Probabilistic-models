from math import  log

class Model_state:

	def __init__(self,next_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_symbol,transition2_symbol,...]]
		"""
		if round(sum(next_matrix[0]),2) != 1.0 and sum(next_matrix[0]) != 0:
			print("Sum of the probabilies of the next_matrix should be 1 or 0 here it's ",sum(next_matrix[0]))
			#return False
		self.next_matrix = next_matrix

	def next(self):
		#overrided
		pass

	def tau(self,state,obs):
		#overrided
		pass

	def observations(self):
		#overrided
		pass

	def pprint(self):
		#overrided
		pass

	def __str__(self):
		#overrided
		pass



class Model:

	def __init__(self,states,initial_state,name):
		self.initial_state = initial_state
		self.states = states
		self.name = name

	def __str__(self):
		return self.name

	def save(self,file_path):
		f = open(file_path,'w')
		f.write(self.name)
		f.write('\n')
		f.write(str(self.initial_state))
		f.write('\n')
		for s in self.states:
			f.write(str(s))
		f.close()

	def tau(self,s1,s2,obs):
		return self.states[s1].tau(s2,obs)

	def observations(self):
		res = []
		for s in self.states:
			res += s.observations()
		return list(set(res))

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0
			
	def run(self,number_steps):
		output = []
		current = self.initial_state

		while len(output) < number_steps:
			[next_state, symbol] = self.states[current].next()
			output.append(symbol)
			current = next_state

		return output

	def pprint(self):
		print("Name:",self.name)
		print("Initial state: s",self.initial_state,sep='')
		for i in range(len(self.states)):
			self.states[i].pprint(i)
		print()

	def computeAlphaMatrix(self,sequence,common,alpha_matrix):
		for k in range(common,len(sequence)):
			for s in range(len(self.states)):
				summ = 0.0
				for ss in range(len(self.states)):
					p = self.states[ss].tau(s,sequence[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ
		return alpha_matrix

	def initAlphaMatrix(self,len_seq):
		alpha_matrix = []
		for s in range(len(self.states)):
			if s == self.initial_state:
				alpha_matrix.append([1.0])
			else:
				alpha_matrix.append([0.0])
			alpha_matrix[-1] += [None for i in range(len_seq)]
		return alpha_matrix

	def logLikelihood(self,sequences):
		"""Assumes all sequences have same length"""
		sequences_sorted = sequences[0][:]
		sequences_sorted.sort()
		loglikelihood = 0.0

		alpha_matrix = self.initAlphaMatrix(len(sequences_sorted[0]))
		for seq in range(len(sequences_sorted)):
			sequence = sequences_sorted[seq]
			times = sequences[1][sequences[0].index(sequence)]
			common = 0
			if seq > 0:
				while sequences_sorted[seq-1][common] == sequence[common]:
					common += 1
			alpha_matrix = self.computeAlphaMatrix(sequence,common,alpha_matrix)

			if sum([alpha_matrix[s][-1] for s in range(len(self.states))]) > 0:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.states))])) * times

		return loglikelihood / sum(sequences[1])
