from math import  log
from tools import resolveRandom

class Model_state:
	"""
	Abstract class that represent a state.
	Takes on input the transition matrix of this state

	:param transition_matrix: The transition matrix, depends on the instanciated class.
	:type transition_matrix: list
	"""
	def __init__(self,next_matrix: list) -> None:

		if round(sum(next_matrix[0]),2) != 1.0 and sum(next_matrix[0]) != 0:
			print("Sum of the probabilies of the next_matrix should be 1 or 0 here it's ",sum(next_matrix[0]))
			return False
		self.transition_matrix = next_matrix

	def next(self):
		"""
		overrided
		"""
		pass

	def tau(self,state,obs):
		"""
		overrided
		"""
		pass

	def observations(self):
		"""
		overrided
		"""
		pass

	def pprint(self):
		"""
		overrided
		"""
		pass

	def __str__(self):
		"""
		overrided
		"""
		pass



class Model:
	"""
	Abstract class that represent a model.

	:param states: a list of states
	:type states: list

	:param initial_state: determine which state is the initial one (then it's the id of the state), or what are the probability to start in each state (then it's a list of probabilities) 
	:type initial_state: int or list of float

	:param name: name of the model
	:type name: str
	"""
	def __init__(self,states: list,initial_state,name: str) -> None:
		# initial_state can be a list of probability or an int
		if type(initial_state) == int:
			self.initial_state = [0.0 for i in range(len(states))]
			self.initial_state[initial_state] = 1.0
		else:
			self.initial_state = initial_state
		self.states = states
		self.name = name

	def __str__(self) -> str:
		return self.name

	def save(self,file_path: str) -> None:
		"""
		Save the model into a txt file

		:param file_path: path of the output file
		:type states: str
		"""
		f = open(file_path,'w')
		f.write(self.name)
		f.write('\n')
		f.write(str(self.initial_state))
		f.write('\n')
		for s in self.states:
			f.write(str(s))
		f.close()

	def tau(self,s1: int,s2: int,obs: str) -> float:
		"""
		Return the probability of moving from state s1 to s2 generating observation obs

		:param s1: source state
		:type states: int

		:param s2: destination state
		:type states: int

		:param obs: generated observation
		:type states: str
		"""
		return self.states[s1].tau(s2,obs)

	def observations(self):
		"""
		Return the alphabet of the model

		:return: the alphabet of the model
		:rtype: list of string
		"""
		res = []
		for s in self.states:
			res += s.observations()
		return list(set(res))

	def pi(self,s: int) -> float:
		"""
		Return the probability of starting in state s

		:param s: state s
		:type s: int

		:return: the probability of starting in state s
		:rtype: float
		"""
		return self.initial_state[s]

	def run(self,number_steps: int) -> list:
		"""
		Simulate a run of length number_steps of the model and return the sequence of observations generated.

		:param number_steps: length of the simulation
		:type s: int

		:return: trace generated by the run
		:rtype: list of str
		"""
		output = []
		current = resolveRandom(self.initial_state)

		while len(output) < number_steps:
			[next_state, symbol] = self.states[current].next()
			output.append(symbol)
			current = next_state

		return output

	def pprint(self) -> None:
		"""
		Print the model on terminal.
		"""
		print("Name:",self.name)
		if 1.0 in self.initial_state:
			print("Initial state: s",self.initial_state,sep='')
		else:
			print("Initial state: ",end='')
			for i in range(len(self.states)):
				if self.initial_state[i] != 0.0:
					print('s'+str(i)+':',round(self.initial_state[i],3),end=', ')
			print()
		for i in range(len(self.states)):
			self.states[i].pprint(i)
		print()

	def _updateAlphaMatrix(self,sequence: list,common: int,alpha_matrix: list) -> list:
		"""
		Update the given alpha values for all the states for a new <sequence> of observations. It keeps the alpha values for the <common> first observations of the sequence.
		The idea is the following: if you have already computed the alpha values for a previous sequence and you want to compute the alpha values of a new sequence that starts *
		with the same <common> observations you don't need to compute again the first <common> alpha values for each states.
		If, on the other hand, you have still not compute any alpha values you can simply set common to 0 and give an empty <alpha_matrix> which has the right size. The method initAlphaMatrix
		can generate such matrix.

		:param sequence: the sequence of observations for which we compute the alpha values
		:type s: list of str
		
		:param common: for each state, the first <common> alpha values will be keept unchanged
		:type common: int

		param alpha_matrix: the alpha_matrix to update. Can be generated by the method initAlphaMatrix
		:type common: list of list of float

		:return: the alpha matrix containing all the alpha values for all the states for this sequence: alpha_matrix[s][t] is the probability of being in state s after seing the t-1 first observation of <sequence>
		:rtype: list of list of float
		"""
		for k in range(common,len(sequence)):
			for s in range(len(self.states)):
				summ = 0.0
				for ss in range(len(self.states)):
					p = self.states[ss].tau(s,sequence[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ
		return alpha_matrix

	def _initAlphaMatrix(self,len_seq: int) -> list:
		"""
		Return a matrix with the correct size to be updated by the updateAlphaMatrix method

		:param len_seq: length of the sequence for which we will compute the alpha values
		:type len_seq: int

		:return: a matrix with the correct size to be updated by the updateAlphaMatrix method
		:rtype: list of list of None
		"""
		alpha_matrix = []
		for s in range(len(self.states)):
			alpha_matrix.append([self.initial_state[s]])
			alpha_matrix[-1] += [None for i in range(len_seq)]
		return alpha_matrix

	def logLikelihood_oneseq(self,sequence: list) -> float:
		alpha_matrix = self._initAlphaMatrix(len(sequence))
		alpha_matrix = self._updateAlphaMatrix(sequence,0,alpha_matrix)
		loglikelihood = log(sum([alpha_matrix[s][-1] for s in range(len(self.states))]))
		return loglikelihood


	def logLikelihood(self,sequences: list) -> float:
		"""
		Compute the average loglikelihood of a set of sequences which all have the same length

		:param sequences: set of sequences of observations that all have the same length
		:type sequences: list of list of str
		"""
		#Assumes all sequences have same length
		sequences_sorted = sequences[0][:]
		sequences_sorted.sort()
		loglikelihood = 0.0

		alpha_matrix = self._initAlphaMatrix(len(sequences_sorted[0]))
		for seq in range(len(sequences_sorted)):
			sequence = sequences_sorted[seq]
			times = sequences[1][sequences[0].index(sequence)]
			common = 0
			if seq > 0:
				while sequences_sorted[seq-1][common] == sequence[common]:
					common += 1
			alpha_matrix = self._updateAlphaMatrix(sequence,common,alpha_matrix)

			if sum([alpha_matrix[s][-1] for s in range(len(self.states))]) > 0:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.states))])) * times

		return loglikelihood / sum(sequences[1])
