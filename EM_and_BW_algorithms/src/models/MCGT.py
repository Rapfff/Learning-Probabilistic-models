import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import resolveRandom
from math import  log

class MCGT_state:

	def __init__(self,next_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_symbol,transition2_symbol,...]]
		"""
		if round(sum(next_matrix[0]),2) < 1.0 and sum(next_matrix[0]) != 0:
			print("Sum of the probabilies of the next_matrix should be 1 or 0 here it's ",sum(next_matrix[0]))
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

	def __str__(self):
		if len(self.next_matrix[0]) == 0: #end state
			return "-\n"
		else:
			res = ""
			for proba in self.next_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.next_matrix[1]:
				res += str(state)+' '
			res += '\n'
			for obs in self.next_matrix[2]:
				res += str(obs)+' '
			res += '\n'
			return res

class MCGT:

	def __init__(self,states,initial_state,name="unknown MCGT"):
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

	def g(self,s1,s2,obs):
		return self.states[s1].g(s2,obs)

	def observations(self):
		res = []
		for s in self.states:
			res += s.next_matrix[2]
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
		print(self.name)
		print(self.initial_state)
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			for j in range(len(self.states[i].next_matrix[0])):
				if self.states[i].next_matrix[0][j] > 0.0:
					print("s",i," - (",self.states[i].next_matrix[2][j],") -> s",self.states[i].next_matrix[1][j]," : ",self.states[i].next_matrix[0][j],sep='')
		print()

	def computeAlphaMatrix(self,sequence,common,alpha_matrix):
		for k in range(common,len(sequence)):
			for s in range(len(self.states)):
				summ = 0.0
				for ss in range(len(self.states)):
					p = self.states[ss].g(s,sequence[k])
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
		sequences_sorted = sequences[0]
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

			if sum([alpha_matrix[s][-1] for s in range(len(self.states))]) <= 0:
				return None
			else:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.states))])) * times

		return loglikelihood / sum(sequences[1])

	def probasSequences(self,sequences):
		#given sequences = [seq1,seq2...] /!\ all sequences should be pairwise different
		#return probas   = [prob_seq1,prob_seq2,...]
		sequences_sorted = sequences
		sequences_sorted.sort()
		alpha_matrix = self.initAlphaMatrix(len(sequences_sorted[0]))
		probas = []
		for seq in range(len(sequences_sorted)):
			sequence = sequences_sorted[seq]
			common = 0
			if seq > 0:
				while sequences_sorted[seq-1][common] == sequence[common]:
					common += 1
			alpha_matrix = self.computeAlphaMatrix(sequence,common,alpha_matrix)
			probas.append(sum([alpha_matrix[s][-1] for s in range(len(self.states))]))
		return probas

def KLDivergence(m1,m2,test_set):
	pm1 = m1.probasSequences(test_set)
	tot_m1 = sum(pm1)
	pm2 = m2.probasSequences(test_set)
	res = 0.0
	for seq in range(len(test_set)):
		if pm2[seq] <= 0.0:
			print(test_set[seq])
			return 256
		if tot_m1 > 0.0 and pm1[seq] > 0.0:
			res += (pm1[seq]/tot_m1)*log(pm1[seq]/pm2[seq])
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


def loadMCGT(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = int(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(MCGT_state([[],[],[]]))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = f.readline()[:-2].split(' ')
			states.append(MCGT_state([p,s,o]))

		l = f.readline()

	return MCGT(states,initial_state,name)
