from MCGT import *
from math import sqrt, log
from tools import correct_proba
class Alergia:

	def __init__(self,sample,alpha):
		"""
		Given a set of seq of observations return the MCGT learned by ALERGIA
		sample = [[seq1,seq2,...],[val1,val2,...]]
		all seq have same length
		"""
		self.alpha = alpha
		self.alphabet = getAlphabetFromSequences(sample[0])

		N = sum(sample[1])
		n = len(sample[0][0])

		self.states_lbl = [""]
		self.states_counter= [N]
		
		#states_transitions = [
		#						[state1: [proba1,proba2,...],[state1,state2,...],[obs1,obs2,...]]
		#						[state2: [proba1,proba2,...],[state1,state2,...],[obs1,obs2,...]]
		#						...
		#					  ]

		self.states_transitions = []

		#init self.states_lbl and self.states_counter
		for i in range(n):
			for seq in range(len(sample[0])):
				if not sample[0][seq][:i+1] in self.states_lbl:
					self.states_lbl.append(sample[0][seq][:1+i])
					self.states_counter.append(sample[1][seq])
				else:
					self.states_counter[self.states_lbl.index(sample[0][seq][:i+1])] += sample[1][seq]

		#init self.states_transitions
		for s1 in range(len(self.states_lbl)):
			self.states_transitions.append([[],[],[]])
			
			len_s1 = len(self.states_lbl[s1])
			
			s2 = s1 + 1
			while s2 < len(self.states_lbl):
				if len(self.states_lbl[s2]) == len_s1: # too short
					s2 += 1
				elif len(self.states_lbl[s2]) == len_s1 + 2: # too long
					break
				elif self.states_lbl[s2][:-1] != self.states_lbl[s1]: # not same prefix
					s2 += 1
				else: # OK
					self.states_transitions[-1][0].append(self.states_counter[s2])
					self.states_transitions[-1][1].append(s2)
					self.states_transitions[-1][2].append(self.states_lbl[s2][-1])
					s2 += 1

	def learn(self):

		for j in range(1,len(self.states_lbl)):
			if self.states_lbl[j] != None:
				for i in range(j):
					if self.states_lbl[i] != None:
						if self.compatibleMerge(i,j):
							j -= 1
							break

		return self.toMCGT()

	def transitionStateAction(self,state,action):
		try:
			return self.states_transitions[state][1][self.states_transitions[state][2].index(action)]
		except ValueError:
			return None
	
	def different(self,i,j,a):
		ni = self.states_counter[i]
		nj = self.states_counter[j]
		try:
			fi = self.states_transitions[i][0][self.states_transitions[i][2].index(a)]
		except ValueError:
			fi = 0
		try:
			fj = self.states_transitions[j][0][self.states_transitions[j][2].index(a)]
		except ValueError:
			fj = 0
		return ( abs((fi/ni) - (fj/nj)) > sqrt(0.5*log(2/self.alpha))*((1/sqrt(ni)) + (1/sqrt(nj))) )


	def compatibleMerge(self,i,j):
		#input()
		#print(i,j)
		if i == None or j == None:
			return True

		if i == j:
			return True

		for a in self.alphabet:
			#print("looking for action",a)
			if self.different(i,j,a):
				return False
			if not self.compatibleMerge(self.transitionStateAction(i,a),self.transitionStateAction(j,a)):
				return False
		print("Merging",self.states_lbl[i],self.states_lbl[j])
		self.merge(i,j)
		return True

	def merge(self,i,j):
		for state in range(j):
			if self.states_lbl != None:
				for transition in range(len(self.states_transitions[state][1])):
					if self.states_transitions[state][1][transition] == j:
						self.states_transitions[state][1][transition] = i
		
		for a in self.states_transitions[j][2]:
			
			ja = self.states_transitions[j][2].index(a)
			
			if a in self.states_transitions[i][2]:
				self.states_transitions[i][0][self.states_transitions[i][2].index(a)] += self.states_transitions[j][0][ja]
			
			else:
				self.states_transitions[i][0].append(self.states_transitions[j][0][ja])
				self.states_transitions[i][1].append(self.states_transitions[j][1][ja])
				self.states_transitions[i][2].append(a)


		self.states_counter[i] += self.states_counter[j]
		self.states_lbl[j] = None


	def toMCGT(self):
		states = []
		for i in range(len(self.states_transitions)):
			if self.states_lbl[i] != None:
				self.states_transitions[i][0] = [j/self.states_counter[i] for j in self.states_transitions[i][0]]
				self.states_transitions[i][0] = correct_proba(self.states_transitions[i][0])
				self.states_transitions[i][1] = [j-self.states_lbl[:j].count(None) for j in self.states_transitions[i][1]]
				states.append(MCGT_state(self.states_transitions[i]))
		return MCGT(states,0)



def getAlphabetFromSequences(sequences):
	if type(sequences) == str:
		return list(set(sequences))
	else:
		seq = ""
		sequences = list(set(sequences))
		for i in sequences:
			seq += i
		return list(set(seq))