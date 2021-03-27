from MCGT import *
from tools import correct_proba
from time import time
import datetime
from math import log


class Estimation_algorithm_MCGT:
	def __init__(self,h,alphabet):
		"""
		h is a MCGT
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.hhat = h
		self.alphabet = alphabet

	def checkEnd(self):
		for i in range(len(self.h.states)):
			for j in range(len(self.h.states)):
				for k in self.alphabet:
					if self.h.states[i].g(j,k) != self.hhat.states[i].g(j,k):
						return False
		return True

	def problem1(self,sequence):
		"""
		return the probability that the MCGT returns the given sequence of observations
		"""
		return self.beta_matrix[self.h.initial_state][0]

	def problem3(self,sequences,pp=''):
		"""
		Given sequences of observations it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		sequences = [[sequence1,sequence2,...],[number_of_seq1,number_of_seq2,...]]
		"""
		c = 0
		self.betas0values = [None for i in range(len(sequences[0]))]
		prevloglikelihood = -256 #it contains the loglikelihood of the previous h
		start_time = time()
		self.sequences = sequences
		sequences = []
		for i in self.sequences[0]:
			for j in i:
				if not j in sequences:
					sequences.append(j)

		while True:
			c += 1
			print(datetime.datetime.now(),pp,c, prevloglikelihood)
			new_states = []
			for i in range(len(self.h.states)):
				next_probas = []
				next_states = []
				next_obs    = []

				for j in range(len(self.h.states)):
					for k in list(set(sequences)):
						next_probas.append(self.ghatmultiple(i,j,k))
						next_states.append(j)
						next_obs.append(k)
					for k in [ letter for letter in self.alphabet if not letter in list(set(sequences)) ]:
						next_probas.append(0)
						next_states.append(j)
						next_obs.append(k)

				next_probas = correct_proba(next_probas)
				new_states.append(MCGT_state([ next_probas, next_states, next_obs]))
			self.hhat = MCGT(new_states,self.h.initial_state)
			
			currentloglikelihood = sum([ log(i) for i in self.betas0values]) # it contains the loglikelihood of h 
			if abs(prevloglikelihood -currentloglikelihood)<0.0001:#or self.checkEnd() #or time() - start_time > 120
				self.h = self.hhat
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat
		#self.h.pprint()
		return [currentloglikelihood,time()-start_time]

	def precomputeMatrices(self, common):
		"""Here we compute all the values alpha(k,t) and beta(t,k) for a given self.sequence"""
		if common == -1:
			common = 0
			self.alpha_matrix = []

			for s in range(len(self.h.states)):
				if s == self.h.initial_state:
					self.alpha_matrix.append([1.0])
				else:
					self.alpha_matrix.append([0.0])
				self.alpha_matrix[-1] += [None for i in range(len(self.sequence))]
		
		for k in range(common,len(self.sequence)):
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					p = self.h.states[ss].g(s,self.sequence[k])
					summ += self.alpha_matrix[ss][k]*p
				self.alpha_matrix[s][k+1] = summ

		self.beta_matrix = []

		for s in range(len(self.h.states)):
			self.beta_matrix.append([1.0])
		
		for k in range(len(self.sequence)-1,-1,-1):
			for s in range(len(self.h.states)):
				summ = 0
				for ss in range(len(self.h.states)):
					p = self.h.states[s].g(ss,self.sequence[k])
					if p > 0:
						summ += self.beta_matrix[ss][1 if ss<s else 0]*p
				self.beta_matrix[s].insert(0,summ)


	def gamma(self,s,k):
		"""
		Returns the probability to generate the kth obs of o in state s * bigK
		Note: it's important to compute self.bigK before
		"""
		return self.alpha_matrix[s][k]*self.beta_matrix[s][k]/self.bigK

	def xi(self,s1,s2,k):
		"""
		Returns the probabilty to move from state s1 to state s2 at time step k
		Note: it's important to compute self.bigK and the TUKmatrix before
		"""
		return self.alpha_matrix[s1][k]*self.h.states[s1].g(s2,self.sequence[k])*self.beta_matrix[s2][k+1]/self.bigK

	def computeK(self):
		"""
		Returns sum over ss of alpha(k-1,ss) * beta(ss,k)
		"""
		self.bigK = self.beta_matrix[self.h.initial_state][0]

	def ghatmultiple(self,s1,s2,obs):
		num = 0
		den = 0
		sequences_sorted = self.sequences[0]
		sequences_sorted.sort()

		for seq in range(len(sequences_sorted)):
			self.sequence = sequences_sorted[seq]
			times = self.sequences[1][self.sequences[0].index(self.sequence)]
			if seq == 0:
				self.precomputeMatrices(-1)
			else:
				common = 0 
				while sequences_sorted[seq-1][common] == self.sequence[common]:
					common += 1
				self.precomputeMatrices(common)

			self.betas0values[seq] = sum([ self.beta_matrix[s][0] for s in range(len(self.h.states)) ])

			self.computeK()
			for k in range(len(self.sequence)):
				den += self.gamma(s1,k) * times
				if self.sequence[k] == obs:
					num += self.xi(s1,s2,k) * times
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return num/den
