from MCGT import *
from tools import correct_proba
from time import time
from math import log

#Le proleme est dans precomputematrices_multiple: on obtient des valeurs trop petites => il les mets
#a 0. Il faut trouver une solution (normaliser le tout ??)


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

	def problem3multiple(self,sequences):
		"""
		Given sequences of observations it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		sequences = [[sequence1,sequence2,...],[number_of_seq1,number_of_seq2,...]]
		"""
		prevloglikelihood = self.h.logLikelihood(sequences)
		start_time = time()
		self.sequences = sequences
		sequences = ""
		for i in self.sequences[0]:
			sequences += i
		sequences = list(set(sequences))
		while True:
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
			
			currentloglikelihood = self.hhat.logLikelihood(self.sequences)
			print("loglikelihood :",currentloglikelihood)
			if self.checkEnd() or prevloglikelihood == currentloglikelihood:#or time() - start_time > 120
				self.h = self.hhat
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat
				#self.h.pprint()
		self.h.pprint()
		#print("\nSeq/prob/freq")
		#total = sum(self.sequences[1])
		#for i in range(len(self.sequences[0])):
			#print(self.sequences[0][i],round(self.problem1(self.sequences[0][i]),3),self.sequences[1][i]/total)
		#print("\nAccuracy:",self.accuracy())
		#print("Duration:",time()-start_time)
		return [self.h.logLikelihood(self.sequences),time()-start_time]

	
	def precomputeMatrices(self,sequence):
		"""Here we compute all the values alpha(k,t) and beta(t,k) for a given sequence"""
		self.alpha_matrix = []

		for s in range(len(self.h.states)):
			if s == self.h.initial_state:
				self.alpha_matrix.append([1.0])
			else:
				self.alpha_matrix.append([0.0])
		
		for k in range(len(sequence)):
			for s in range(len(self.h.states)):
				summ = 0
				for ss in range(len(self.h.states)):
					p = self.h.states[ss].g(s,sequence[k])
					summ += self.alpha_matrix[ss][k]*p
				self.alpha_matrix[s].append(summ)

		self.beta_matrix = []

		for s in range(len(self.h.states)):
			self.beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			for s in range(len(self.h.states)):
				summ = 0
				for ss in range(len(self.h.states)):
					p = self.h.states[s].g(ss,sequence[k])
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
		for seq in range(len(self.sequences[0])):
			self.sequence = self.sequences[0][seq]
			times = self.sequences[1][seq]
			self.precomputeMatrices(self.sequence)
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
