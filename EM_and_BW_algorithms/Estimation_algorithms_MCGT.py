from MCGT import *
from tools import correct_proba
from time import time
from math import log

class Estimation_algorithms_MCGT:
	def __init__(self,h,alphabet):
		"""
		h is a MCGT
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.alphabet = alphabet

	def checkEnd(self):
		for i in range(len(self.h.states)):
			for j in range(len(self.h.states)):
				for k in self.alphabet:
					if self.h.states[i].g(j,k) != self.hhat.states[i].g(j,k):
						return False
		return True

	def logLikelihood(self):
		try:
			return log(self.hhat.probabilityObservations(self.sequence))
		except AttributeError:
			res = 0
			for i in range(len(self.sequences[0])):
				res += log(self.hhat.probabilityObservations(self.sequences[0][i])) * self.sequences[1][i]
			return res / sum(self.sequences[1])

	def precomputeMatrices(self):
		"""Here we compute all the values alpha(k,t) and beta(t,k)"""
		self.alpha_matrix = []

		for s in range(len(self.h.states)):
			if s == self.h.initial_state:
				self.alpha_matrix.append([1.0])
			else:
				self.alpha_matrix.append([0.0])
		
		for k in range(len(self.sequence)):
			for s in range(len(self.h.states)):
				summ = 0
				for ss in range(len(self.h.states)):
					p = self.h.states[ss].g(s,self.sequence[k])
					if p > 0:
						summ += self.alpha_matrix[ss][k]*p
				self.alpha_matrix[s].append(summ)


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

	def problem1(self,sequence):
		"""
		return the probability that the MCGT returns the given sequence of observations
		"""
		return self.beta_matrix[self.h.initial_state][0]

	#def problem2(self,sequence)

	def problem3(self,sequence):
		"""
		Given a sequence of observations it adapts the parameters of h in order to maximize the probability to get 
		this sequence of observations.
		"""
		start_time = time()
		self.sequence = sequence
		while True:
			#print("probability:",round(self.problem1(sequence),4))
			self.precomputeMatrices()
			new_states = []
			for i in range(len(self.h.states)):
				next_probas = []
				next_states = []
				next_obs    = []
				for j in range(len(self.h.states)):
					for k in list(set(sequence)): #et si deux fois la meme lettre dans sequence
						next_probas.append(self.ghat(i,j,k))
						next_states.append(j)
						next_obs.append(k)
					for k in [ letter for letter in self.alphabet if not letter in sequence]:
						next_probas.append(0)
						next_states.append(j)
						next_obs.append(k)

				next_probas = correct_proba(next_probas)
				new_states.append(MCGT_state([ next_probas, next_states, next_obs]))
			self.hhat = MCGT(new_states,self.h.initial_state)
			currentloglikelihood = self.logLikelihood()
			print("loglikelihood :",currentloglikelihood)
			if self.checkEnd():
				break
			else:
				self.h = self.hhat
		self.h.pprint()
		#print("probability:",round(self.problem1(sequence),4))
		print("Duration:",time()-start_time)
		return self.h



class BW_ON_MCGT(Estimation_algorithms_MCGT):
	"""implementation of the Baun-Welch algorithm on Hidden Markov Model"""

	def __init__(self,h,alphabet):
		"""
		h is a MCGT
		alphabet is a list of the possible observations (list of strings)
		"""
		super().__init__(h,alphabet)

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

	def ghat(self,s1,s2,obs):
		num = 0
		den = 0
		self.computeK()

		for k in range(len(self.sequence)):
			den += self.gamma(s1,k)
			if self.sequence[k] == obs:
				num += self.xi(s1,s2,k) #this is equal to xi_k(s1,s2)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return num/den


class EM_ON_MCGT(Estimation_algorithms_MCGT):
	"""implementation of the Baun-Welch algorithm on Hidden Markov Model"""
	def __init__(self,h,alphabet):
		"""
		h is a MCGT
		alphabet is a list of the possible observations (list of strings)
		"""
		super().__init__(h,alphabet)

	def ghat(self,t,u,o):
		"""Here we use the TUKmatrix"""
		num = 0
		den = 0
		for k in range(0,len(self.sequence)):
			den += self.alpha_matrix[t][k]*self.beta_matrix[t][k]
			if o == self.sequence[k]:
				num += self.alpha_matrix[t][k]*self.h.states[t].g(u,self.sequence[k])*self.beta_matrix[u][k+1]
		if den == 0:
			return 0.0
		return num/den