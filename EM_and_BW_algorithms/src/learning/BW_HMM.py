import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.HMM import *
from time import time
from tools import correct_proba
from math import log

class Estimation_algorithms_HMM:
	def __init__(self,h,alphabet):
		"""
		h is a HMM
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.alphabet = alphabet

	def accuracy(self):
		expectations = []
		s = 0
		total = sum(self.sequences[1])
		for i in self.sequences[0]:
			expectations.append(self.problem1(i))
		for i in range(len(self.sequences[0])):
			s += min(expectations[i],self.sequences[1][i]/total)
		return s

	def checkEnd(self):
		for i in range(len(self.h.states)):
			for j in self.alphabet:
				if self.h.states[i].b(j) != self.hhat.states[i].b(j):
					return False
			for j in range(len(self.h.states)):
				if self.h.states[i].a(j) != self.hhat.states[i].a(j):
					return False
		return True

	def alpha(self,s,k,o): # /!\ k = {0,1,...,T-1}
		"""
		Forward variable. s is the index of the state in h.states, o a seq of obs and k an index in o.
		Return the probability to be in state s after generating the k+1 first obs.
		"""
		if k == 0:
			return self.h.pi(s)*self.h.states[s].b(o[k])
		else:
			summ = 0
			for ss in range(len(self.h.states)):
				p = self.h.states[ss].a(s)
				if p > 0:
					summ += self.alpha(ss,k-1,o)*p
			
			return summ*self.h.states[s].b(o[k])

	def beta(self,s,k,o): # /!\ k = {0,1,...,T-1}
		"""
		Backward variable. s is the index of the state in h.states, o a seq of obs and k an index in o.
		Return the probability to generate the T-k last obs of o starting in state s.
		"""
		if k == len(o)-1:
			return 1
		
		else:
			summ = 0
			for ss in range(len(self.h.states)):
				p = self.h.states[s].a(ss)*self.h.states[ss].b(o[k+1])
				if p > 0:
					summ += self.beta(ss,k+1,o)*p
			return summ

	def problem1(self,sequence):
		"""
		return the probability that the HMM returns the given sequence of observations
		"""
		summ = 0
		for i in range(len(self.h.states)):
			summ += self.alpha(i,len(sequence)-1,sequence)
		return summ

	def problem2(self,sequence):
		"""
		Given a sequence of at least 2 observations it returns a list [p,s] where:
			- s is a sequence of states in h
			- p is the probability that the sequence s of states is used to generate the given sequence of obs
		"""
		delta = [[]]
		psi = [[]]
		for j in range(len(self.h.states)):
			psi[0].append(0)
			delta[0].append(self.h.pi(j)*self.h.states[j].b(sequence[0]))

		for i in range(1,len(sequence)):
			delta.append([])
			psi.append([])

			for j in range(len(self.h.states)):
				val = delta[i-1][0]*self.h.states[0].a(j)
				ind = 0
				for k in range(1,len(self.h.states)):
					if val < delta[i-1][k]*self.h.states[k].a(j):
						val = delta[i-1][k]*self.h.states[k].a(j)
						ind = k
				psi[i].append(ind)
				delta[i].append(val*self.h.states[j].b(sequence[i])) 
		
		p = delta[len(sequence)-1][0]
		q = 0
		for i in range(1,len(self.h.states)):
			if p < delta[len(sequence)-1][i]:
				p = delta[len(sequence)-1][i]
				q = i

		res = [q]
		for i in range(len(sequence)-1,0,-1):
			res.append(psi[i][res[-1]])
		res.reverse()
		return [p,res]



	def problem3single(self,sequence):
		"""
		Given a sequence of observations it adapts the parameters of h in order to maximize the probability to get 
		this sequence of observations.
		"""
		self.sequence = sequence
		while True:
			new_states = []
			for i in range(len(self.h.states)):
				output_probas = correct_proba([self.bhat(i,self.alphabet[j]) for j in range(len(self.alphabet))])
				next_probas   = correct_proba([self.ahat(i,j) for j in range(len(self.h.states))])
				new_states.append(HMM_state([ output_probas,self.alphabet],[next_probas, [j for j in range(len(self.h.states))]]))
			self.hhat = HMM(new_states,self.h.initial_state)
			#self.hhat.pprint()
			#print()
			if self.checkEnd():
				break
			else:
				self.h = self.hhat
		self.h.pprint()
		print("probability:",round(self.problem1(sequence),4))

	def learn(self,sequences):
		"""
		Given sequences of observations it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		sequences = [[sequence1,sequence2,...],[number_of_seq1,number_of_seq2,...]]
		"""
		start_time = time()
		self.sequences = sequences
		f = True
		while f:
			new_states = []
			for i in range(len(self.h.states)):
				output_probas = correct_proba([self.bhatmultiple(i,self.alphabet[j]) for j in range(len(self.alphabet))])
				next_probas   = correct_proba([self.ahatmultiple(i,j) for j in range(len(self.h.states))])
				new_states.append(HMM_state([ output_probas,self.alphabet],[next_probas, [j for j in range(len(self.h.states))]]))
			self.hhat = HMM(new_states,self.h.initial_state)
			
			if self.checkEnd():
				self.h = self.hhat
				break
			else:
				self.h = self.hhat
				f = True
		self.h.pprint()

		loglikelihood = 0
		for i in range(len(sequences[0])):
			p = self.problem1(sequences[0][i])
			if p == 0:
				loglikelihood = -256 * sum(sequences[1])
				break
			loglikelihood += log(p) * sequences[1][i]
		loglikelihood /= sum(sequences[1])
		running_time = time()-start_time
		return self.h



class BW_ON_HMM(Estimation_algorithms_HMM):
	"""implementation of the Baun-Welch algorithm on Hidden Markov Model"""

	def __init__(self,h,alphabet):
		"""
		h is a HMM
		alphabet is a list of the possible observations (list of strings)
		"""
		super().__init__(h,alphabet)

	def xi(self,s1,s2,k,o):
		"""
		Returns the probability to generate the kth obs of o in state s1 and then move to s2.
		"""
		num = self.alpha(s1,k,o)*self.h.states[s1].a(s2)*self.h.states[s2].b(o[k+1])*self.beta(s2,k+1,o)
		den = 0
		for t in range(len(self.h.states)):
			for u in range(len(self.h.states)):
				p = self.h.states[t].a(u)*self.h.states[u].b(o[k+1])
				if p > 0:
					den += self.alpha(t,k,o)*self.beta(u,k+1,o)*p
		
		return round(num/den,2)

	def gamma(self,s,k,o):
		"""
		Returns the probability to generate the kth obs of o in state s.
		"""
		num = self.alpha(s,k,o)*self.beta(s,k,o)
		den = 0
		for ss in range(len(self.h.states)):
			den += self.alpha(ss,k,o)*self.beta(ss,k,o)
		return num/den

	def ahat(self,s1,s2):
		num = 0
		den = 0
		for k in range(len(self.sequence)-1):
			num += self.xi(s1,s2,k,self.sequence)
			den += self.gamma(s1,k,self.sequence)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)

	def bhat(self,s1,o):
		num = 0
		den = 0
		for k in range(len(self.sequence)):
			if self.sequence[k] == o:
				num += self.gamma(s1,k,self.sequence)
			den += self.gamma(s1,k,self.sequence)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)


	def ahatmultiple(self,s1,s2):
		num = 0
		den = 0
		for i in range(len(self.sequences[0])):
			for k in range(len(self.sequences[0][i])-1):
				num += self.xi(s1,s2,k,self.sequences[0][i])*self.sequences[1][i]
				den += self.gamma(s1,k,self.sequences[0][i])*self.sequences[1][i]
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)

	def bhatmultiple(self,s1,o):
		num = 0
		den = 0
		for i in range(len(self.sequences[0])):
			for k in range(len(self.sequences[0][i])):
				if self.sequences[0][i][k] == o:
					num += self.gamma(s1,k,self.sequences[0][i])*self.sequences[1][i]
				den += self.gamma(s1,k,self.sequences[0][i])*self.sequences[1][i]
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)


class EM_ON_HMM(Estimation_algorithms_HMM):
	"""implementation of the Baun-Welch algorithm on Hidden Markov Model"""
	def __init__(self,h,alphabet):
		"""
		h is a HMM
		alphabet is a list of the possible observations (list of strings)
		"""
		super().__init__(h,alphabet)

	def ahat(self,s1,s2):
		num = 0
		den = 0
		for k in range(len(self.sequence)-1):
			num += self.alpha(s1,k,self.sequence)*self.h.states[s1].a(s2)*self.h.states[s2].b(self.sequence[k+1])*self.beta(s2,k+1,self.sequence)
			den += self.alpha(s1,k,self.sequence)*self.beta(s1,k,self.sequence)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)

	def bhat(self,s1,o):
		num = 0
		den = 0
		for k in range(len(self.sequence)):
			if self.sequence[k] == o:
				num += self.alpha(s1,k,self.sequence)*self.beta(s1,k,self.sequence)
			den += self.alpha(s1,k,self.sequence)*self.beta(s1,k,self.sequence)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)


	def ahatmultiple(self,s1,s2):
		num = 0
		den = 0
		for i in range(len(self.sequences[0])):
			for k in range(len(self.sequences[0][i])-1):
				num += self.alpha(s1,k,self.sequences[0][i])*self.h.states[s1].a(s2)*self.h.states[s2].b(self.sequences[0][i][k+1])*self.beta(s2,k+1,self.sequences[0][i])*self.sequences[1][i]
				den += self.alpha(s1,k,self.sequences[0][i])*self.beta(s1,k,self.sequences[0][i])*self.sequences[1][i]
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)

	def bhatmultiple(self,s1,o):
		num = 0
		den = 0
		for i in range(len(self.sequences[0])):
			for k in range(len(self.sequences[0][i])):
				if self.sequences[0][i][k] == o:
					num += self.alpha(s1,k,self.sequences[0][i])*self.beta(s1,k,self.sequences[0][i])*self.sequences[1][i]
				den += self.alpha(s1,k,self.sequences[0][i])*self.beta(s1,k,self.sequences[0][i])*self.sequences[1][i]
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)


class EMprime_ON_HMM(Estimation_algorithms_HMM):
	"""implementation of the Baun-Welch algorithm on Hidden Markov Model"""
	def __init__(self,h,alphabet):
		"""
		h is a HMM
		alphabet is a list of the possible observations (list of strings)
		"""
		super().__init__(h,alphabet)

	def ahat(self,s1,s2):
		num = 0
		den = 0
		for k in range(len(self.sequence)-1):
			num += self.alpha(s1,k,self.sequence)*self.h.states[s1].a(s2)*self.h.states[s2].b(self.sequence[k+1])*self.beta(s2,k+1,self.sequence)
			den += self.alpha(s1,k,self.sequence)*self.beta(s1,k,self.sequence)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)

	def bhat(self,s1,o):
		num = 0
		den = 0
		for k in range(len(self.sequence)):
			if self.sequence[k] == o:
				num += self.alpha(s1,k,self.sequence)*self.beta(s1,k,self.sequence)
			den += self.alpha(s1,k,self.sequence)*self.beta(s1,k,self.sequence)
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)


	def ahatmultiple(self,s1,s2):
		num = 0
		den = 0
		for seq in range(len(self.sequences[0])):
			for path in self.h.allStatesPath(self.sequences[0][seq]):
				t1 = 0
				t2 = 0
				for i in range(len(path)-1):
					if path[i] == s1:
						t2 += 1
						if path[i+1] == s2 : 
							t1 += 1
				t3 = self.h.prob(path,self.sequences[0][seq])*self.sequences[1][seq]
				num += t3*t1
				den += t3*t2
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)

	def bhatmultiple(self,s1,o):
		num = 0
		den = 0
		for seq in range(len(self.sequences[0])):
			for path in self.h.allStatesPath(self.sequences[0][seq]):
				t1 = 0
				t2 = 0
				for i in range(len(path)):
					if path[i] == s1:
						t2 += 1
						if self.sequences[0][i] == o : 
							t1 += 1
				t3 = self.h.prob(path,self.sequences[0][seq])*self.sequences[1][seq]
				num += t3*t1
				den += t3*t2
		if den == 0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			return 0.0
		return round(num/den,2)
