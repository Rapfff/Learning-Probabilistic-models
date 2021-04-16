from MDP import *
from tools import correct_proba
from random import randint
from time import time

class Estimation_algorithm_MDP:
	def __init__(self,h,alphabet,actions):
		"""
		h is a MDP
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.hhat = h
		self.alphabet = alphabet
		self.actions = actions

	def checkEnd(self):
		for i in range(len(self.h.states)):
			for j in range(len(self.h.states)):
				for act in self.actions:
					for k in self.alphabet:
						if self.h_g(i,act,j,k) != self.hhat_g(i,act,j,k):
							return False
		return True

	def checkEndLikelihood(self,c):
		if c < 10:
			self.prevloglikelihood = -256
			print(c)
			return False
		print(c,"- loglikelihood",end=" ")
		currentloglikelihood = self.logLikelihood()
		print(currentloglikelihood)
		if self.prevloglikelihood == currentloglikelihood:
			self.prevloglikelihood = currentloglikelihood
			return True
		else:
			self.prevloglikelihood = currentloglikelihood
			return False

	def h_g(self,s1,act,s2,obs):
		return self.h.g(s1,act,s2,obs)

	def hhat_g(self,s1,act,s2,obs):
		return self.hhat.g(s1,act,s2,obs)
	
	def precomputeMatrices(self,sequence,common):
		"""Here we compute all the values alpha(k,t) and beta(t,k) for a given sequence"""
		if common == -1:
			common = 0
			self.alpha_matrix = []

			for s in range(len(self.h.states)):
				if s == self.h.initial_state:
					self.alpha_matrix.append([1.0])
				else:
					self.alpha_matrix.append([0.0])
				self.alpha_matrix[-1] += [None for i in range(len(sequence))]

		for k in range(common,len(sequence)):
			action = self.sequence_actions[k]
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					p = self.h_g(ss,action,s,sequence[k])
					summ += self.alpha_matrix[ss][k]*p
				self.alpha_matrix[s][k+1] = summ

		self.beta_matrix = []

		for s in range(len(self.h.states)):
			self.beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			action = self.sequence_actions[k]
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					p = self.h_g(s,action,ss,sequence[k])
					summ += self.beta_matrix[ss][1 if ss<s else 0]*p
				self.beta_matrix[s].insert(0,summ)
	

	def problem3(self,traces,output_file="output_model.txt",limit=0.00001,pp=''):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [action,obs1,action2,obs2,...,actionx,obsx]
		"""
		self.sequences_sorted = traces[0]
		self.sequences_sorted.sort()
		self.times = [ traces[1][traces[0].index(self.sequences_sorted[seq])] for seq in range(len(self.sequences_sorted))]

		start_time = time()

		self.observations = []
		for seq in self.sequences_sorted:
			for i in range(1,len(seq),2):
				if not seq[i] in self.observations:
					self.observations.append(seq[i])

		counter = 0
		prevloglikelihood = self.logLikelihood()
		while True:
			print(counter,prevloglikelihood)
			new_states = []
			for i in range(len(self.h.states)):
				next_probas = []
				next_states = []
				next_obs    = []

				for j in range(len(self.h.states)):
					for k in self.observations:
						next_probas.append(self.ghatmultiple(i,j,k))
						next_states.append(j)
						next_obs.append(k)
						
					for k in [ letter for letter in self.alphabet if not letter in self.observations ]:
						next_probas.append([0.0]*len(self.actions))
						next_states.append(j)
						next_obs.append(k)

				dic = {}
				for act in range(len(self.actions)):
					probas = [k[act] for k in next_probas]
					probas = correct_proba(probas)
					dic[self.actions[act]] = [probas, next_states, next_obs]
				new_states.append(MDP_state(dic))

			self.hhat = MDP(new_states,self.h.initial_state)
			
			counter += 1
			currentloglikelihood = self.logLikelihood()
			if abs(prevloglikelihood - currentloglikelihood) < limit:
				self.h = self.hhat
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat
		print()
		self.h.pprint()
		self.h.save(output_file)
		return [currentloglikelihood,time()-start_time]

	def gamma(self,s,k):
		"""
		Returns the probability to generate the kth obs of o in state s * bigK
		Note: it's important to compute self.bigK before
		"""
		return self.alpha_matrix[s][k]*self.beta_matrix[s][k]/self.bigK

	def xi(self,sequence,action,s1,s2,k):
		"""
		Returns the probabilty to move from state s1 to state s2 with action at time step k
		Note: it's important to compute self.bigK before
		"""
		return self.alpha_matrix[s1][k]*self.h_g(s1,action,s2,sequence[k])*self.beta_matrix[s2][k+1]/self.bigK

	def computeK(self):
		"""
		Returns sum over ss of alpha(k-1,ss) * beta(ss,k)
		"""
		self.bigK = self.beta_matrix[self.h.initial_state][0]

	def logLikelihood(self):
		return self.hhat.logLikelihoodTraces([self.sequences_sorted,self.times])

	def ghatmultiple(self,s1,s2,obs):
		"""
		returns the new values, for all the possible actions, for g[action][s1][s2][obs], format:
		[p1,p2,p3,...] with p1 = g[action1][s1][s2][obs], p2 = g[action2][s1][s2][obs], ...
		"""
		num = []
		den = []

		for i in range(len(self.actions)):
			num.append(0.0)
			den.append(0.0)

		for seq in range(len(self.sequences_sorted)):
			self.sequence_actions = [self.sequences_sorted[seq][i] for i in range(0,len(self.sequences_sorted[seq]),2)]
			sequence_obs = [self.sequences_sorted[seq][i+1] for i in range(0,len(self.sequences_sorted[seq]),2)]
			if seq == 0:
				self.precomputeMatrices(sequence_obs,-1)
			else:
				common = 0 
				while self.sequences_sorted[seq-1][common] == self.sequences_sorted[seq][common]:
					common += 1
				self.precomputeMatrices(sequence_obs,int(common/2))
			self.computeK()
			
			for k in range(len(sequence_obs)):
				den[self.actions.index(self.sequence_actions[k])] += self.gamma(s1,k) * self.times[seq]
				if sequence_obs[k] == obs:
					num[self.actions.index(self.sequence_actions[k])] += self.xi(sequence_obs,self.sequence_actions[k],s1,s2,k) * self.times[seq]
		
		return [num[i]/den[i] if den[i] != 0.0 else 0.0 for i in range(len(num))]

