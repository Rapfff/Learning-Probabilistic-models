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
	
	def precomputeMatrices(self,sequence):
		"""Here we compute all the values alpha(k,t) and beta(t,k) for a given sequence"""
		self.alpha_matrix = []

		for s in range(len(self.h.states)):
			if s == self.h.initial_state:
				self.alpha_matrix.append([1.0])
			else:
				self.alpha_matrix.append([0.0])
		
		for k in range(len(sequence)):
			possible_actions = self.getActions(k)
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					for action, proba_action in possible_actions:
						p = self.h_g(ss,action,s,sequence[k])*proba_action
						summ += self.alpha_matrix[ss][k]*p
				self.alpha_matrix[s].append(summ)

		self.beta_matrix = []

		for s in range(len(self.h.states)):
			self.beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			possible_actions = self.getActions(k)
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					for action, proba_action in possible_actions:
						p = self.h_g(s,action,ss,sequence[k])*proba_action
						summ += self.beta_matrix[ss][1 if ss<s else 0]*p
				self.beta_matrix[s].insert(0,summ)

	def problem3(self,observations):
		start_time = time()
		counter = 0
		while True:
			new_states = []
			for i in range(len(self.h.states)):
				next_probas = []
				next_states = []
				next_obs    = []

				for j in range(len(self.h.states)):
					for k in observations:
						next_probas.append(self.ghatmultiple(i,j,k))
						next_states.append(j)
						next_obs.append(k)
						
					for k in [ letter for letter in self.alphabet if not letter in observations ]:
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
			if self.checkEnd() or self.checkEndLikelihood(counter):
				self.h = self.hhat
				break
			else:
				self.h = self.hhat

		self.h.pprint()
		return [self.prevloglikelihood,time()-start_time]

	def gamma(self,s,k):
		"""
		Returns the probability to generate the kth obs of o in state s * bigK
		Note: it's important to compute self.bigK before
		"""
		return self.alpha_matrix[s][k]*self.beta_matrix[s][k]/self.bigK

	def xi(self,sequence,action,s1,s2,k):
		"""
		Returns the probabilty to move from state s1 to state s2 with action at time step k
		Note: it's important to compute self.bigK and the TUKmatrix before
		"""
		return self.alpha_matrix[s1][k]*self.h_g(s1,action,s2,sequence[k])*self.beta_matrix[s2][k+1]/self.bigK

	def computeK(self):
		"""
		Returns sum over ss of alpha(k-1,ss) * beta(ss,k)
		"""
		self.bigK = self.beta_matrix[self.h.initial_state][0]

class Estimation_algorithm_MDP_sequences(Estimation_algorithm_MDP):
	def problem3(self,traces):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [action,obs1,action2,obs2,...,actionx,obsx]
		"""
		self.traces = traces
		observations = []
		for seq in traces[0]:
			for i in range(1,len(seq),2):
				if not seq[i] in observations:
					observations.append(seq[i])

		return super().problem3(observations)

	def logLikelihood(self):
		return self.hhat.logLikelihoodTraces(self.traces)

	def getActions(self,k):
		"""Return the action at time step k, and probaility=1.0"""
		return [(self.sequence_actions[k],1.0)]

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

		for seq in range(len(self.traces[0])):
			self.sequence_actions = [self.traces[0][seq][i] for i in range(0,len(self.traces[0][seq]),2)]
			sequence_obs = [self.traces[0][seq][i+1] for i in range(0,len(self.traces[0][seq]),2)]
			times = self.traces[1][seq]
			
			self.precomputeMatrices(sequence_obs)
			self.computeK()
			
			for k in range(len(sequence_obs)):
				den[self.actions.index(self.sequence_actions[k])] += self.gamma(s1,k) * times
				if sequence_obs[k] == obs:
					num[self.actions.index(self.sequence_actions[k])] += self.xi(sequence_obs,self.sequence_actions[k],s1,s2,k) * times
		
		return [num[i]/den[i] if den[i] != 0.0 else 0.0 for i in range(len(num))]



class Estimation_algorithm_MDP_schedulers(Estimation_algorithm_MDP):
	def problem3(self,data):
		"""
		Given a set of pairs scheduler-sequence of observations,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		data = [(scheduler1,traces1),(scheduler2,traces2)]
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [obs1,obs2,...]
		"""
		self.data = data
		print(data)
		observations = []
		for sched in data:
			for seq in sched[1][0]:
				for i in seq:
					if not i in observations:
						observations.append(i)
		observations.sort()
		return super().problem3(observations)
	
	def logLikelihood(self):
		return self.hhat.logLikelihoodObservationsScheduler(self.data)

	def getActions(self,k):
		temp = self.scheduler.get_actions(self.seq_sched_state[k])
		res = []
		for i in range(len(temp[0])):
			res.append((temp[1][i],temp[0][i]))
		return res

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

		
		for data_index in range(len(self.data)):
			self.scheduler = self.data[data_index][0]
			traces = self.data[data_index][1]

			for seq in range(len(traces[0])):
				sequence = traces[0][seq]
				times = traces[1][seq]
				self.seq_sched_state = self.scheduler.get_sequence_states(sequence)
				
				self.precomputeMatrices(sequence)
				self.computeK()
				
				for k in range(len(sequence)):
					for act in range(len(self.actions)):
						den[act] += self.gamma(s1,k) * times
						if sequence[k] == obs:
							num[act] += self.xi(sequence,self.actions[act],s1,s2,k) * times * self.scheduler.get_probability(self.actions[act],self.seq_sched_state[k])

		return [num[i]/den[i] if den[i] != 0.0 else 0.0 for i in range(len(num))]