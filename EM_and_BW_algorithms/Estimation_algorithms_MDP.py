from MDP import *
from tools import correct_proba
from random import randint
from time import time
# a MDP is fully observable if the obs in each state is unique (for all s,s' s.t. s != s' then s.obs != s'.obs )
class Estimation_algorithm_fullyobservable_MDP:
	def __init__(self):
		None

	def learnFromSequences(self,sequences):
		"""Passive learning: we already have the training set"""
		actions_id = []
		for seq in sequences:
			for j in seq[1]:
				if not j in actions_id:
					actions_id.append(j)
		
		states_id = []
		for seq in sequences:
			for j in seq[0]:
				if not j in states_id:
					states_id.append(j)

		count_matrix = []
		for i in range(len(states_id)):
			count_matrix.append([])
			for j in range(len(actions_id)):
				count_matrix[-1].append([])
				for k in range(len(states_id)):
					count_matrix[-1][-1].append(0)

		for seq in sequences:
			for i in range(len(seq[1])):
				count_matrix[states_id.index(seq[0][i])][actions_id.index(seq[1][i])][states_id.index(seq[0][i+1])] += 1

		states = []
		for s1 in range(len(states_id)):
			dic_state = {}
			for a in range(len(actions_id)):
				den = sum(count_matrix[s1][a])
				if den > 0:
					list_action = [[],[]]
					for s2 in range(len(states_id)):
						num = count_matrix[s1][a][s2]
						if num > 0:
							list_action[0].append(num/den)
							list_action[1].append(s2)
					list_action[0] = correct_proba(list_action[0],2)
					dic_state[actions_id[a]] = list_action
			states.append(MDP_state(dic_state, states_id[s1]))

		return MDP(states,states_id.index(sequences[0][0][0]))

	def learnFromBlackBox(self,black_box,l,length_exp):
		"""Active learning: we have access to the MDP and we generate the sequences"""

		#TO DO: we need to bui scheduler_random
		sequences = []
		for i in range(l):
			sequences.append(black_box.run(length_exp, scheduler_random))
		return self.learnFromSequences(sequences)



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
						if self.h_g(i,act,j,k) != self.g_hhat(i,act,j,k):
							return False
		return True

	def checkEndLikelihood(self,c):
		if c < 10:
			self.prevloglikelihood = -256
			print(c)
			return False
		
		currentloglikelihood = self.hhat.logLikelihoodTraces(traces)
		print(c,"- loglikelihood :",currentloglikelihood)
		if self.prevloglikelihood == currentloglikelihood:
			self.prevloglikelihood = currentloglikelihood
			return True
		else:
			self.prevloglikelihood = currentloglikelihood
			return False

	def h_g(self,s1,act,s2,obs):
		return self.h.states[s1].g(act,s2,obs)

	def hhat_g(self,s1,act,s2,obs):
		return self.hhat.states[s1].g(act,s2,obs)

	def problem3(self,traces):
		"""
		Given, for several finite-memory scheduler, a set of sequences of observations,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		data = [[schedulerA, [sequenceA1,sequenceA2,...]],[schedulerB, [sequenceB1,sequenceB2,...]],...]
		sequence = [ob1,obs2,...]
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [action,obs1,action2,obs2,...,actionx,obsx]
		"""
		start_time = time()
		self.traces = traces
		counter = 0

		observations = []
		for seq in traces:
			for i in range(1,len(seq),2):
				if not seq[i] in observations:
					observations.append(seq[i])

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

	
	def precomputeMatrices(self,sequence,actions):
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
					p = self.h_g(ss,actions[k],s,sequence[k])
					summ += self.alpha_matrix[ss][k]*p
				self.alpha_matrix[s].append(summ)

		self.beta_matrix = []

		for s in range(len(self.h.states)):
			self.beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			for s in range(len(self.h.states)):
				summ = 0
				for ss in range(len(self.h.states)):
					p = self.h_g(s,actions[k],ss,sequence[k])
					if p > 0:
						summ += self.beta_matrix[ss][1 if ss<s else 0]*p
				self.beta_matrix[s].insert(0,summ)

	def gamma(self,s,k):
		"""
		Returns the probability to generate the kth obs of o in state s * bigK
		Note: it's important to compute self.bigK before
		"""
		return self.alpha_matrix[s][k]*self.beta_matrix[s][k]/self.bigK

	def xi(self,action,s1,s2,k):
		"""
		Returns the probabilty to move from state s1 to state s2 at time step k
		Note: it's important to compute self.bigK and the TUKmatrix before
		"""
		return self.alpha_matrix[s1][k]*self.h_g(s1,action,s2,self.sequence[k])*self.beta_matrix[s2][k+1]/self.bigK

	def computeK(self):
		"""
		Returns sum over ss of alpha(k-1,ss) * beta(ss,k)
		"""
		self.bigK = self.beta_matrix[self.h.initial_state][0]

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
			actions = [self.traces[0][seq][i] for i in range(0,len(self.traces[0][seq]),2)]
			self.sequence = [self.traces[0][seq][i+1] for i in range(0,len(self.traces[0][seq]),2)]
			times = self.traces[1][seq]
			
			self.precomputeMatrices(self.sequence,actions)
			self.computeK()
			
			for k in range(len(self.sequence)):
				den[self.actions.index(actions[k])] += self.gamma(s1,k) * times
				if self.sequence[k] == obs:
					num[self.actions.index(actions[k])] += self.xi(actions[k],s1,s2,k) * times
		
		return [num[i]/den[i] if den[i] != 0.0 else 0.0 for i in range(len(num))]
