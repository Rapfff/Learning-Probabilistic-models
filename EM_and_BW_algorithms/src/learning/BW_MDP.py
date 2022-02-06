import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.MDP import *
from tools import correct_proba
from random import randint
from multiprocessing import cpu_count, Pool
from time import time
import datetime

NB_PROCESS = 11

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
		self.nb_states = len(self.h.states)

	def h_g(self,s1,act,s2,obs):
		return self.h.g(s1,act,s2,obs)

	def hhat_g(self,s1,act,s2,obs):
		return self.hhat.g(s1,act,s2,obs)

	def computeAlphas(self,sequence,sequence_actions):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		alpha_matrix = []
		for i in range(self.nb_states):
			if i == self.h.initial_state:
				alpha_matrix.append([1.0])
			else:
				alpha_matrix.append([0.0])

		for k in range(0,len(sequence)):
			action = sequence_actions[k]
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(ss,action,s,sequence[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s].append(summ)

		return alpha_matrix

	def computeBetas(self,sequence,sequence_actions):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			action = sequence_actions[k]
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(s,action,ss,sequence[k])
					summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ)

		return beta_matrix

	def processWork(self,sequence,times):
		sequence_actions = [sequence[i] for i in range(0,len(sequence),2)]
		sequence_obs = [sequence[i+1] for i in range(0,len(sequence),2)]
		
		alpha_matrix = self.computeAlphas(sequence_obs,sequence_actions)
		beta_matrix = self.computeBetas(sequence_obs,sequence_actions)
		
		proba_seq = beta_matrix[self.h.initial_state][0]
		if proba_seq != 0.0:
			####################
			den = []
			for s in range(self.nb_states):
				den.append({})
				for a in self.actions:
					den[-1][a] = 0.0
				for t in range(len(sequence_actions)):
					den[-1][sequence_actions[t]] += alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
			####################
			num = []
			for s in range(self.nb_states):
				num.append({})
				for a in self.actions:
					num[-1][a] =  [0.0 for i in range(self.nb_states*len(self.observations))]
				for t in range(len(sequence_obs)):
					action = sequence_actions[t]
					observation = sequence_obs[t]
					
					for ss in range(self.nb_states):
						p = 0.0
						for i in range(len(self.h.states[s].next_matrix[action][0])):
							if self.h.states[s].next_matrix[action][1][i] == ss and self.h.states[s].next_matrix[action][2][i] == observation:
								p = self.h.states[s].next_matrix[action][0][i]
								break
						if p != 0.0:
							num[-1][action][ss*len(self.observations)+self.observations.index(observation)] += alpha_matrix[s][t]*p*beta_matrix[ss][t+1]*times/proba_seq
			####################
			return [den,num, proba_seq,times]
		return False


	def learn(self,traces,output_file="output_model.txt",epsilon=0.01,pp=''):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [action,obs1,action2,obs2,...,actionx,obsx]
		"""
		
		self.observations = []
		for seq in traces[0]:
			for i in range(1,len(seq),2):
				if not seq[i] in self.observations:
					self.observations.append(seq[i])

		counter = 0
		prevloglikelihood = 10
		while True:
			#print(datetime.datetime.now(),pp,counter, prevloglikelihood)
			den = []
			for s in range(self.nb_states):
				den.append({})
				for a in self.actions:
					den[-1][a] = 0
			tau = []
			for s in range(self.nb_states):
				tau.append({})
				for a in self.actions:
					tau[-1][a] = [0 for i in range(self.nb_states*len(self.observations))]
			
			p = Pool(processes = NB_PROCESS)
			tasks = []
			
			for seq in range(len(traces[0])):
				tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
			
			temp = [res.get() for res in tasks if res.get() != False]
			currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

			for s in range(self.nb_states):
				for a in self.actions:
					den[s][a] = sum([i[0][s][a] for i in temp])
					
					for x in range(self.nb_states*len(self.observations)):
						tau[s][a][x] = sum([i[1][s][a][x] for i in temp])

			list_sta = []
			for i in range(self.nb_states):
				for o in self.observations:
					list_sta.append(i)
			list_obs = self.observations*self.nb_states
			new_states = []
			for s in range(self.nb_states):
				dic = {}
				for a in self.actions:
					dic[a] = [ correct_proba([tau[s][a][i]/den[s][a] for i in range(len(list_sta))]) , list_sta, list_obs ]
				new_states.append(MDP_state(dic))

			self.hhat = MDP(new_states,self.h.initial_state)
			
			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat

		self.h.save(output_file)
		return self.h
