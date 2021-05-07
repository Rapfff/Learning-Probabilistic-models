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

	def h_g(self,s1,act,s2,obs):
		return self.h.g(s1,act,s2,obs)

	def hhat_g(self,s1,act,s2,obs):
		return self.hhat.g(s1,act,s2,obs)
	
	def computeAlphas(self,sequence,sequence_actions,common,alpha_matrix):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		for k in range(common,len(sequence)):
			action = sequence_actions[k]
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					p = self.h_g(ss,action,s,sequence[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ

		return alpha_matrix

	def computeBetas(self,sequence,sequence_actions):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []
		for s in range(len(self.h.states)):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			action = sequence_actions[k]
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					p = self.h_g(s,action,ss,sequence[k])
					summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ)

		return beta_matrix

	def learn(self,traces,output_file="output_model.txt",limit=0.01,pp=''):
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
		prevloglikelihood = 10
		while True:
			print(datetime.datetime.now(),pp,counter, prevloglikelihood)
			new_states = []
			for i in range(len(self.h.states)):
				next_probas = []
				next_states = []
				next_obs    = []
				
				p = Pool(processes = cpu_count()-1)
				tasks = []
				#temp = []
				for j in range(len(self.h.states)):
					for k in self.observations:
						tasks.append(p.apply_async(self.ghatmultiple, [i,j,k,]))
						#temp.append(self.ghatmultiple(i,j,k))
				p.close()
				temp = [res.get() for res in tasks]

				next_probas = [ temp[t][2] for t in range(len(temp)) ]
				next_states = [ temp[t][0] for t in range(len(temp)) ]
				next_obs    = [ temp[t][1] for t in range(len(temp)) ]

				for j in temp:
					if len(j) == 4:
						currentloglikelihood = j[3]
				
				for j in range(len(self.h.states)):			
					for k in [ letter for letter in self.alphabet if not letter in self.observations ]:
						next_probas.append([0.0]*len(self.actions))
						next_states.append(j)
						next_obs.append(k)

				dic = {}
				for act in range(len(self.actions)):
					probas = [k[act] for k in next_probas]
					if sum(probas) != 0.0 :
						probas = correct_proba(probas)
						dic[self.actions[act]] = [probas, next_states, next_obs]
				new_states.append(MDP_state(dic))

			self.hhat = MDP(new_states,self.h.initial_state)
			
			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < limit: #ici on compare h et celui avant(pas hhat)
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat

		print()
		self.h.save(output_file)
		return self.h

	def ghatmultiple(self,s1,s2,obs):
		"""
		returns the new values, for all the possible actions, for g[action][s1][s2][obs], format:
		[p1,p2,p3,...] with p1 = g[action1][s1][s2][obs], p2 = g[action2][s1][s2][obs], ...
		"""
		num = []
		den = []

		if s1 == 0 and s2 == 0 and self.observations.index(obs) == 0:
			loglikelihood = 0.0
		else:
			loglikelihood = None

		for i in range(len(self.actions)):
			num.append(0.0)
			den.append(0.0)

		for seq in range(len(self.sequences_sorted)):
			sequence_actions = [self.sequences_sorted[seq][i] for i in range(0,len(self.sequences_sorted[seq]),2)]
			sequence_obs = [self.sequences_sorted[seq][i+1] for i in range(0,len(self.sequences_sorted[seq]),2)]
			if seq == 0:
				common = 0
				alpha_matrix = []

				for s in range(len(self.h.states)):
					if s == self.h.initial_state:
						alpha_matrix.append([1.0])
					else:
						alpha_matrix.append([0.0])
					alpha_matrix[-1] += [None for i in range(len(sequence_obs))]
			else:
				common = 0 
				while self.sequences_sorted[seq-1][common] == self.sequences_sorted[seq][common]:
					common += 1
			
			alpha_matrix = self.computeAlphas(sequence_obs,sequence_actions,int(common/2),alpha_matrix)
			beta_matrix  = self.computeBetas(sequence_obs,sequence_actions)
			
			bigK = beta_matrix[self.h.initial_state][0]
			
			for k in range(len(sequence_obs)):
				gamma_s1_k = alpha_matrix[s1][k] * beta_matrix[s1][k] / bigK
				
				den[self.actions.index(sequence_actions[k])] += gamma_s1_k * self.times[seq]
				
				if sequence_obs[k] == obs:
					num[self.actions.index(sequence_actions[k])] += alpha_matrix[s1][k]*self.h_g(s1,sequence_actions[k],s2,obs)*beta_matrix[s2][k+1] * self.times[seq] / bigK
		
			if loglikelihood != None:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.h.states))]))

		if loglikelihood == None:
			return (s2,obs,[num[i]/den[i] if den[i] != 0.0 else 0.0 for i in range(len(num))])

		return (s2,obs,[num[i]/den[i] if den[i] != 0.0 else 0.0 for i in range(len(num))], loglikelihood)

