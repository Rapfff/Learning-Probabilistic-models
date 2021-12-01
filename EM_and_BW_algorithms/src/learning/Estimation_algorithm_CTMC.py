import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.CTMC import *
from multiprocessing import cpu_count, Pool
from time import time
from tools import correct_proba
import datetime

NB_PROCESS = 11

class Estimation_algorithm_CTMC:
	def __init__(self,h):
		"""
		h is a MCGT
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.hhat = h
		self.alphabet = self.h.observations()
		self.nb_states = len(self.h.states)

	def h_g(self,s1,s2,obs):
		return self.h.g(s1,s2,obs)

	def h_v(self,s,t):
		return self.h.v(s,t)

	def computeAlphas(self,sequence):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		alpha_matrix = []
		for i in range(self.nb_states):
			if i == self.h.initial_state:
				alpha_matrix.append([1.0])
			else:
				alpha_matrix.append([0.0])

		for k in range(1,len(sequence),2):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(ss,s,sequence[k])*self.h_v(ss,sequence[k-1])
					if p > 0.0:
						summ += alpha_matrix[ss][k//2]*p
				alpha_matrix[s].append(summ)
		return alpha_matrix

	def computeBetas(self,sequence):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-2):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(s,ss,sequence[k])
					if p > 0.0:
						summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ*self.h_v(s,sequence[k-1]))

		return beta_matrix

	def processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		
		proba_seq = beta_matrix[self.h.initial_state][0]
		if proba_seq != 0.0:
			####################
			den = []
			rate_parameters = []
			for s in range(self.nb_states):
				den.append(0.0)
				rate_parameters.append(0.0)
				for t in range(len(sequence)//2):
					den[-1] += alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
					rate_parameters[-1] += alpha_matrix[s][t]*self.h_v(s,sequence[t*2])*beta_matrix[s][t]*times/proba_seq
			####################
			num = []
			for s in range(self.nb_states):
				num.append([0.0 for i in range(self.nb_states*len(self.observations))])
				for t in range(len(sequence)//2):
					observation = sequence[t*2+1]
					for ss in range(self.nb_states):
						p = self.h_g(s,ss,observation)
						#for i in range(len(self.h.states[s].next_matrix[0])):
						#	if self.h.states[s].next_matrix[1][i] == ss and self.h.states[s].next_matrix[2][i] == observation:
						#		p = self.h.states[s].next_matrix[0][i]
						#		break
						if p != 0.0:
							num[-1][ss*len(self.observations)+self.observations.index(observation)] += alpha_matrix[s][t]*p*beta_matrix[ss][t+1]*times/proba_seq
			####################
			return [den,num, proba_seq,times, rate_parameters]
		return False


	def learn(self,traces,output_file="output_model.txt",epsilon=0.01,pp=''):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [obs1,obs2,...,obsx]
		"""
		
		self.observations = []
		for seq in traces[0]:
			for i in list(set([seq[j] for j in range(1,len(seq),2)])):
				if not i in self.observations:
					self.observations.append(i)

		counter = 0
		prevloglikelihood = 10
		while True:
			#print(datetime.datetime.now(),pp,counter, prevloglikelihood)
			den = []
			rate_parameters = []
			for s in range(self.nb_states):
				den.append(0.0)
				rate_parameters.append(0.0)
			tau = []
			for s in range(self.nb_states):
				tau.append([0 for i in range(self.nb_states*len(self.observations))])
			
			p = Pool(processes = NB_PROCESS)
			tasks = []
			
			for seq in range(len(traces[0])):
				tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
			
			temp = [res.get() for res in tasks if res.get() != False]
			currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

			for s in range(self.nb_states):
				den[s] = sum([i[0][s] for i in temp])
				rate_parameters[s] = sum([i[4][s] for i in temp])
				for x in range(self.nb_states*len(self.observations)):
					tau[s][x] = sum([i[1][s][x] for i in temp])


			list_sta = []
			for i in range(self.nb_states):
				for o in self.observations:
					list_sta.append(i)
			list_obs = self.observations*self.nb_states
			new_states = []
			for s in range(self.nb_states):
				l1 = [ correct_proba([tau[s][i]/den[s] for i in range(len(list_sta))]) , list_sta, list_obs ]
				exp_lambda = den[s]/rate_parameters[s]
				new_states.append(CTMC_state(l1,exp_lambda))

			self.hhat = CTMC(new_states,self.h.initial_state)
			self.hhat.pprint()
			for i in temp:
				print(i[2])
			input()
			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat

		self.h.save(output_file)
		return self.h
