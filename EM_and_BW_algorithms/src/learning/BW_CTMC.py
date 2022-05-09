import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.CTMC import *
from multiprocessing import cpu_count, Pool
from time import time
from  datetime import datetime
from math import log

NB_PROCESS = cpu_count()-1

class BW_CTMC:
	def __init__(self,h: CTMC) -> None:
		"""
		h is a CTMC
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.alphabet = self.h.observations()
		self.nb_states = len(self.h.states)

	def h_tau(self,s1: int, s2: int, obs: str) -> float:
		return self.h.tau(s1,s2,obs)

	def h_e(self,s: int) -> float:
		return self.h.e(s)
	
	def h_l(self, s1: int, s2: int, obs: str) -> float:
		return self.h.l(s1,s2,obs)

	def computeAlphas(self,obs_seq: list, times_seq: list = None) -> list:
		if times_seq:
			return self.computeAlphas_timed(obs_seq,times_seq)
		else:
			return self.computeAlphas_nontimed(obs_seq)

	def computeBetas(self,obs_seq: list, times_seq: list = None) -> list:
		if times_seq:
			return self.computeBetas_timed(obs_seq,times_seq)
		else:
			return self.computeBetas_nontimed(obs_seq)

	def computeAlphas_timed(self,obs_seq: list, times_seq: list) -> list:
		"""Here we compute all the values alpha(k,t) for a given obs_seq"""
		alpha_matrix = []
		for i in range(self.nb_states):
			alpha_matrix.append([self.h.pi(i)])

		for k in range(len(obs_seq)):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					summ += alpha_matrix[ss][k]*self.h_l(ss,s,obs_seq[k])*exp(-self.h_e(ss)*times_seq[k])
				alpha_matrix[s].append(summ)
		return alpha_matrix

	def computeBetas_timed(self,obs_seq: list, times_seq: list) -> list:
		"""Here we compute all the values beta(t,k) for a given obs_seq"""
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(obs_seq)-1,-1,-1):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					summ += beta_matrix[ss][1 if ss<s else 0]*self.h_l(s,ss,obs_seq[k])
				summ *= exp(-self.h_e(s)*times_seq[k])
				beta_matrix[s].insert(0,summ)
		return beta_matrix

	def computeAlphas_nontimed(self,sequence: list) -> list:
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		# sequence is UNTIMED
		alpha_matrix = []
		for i in range(self.nb_states):
			alpha_matrix.append([self.h.pi(i)])

		for k in range(len(sequence)):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					summ += alpha_matrix[ss][k]*self.h_tau(ss,s,sequence[k])
				alpha_matrix[s].append(summ)
		return alpha_matrix

	def computeBetas_nontimed(self,sequence: list) -> list:
		"""Here we compute all the values beta(t,k) for a given sequence"""
		# sequence is UNTIMED
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					summ += beta_matrix[ss][1 if ss<s else 0]*self.h_tau(s,ss,sequence[k])
				beta_matrix[s].insert(0,summ)
		return beta_matrix

	def splitTime(self,sequence: list) -> tuple:
		if type(sequence[0]) == float and type(sequence[1]) == str:
			times_seq = [sequence[i] for i in range(0,len(sequence),2)]
			obs_seq   = [sequence[i] for i in range(1,len(sequence),2)]
		else:
			times_seq = None
			obs_seq = sequence
		return (times_seq,obs_seq)

	def processWork(self,sequence: list, times: int):
		times_seq, obs_seq = self.splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True

		alpha_matrix = self.computeAlphas(obs_seq, times_seq)
		beta_matrix  = self.computeBetas( obs_seq, times_seq)
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq == 0.0:
			return False
		####################
		den = []
		num = []
		num_init = []
		for s in range(self.nb_states):
			den.append(0.0)
			num.append([0.0 for i in range(self.nb_states*len(self.alphabet))])
			
			for t in range(len(obs_seq)):
				if timed:
					den[-1] += alpha_matrix[s][t]*beta_matrix[s][t]*times_seq[t]
				else:
					den[-1] += alpha_matrix[s][t]*beta_matrix[s][t]

				observation = obs_seq[t]
				for ss in range(self.nb_states):
					if timed:
						num[-1][ss*len(self.alphabet)+self.alphabet.index(observation)] += alpha_matrix[s][t]*exp(-self.h_e(s)*times_seq[t])*self.h_l(s,ss,observation)*beta_matrix[ss][t+1]
					else:
						num[-1][ss*len(self.alphabet)+self.alphabet.index(observation)] += alpha_matrix[s][t]*self.h_l(s,ss,observation)*beta_matrix[ss][t+1]
					
			num[-1]  = [i*times/proba_seq for i in num[-1]]
			den[-1] *=    times/proba_seq
			num_init.append(alpha_matrix[s][0]*beta_matrix[s][0]*times/proba_seq)
		####################
		return [den, num, proba_seq, times, num_init]

	def _newProbabilities(self,tau,den):
		return [i/den for i in tau]

	def generateHhat(self,traces):
		p = Pool(processes = NB_PROCESS)
		tasks = []
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		temp = [res.get() for res in tasks if res.get() != False]
		#temp = []
		#for seq in range(len(traces[0])):
		#	temp.append(self.processWork(traces[0][seq], traces[1][seq]))

		currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

		den = []
		for s in range(self.nb_states):
			den.append(0.0)
		tau = []
		for s in range(self.nb_states):
			tau.append([0 for i in range(self.nb_states*len(self.alphabet))])
		num_init = [0.0 for s in range(self.nb_states)]

		for i in temp:
			for s in range(self.nb_states):
				num_init[s] += i[4][s]

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.nb_states*len(self.alphabet)):
				tau[s][x] = sum([i[1][s][x] for i in temp])

		list_sta = []
		for i in range(self.nb_states):
			for o in self.alphabet:
				list_sta.append(i)
		list_obs = self.alphabet*self.nb_states
		new_states = []
		for s in range(self.nb_states):
			l = [self._newProbabilities(tau[s],den[s]), list_sta, list_obs]
			new_states.append(CTMC_state(l))

		initial_state = [num_init[s]/sum(traces[1]) for s in range(self.nb_states)]

		return [CTMC(new_states,initial_state),currentloglikelihood]

	def learn(self,traces,output_file=None,epsilon=0.01,verbose=False,pp=''):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [obs1,obs2,...,obsx]
		"""
		counter = 0
		prevloglikelihood = 10
		nb_traces = sum(traces[1])
		while True:
			if verbose:
				print(datetime.now(),pp,counter, prevloglikelihood/nb_traces,end='\r')
			hhat, currentloglikelihood = self.generateHhat(traces)

			counter += 1
			self.h = hhat
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
		if output_file:
			self.h.save(output_file)
		if verbose:
			print()
		return self.h
