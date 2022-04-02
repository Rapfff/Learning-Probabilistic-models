import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from multiprocessing import cpu_count, Pool
NB_PROCESS = cpu_count()-1
from datetime import datetime

class BW:
	def __init__(self,initial_model):
		"""
		h is a model
		"""
		self.h = initial_model
		self.hhat = initial_model
		self.nb_states = len(self.h.states)
		#OVERRIDED self.alphabet

	def h_tau(self,s1,s2,obs):
		return self.h.tau(s1,s2,obs)

	def computeAlphas(self,sequence):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		alpha_matrix = []
		for i in range(self.nb_states):
			alpha_matrix.append([self.h.initial_state[i]])

		for k in range(len(sequence)):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_tau(ss,s,sequence[k])
					if p > 0.0:
						summ += alpha_matrix[ss][k]*p
				alpha_matrix[s].append(summ)
		return alpha_matrix

	def computeBetas(self,sequence):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_tau(s,ss,sequence[k])
					if p > 0.0:
						summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ)

		return beta_matrix

	def processWork(self,sequence,times):
		#overrided
		pass

	def generateHhat(self):
		#overrided
		pass

	def learn(self,traces,output_file="output_model.txt",epsilon=0.01,verbose=False,pp=''):
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
				print(datetime.now(),pp,counter, prevloglikelihood/nb_traces)
			self.hhat, currentloglikelihood = self.generateHhat(traces)
			
			counter += 1
			self.h = self.hhat
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				
		self.h.save(output_file)
		return self.h
