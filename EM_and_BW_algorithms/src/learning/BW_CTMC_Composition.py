import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.CTMC import *
from learning.BW_CTMC import *

NB_PROCESS = 11

class BW_CTMC_Composition(BW_CTMC):
	def __init__(self,h1: CTMC, h2: CTMC) -> None:
		self.nb_states_1 = len(h1.states)
		self.nb_states_2 = len(h2.states)
		super().__init__(parallelComposition(h1,h2))
	
	def _newProbabilities(self,tau,den,len_list_sta,s1):
		return [tau[i]/den if i%self.nb_states_1 != s1%self.nb_states_1 else self.h.states[s1].lambda_matrix[i] for i in range(len_list_sta)]


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
