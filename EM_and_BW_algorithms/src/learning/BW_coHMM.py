import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.coHMM import *
from learning.BW import *
from multiprocessing import cpu_count, Pool
from tools import correct_proba
from math import sqrt, log

class BW_coHMM(BW):
	def __init__(self,initial_model):
		"""
		h is a HMM
		"""
		super().__init__(initial_model)

	def processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		
		proba_seq = beta_matrix[self.h.initial_state][0]
		if proba_seq != 0.0:
			####################
			den = []
			for s in range(self.nb_states):
				den.append(0.0)
				for t in range(len(sequence)):
					den[-1] += alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
			####################
			num_a = []
			num_mu = []
			num_va = []
			for s in range(self.nb_states):
				num_a.append([0.0 for i in range(self.nb_states)])
				num_mu.append(0.0)
				num_va.append(0.0)
				for t in range(len(sequence)):
					observation = sequence[t]
					for ss in range(self.nb_states):
						gamma = alpha_matrix[s][t]*beta_matrix[s][t]
						num_a[-1][ss]  += alpha_matrix[s][t]*self.h.tau(s,ss,observation)*beta_matrix[ss][t+1]*times/proba_seq
						num_mu[-1] += gamma*observation*times/proba_seq
						num_va[-1] += gamma*(observation-self.h.states[s].output_parameters[0])**2*times/proba_seq
			####################
			return [den,num_a,num_mu,num_va,proba_seq,times]
		return False

	def generateHhat(self,traces):
		den = []
		for s in range(self.nb_states):
			den.append(0.0)
		a = []
		for s in range(self.nb_states):
			a.append([0 for i in range(self.nb_states)])
		mu  = [ 0.0 for i in range(self.nb_states)]
		std = [ 0.0 for i in range(self.nb_states)]
		
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[4])*i[5] for i in temp])

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.nb_states):
				a[s][x] = sum([i[1][s][x] for i in temp])

			mu[s]  = sum(i[2][s] for i in temp)
			std[s] = sum(i[3][s] for i in temp)

		new_states = []
		for s in range(self.nb_states):
			la = [ correct_proba([a[s][i]/den[s] for i in range(self.nb_states)]) , list(range(self.nb_states))]
			lb = [mu[s]/den[s],sqrt(std[s]/den[s])]
			new_states.append(coHMM_state(la,lb))

		return [coHMM(new_states,self.h.initial_state),currentloglikelihood]

