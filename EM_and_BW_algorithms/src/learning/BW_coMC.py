import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.coMC import *
from learning.BW import *
from multiprocessing import cpu_count, Pool
from tools import correct_proba
from math import sqrt, log

class BW_coMC(BW):
	def __init__(self,initial_model):
		"""
		h is a MC
		"""
		super().__init__(initial_model)

	def processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq != 0.0:
			den_a = []
			num_a = []
			num_mu = []
			num_va = []
			for s in range(self.nb_states):
				den_a.append(0.0)
				# den_b == num_a  !!
				num_a.append([0.0 for i in range(self.nb_states)])
				num_mu.append([0.0 for i in range(self.nb_states)])
				num_va.append([0.0 for i in range(self.nb_states)])
				for t in range(len(sequence)):
					den_a[-1] += alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
					observation = sequence[t]
					for ss in range(self.nb_states):
						xi = alpha_matrix[s][t]*self.h.tau(s,ss,observation)*beta_matrix[ss][t+1]*times/proba_seq
						num_a[-1][ss]  += xi
						num_mu[-1][ss] += xi*observation
						num_va[-1][ss] += xi*(observation-self.h.states[s].obs_matrix[ss][0])**2
			####################
			num_init = [alpha_matrix[s][0] for s in range(self.nb_states)]
			return [den_a,num_a,num_mu,num_va,proba_seq,times,num_init]
		return False

	def generateHhat(self,traces):
		den = []
		num_a = []
		mu = []
		std = []
		for s in range(self.nb_states):
			den.append(0.0)
			num_a.append(  [0.0 for i in range(self.nb_states)])
			mu.append( [ 0.0 for i in range(self.nb_states)])
			std.append([ 0.0 for i in range(self.nb_states)])
			
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[4])*i[5] for i in temp])

		num_init = [0.0 for s in range(self.nb_states)]
		for i in temp:
			for s in range(self.nb_states):
				num_init[s] += i[6][s]*i[5]

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.nb_states):
				num_a[s][x]   = sum([i[1][s][x] for i in temp])
				mu[s][x]  = sum([i[2][s][x] for i in temp])
				std[s][x] = sum([i[3][s][x] for i in temp])

		new_states = []
		for s in range(self.nb_states):
			la = [ correct_proba([num_a[s][i]/den[s] for i in range(self.nb_states)]) , list(range(self.nb_states))]
			d = {}
			for ss in range(self.nb_states):
				d[ss] = [mu[s][ss]/num_a[s][ss],sqrt(std[s][ss]/num_a[s][ss])]
			new_states.append(coMC_state(la,d))

		initial_state = [num_init[s]/sum(traces[1]) for s in range(self.nb_states)]

		return [coMC(new_states,self.h.initial_state),currentloglikelihood]

