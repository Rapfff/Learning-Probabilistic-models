import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.coHMM import *
from learning.BW import *
from multiprocessing import cpu_count, Pool
from tools import correct_proba
from math import sqrt, log
from statistics import stdev, mean


class BW_coHMM(BW):
	def __init__(self,initial_model):
		"""
		h is a coHMM
		"""
		super().__init__(initial_model)

	def processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq != 0.0:
			den = []
			num_a = []
			num_mu = []
			num_va = []
			for s in range(self.nb_states):
				den.append(0.0)
				num_a.append([0.0 for i in range(self.nb_states)])
				num_mu.append(0.0)
				num_va.append(0.0)
				for t in range(len(sequence)):
					gamma = alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
					den[-1] += gamma
					observation = sequence[t]
					for ss in range(self.nb_states):
						num_a[-1][ss] += alpha_matrix[s][t]*self.h.tau(s,ss,observation)*beta_matrix[ss][t+1]*times/proba_seq
						num_mu[-1] += gamma*observation
						num_va[-1] += gamma*(observation-self.h.states[s].output_parameters[0])**2
			####################
			num_init = [alpha_matrix[s][0]*beta_matrix[s][0]*times/proba_seq for s in range(self.nb_states)]
			return [den,num_a,num_mu,num_va,proba_seq,times,num_init]
		return False

	def generateHhat(self,traces):
		a = []
		for s in range(self.nb_states):
			a.append([0 for i in range(self.nb_states)])
		mu       = [ 0.0 for i in range(self.nb_states)]
		va       = [ 0.0 for i in range(self.nb_states)]
		den      = [ 0.0 for i in range(self.nb_states)]
		num_init = [ 0.0 for i in range(self.nb_states)]
		 
		p = Pool(processes = NB_PROCESS)
		tasks = []
		temp = []
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
			temp.append(self.processWork(traces[0][seq], traces[1][seq]))	

		temp = [res.get() for res in tasks if res.get() != False]
		
		# We remove the sequences that are very very different
		list_loglikelihoods = [log(i[4])*i[5] for i in temp]
		moyenne = mean(list_loglikelihoods)
		ecarttype = stdev(list_loglikelihoods)
		temp = [i for i in temp if abs(moyenne-log(i[4]))<ecarttype]
		#######################################################

		currentloglikelihood = sum([log(i[4])*i[5] for i in temp])

		for s in range(self.nb_states):
			for x in range(self.nb_states):
				a[s][x] = sum([i[1][s][x] for i in temp])
			
			den[s]      = sum(i[0][s] for i in temp)
			num_init[s] = sum(i[6][s] for i in temp)
			mu[s]       = sum(i[2][s] for i in temp)
			va[s]       = sum(i[3][s] for i in temp)

		new_states = []
		for s in range(self.nb_states):
			la = [ correct_proba([a[s][i]/den[s] for i in range(self.nb_states)]) , list(range(self.nb_states))]
			lb = [ mu[s]/den[s] , sqrt(va[s]/den[s])]
			new_states.append(coHMM_state(la,lb))

		initial_state = [num_init[s]/sum(num_init) for s in range(self.nb_states)]
		
		return [coHMM(new_states,initial_state),currentloglikelihood]

