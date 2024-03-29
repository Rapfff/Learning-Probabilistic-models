import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.HMM import *
from learning.BW import *
from multiprocessing import cpu_count, Pool
from tools import correct_proba
from math import log

class BW_HMM(BW):
	def __init__(self,initial_model):
		"""
		h is a HMM
		"""
		super().__init__(initial_model)

	def processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq != 0.0:
			num_a = []
			num_b = []
			den   = []
			for s in range(self.nb_states):
				den.append(0.0)
				num_a.append([0.0 for i in range(self.nb_states)])
				num_b.append([0.0 for i in range(len(self.h.observations()))])
				for t in range(len(sequence)):
					gamma = alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
					den[-1] += gamma
					observation = sequence[t]
					for ss in range(self.nb_states):
						num_a[-1][ss] += alpha_matrix[s][t]*self.h.a(s,ss)*self.h.b(s,observation)*beta_matrix[ss][t+1]*times/proba_seq
					num_b[-1][self.h.observations().index(observation)] += gamma
			####################
			num_init = [alpha_matrix[s][0]*beta_matrix[s][0]*times/proba_seq for s in range(self.nb_states)]
			return [den,num_a,num_b,proba_seq,times,num_init]
		return False

	def generateHhat(self,traces):
		den = []
		for s in range(self.nb_states):
			den.append(0.0)
		a = []
		for s in range(self.nb_states):
			a.append([0 for i in range(self.nb_states)])
		b = []
		for s in range(self.nb_states):
			b.append([0 for i in range(len(self.h.observations()))])
		
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[3])*i[4] for i in temp])
		#sum_proba= sum([i[3]*i[4] for i in temp ])

		num_init = [0.0 for s in range(self.nb_states)]
		for i in temp:

			for s in range(self.nb_states):
				num_init[s] += i[5][s]*i[4]

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.nb_states):
				a[s][x] = sum([i[1][s][x] for i in temp])

			for x in range(len(self.h.observations())):
				b[s][x] = sum([i[2][s][x] for i in temp])

		new_states = []
		for s in range(self.nb_states):
			la = [ correct_proba([a[s][i]/den[s] for i in range(self.nb_states)]) , list(range(self.nb_states))]
			lb = [ correct_proba([b[s][i]/den[s] for i in range(len(self.h.observations()))]) , self.h.observations()]

			new_states.append(HMM_state(lb,la))

		initial_state = [num_init[s]/sum(num_init) for s in range(self.nb_states)]
		
		return [HMM(new_states,initial_state),currentloglikelihood]