import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.HMM import *
from learning.BW import *
from multiprocessing import cpu_count, Pool
from time import time
from tools import correct_proba
import datetime

class BW_HMM(BW):
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
			num_b = []
			for s in range(self.nb_states):
				num_a.append([0.0 for i in range(self.nb_states)])
				num_b.append([0.0 for i in range(self.observations)])
				for t in range(len(sequence)):
					observation = sequence[t]
					for ss in range(self.nb_states):
						num_a[ss] += alpha_matrix[s][t]*self.h.a(s,ss)*self.h.b(s,observation)*beta_matrix[ss][t+1]
						num_b[self.observations.index(observation)] += alpha_matrix[s][t]*beta_matrix[s][t]
			####################
			return [den,num_a,num_b,proba_seq,times]
		return False

	def generateHhat(self,traces):
		den = []
		for s in range(self.nb_states):
			den.append(0.0)
		a = []
		for s in range(self.nb_states):
			a.append([0 for i in range(self.nb_states)])
		b = []
		for s in range(len(self.observations)):
			b.append([0 for i in range(self.observations)])
		
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[3])*i[4] for i in temp])

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.nb_states):
				a[s][x] = sum([i[1][s][x] for i in temp])

			for x in range(len(self.observations)):
				b[s][x] = sum([i[2][s][x] for i in temp])

		new_states = []
		for s in range(self.nb_states):
			la = [ correct_proba([a[s][i]/den[s] for i in range(self.nb_states)]) , list(range(self.nb_states))]
			lb = [ correct_proba([b[s][i]/den[s] for i in range(len(self.observations))]) , self.observations]
			new_states.append(HMM_state(lb,la))

		return [HMM(new_states,self.h.initial_state),currentloglikelihood]

