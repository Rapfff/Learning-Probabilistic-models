import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.MC import *
from learning.BW import *
from multiprocessing import cpu_count, Pool
from tools import correct_proba

class BW_MCGT(BW):
	def __init__(self,initial_model):
		"""
		h is a MCGT
		"""
		super().__init__(initial_model)

	def processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq != 0.0:
			####################
			den = []
			for s in range(self.nb_states):
				den.append(0.0)
				for t in range(len(sequence)):
					den[-1] += alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
			####################
			num = []
			for s in range(self.nb_states):
				num.append([0.0 for i in range(self.nb_states*len(self.h.observations()))])
				for t in range(len(sequence)):
					observation = sequence[t]
					for ss in range(self.nb_states):
						p = 0.0
						for i in range(len(self.h.states[s].next_matrix[0])):
							if self.h.states[s].next_matrix[1][i] == ss and self.h.states[s].next_matrix[2][i] == observation:
								p = self.h.states[s].next_matrix[0][i]
								break
						if p != 0.0:
							num[-1][ss*len(self.h.observations())+self.h.observations().index(observation)] += alpha_matrix[s][t]*p*beta_matrix[ss][t+1]*times/proba_seq
			####################
			num_init = [alpha_matrix[s][0]*beta_matrix[s][0]*times/proba_seq for s in range(self.nb_states)]
			return [den,num, proba_seq,times,num_init]
		return False

	def generateHhat(self,traces):
		den = []
		for s in range(self.nb_states):
			den.append(0.0)
		tau = []
		for s in range(self.nb_states):
			tau.append([0 for i in range(self.nb_states*len(self.h.observations()))])
		
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

		num_init = [0.0 for s in range(self.nb_states)]
		for i in temp:
			for s in range(self.nb_states):
				num_init[s] += i[4][s]*i[3]

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.nb_states*len(self.h.observations())):
				tau[s][x] = sum([i[1][s][x] for i in temp])

		list_sta = []
		for i in range(self.nb_states):
			for o in self.h.observations():
				list_sta.append(i)
		list_obs = self.h.observations()*self.nb_states
		new_states = []
		for s in range(self.nb_states):
			l = [ correct_proba([tau[s][i]/den[s] for i in range(len(list_sta))]) , list_sta, list_obs ]
			new_states.append(MC_state(l))

		initial_state = [num_init[s]/sum(num_init) for s in range(self.nb_states)]

		return [MC(new_states,initial_state),currentloglikelihood]