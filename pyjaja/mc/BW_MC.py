from .MC import *
from ..base.BW import *
from multiprocessing import Pool
from ..base.tools import correct_proba, getAlphabetFromSequences
from numpy import dot, zeros, log

class BW_MC(BW):
	def __init__(self):
		super().__init__()
	
	def fit(self, traces, initial_model=None, nb_states=0, random_initial_state=False, output_file="output_model.txt", epsilon=0.01, verbose=False, pp=''):
		if not initial_model:
			if nb_states == 0:
				print("ERROR")
				return
			initial_model = MC_random(nb_states,getAlphabetFromSequences(traces),random_initial_state)
		return super().fit(traces, initial_model, output_file, epsilon, verbose, pp)

	def _processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			####################
			den = dot(alpha_matrix*beta_matrix,times/proba_seq).sum(axis=1)
			####################
			num = zeros(shape=(self.nb_states,self.nb_states*len(self.alphabet)))
			for s in range(self.nb_states):
				c = 0
				for ss in range(self.nb_states):
					for obs in self.alphabet:
						arr_dirak = [1.0 if t == obs else 0.0 for t in sequence]
						p = array([self.h_tau(s,ss,o) if o == obs else 0.0 for o in sequence])
						num[s,c] = dot(alpha_matrix[s][:-1]*arr_dirak*beta_matrix[ss][1:]*p,times/proba_seq).sum()
						c += 1
			####################
			num_init = alpha_matrix.T[0]*beta_matrix.T[0]*times/proba_seq
			####################
			return [den,num, proba_seq,times,num_init]
		return False

	def _generateHhat(self,temp):
		den = zeros(shape=(self.nb_states,))
		tau = zeros(shape=(self.nb_states,self.nb_states*len(self.alphabet)))
		lst_den = array([i[0] for i in temp]).T
		lst_num = array([i[1] for i in temp]).T.reshape(self.nb_states*self.nb_states*len(self.alphabet),len(temp))
		lst_proba=array([i[2] for i in temp])
		lst_times=array([i[3] for i in temp])
		lst_init =array([i[4] for i in temp]).T

		currentloglikelihood = dot(log(lst_proba),lst_times)

		for s in range(self.nb_states):
			den[s] = lst_den[s].sum()
			for x in range(self.nb_states*len(self.alphabet)):
				tau[s,x] = lst_num[x*self.nb_states+s].sum()
		list_sta = []
		for i in range(self.nb_states):
			for _ in self.alphabet:
				list_sta.append(i)
		list_obs = self.alphabet*self.nb_states
		new_states = []
		for s in range(self.nb_states):
			l = [ correct_proba(tau[s]/den[s]) , list_sta, list_obs ]
			new_states.append(MC_state(l,s))
		initial_state = [lst_init[s].sum()/lst_init.sum() for s in range(self.nb_states)]
		return [MC(new_states,initial_state),currentloglikelihood]
		