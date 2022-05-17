from .MC import *
from ..base.BW import *
from ..base.tools import getAlphabetFromSequences
from numpy import log

class BW_MC(BW):
	"""
	class for general Baum-Welch algorithm on MC.
	"""
	def __init__(self):
		super().__init__()
	
	def fit(self, traces: list, initial_model: bool=None, nb_states: int=None,
			random_initial_state: bool=False, output_file: str=None,
			epsilon: float=0.01, pp: str=''):
		"""
		Fits the model according to ``traces``.

		Parameters
		----------
		traces : list
			training set.
		initial_model : MC, optional.
			first hypothesis. If not set it will create a random MC with
			``nb_states`` states. Should be set if ``nb_states`` is not set.
		nb_states: int
			If ``initial_model`` is not set it will create a random MC with
			``nb_states`` states. Should be set if ``initial_model`` is not set.
			Default is None.
		random_initial_state: bool
			If ``initial_model`` is not set it will create a random MC with
			random initial state according to this sequence of probabilities.
			Should be set if ``initial_model`` is not set.
			Default is False.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		pp : str, optional
			Will be printed at each iteration. By default ''

		Returns
		-------
		MC
			fitted MC.
		"""
		if not initial_model:
			if not nb_states:
				print("Either nb_states or initial_model should be set")
				return
			initial_model = MC_random(nb_states,getAlphabetFromSequences(traces),random_initial_state)
		self.alphabet = initial_model.observations()
		return super().fit(traces, initial_model, output_file, epsilon, pp)

	def _processWork(self,sequence,times):
		alpha_matrix = self.computeAlphas(sequence)
		beta_matrix = self.computeBetas(sequence)
		proba_seq = alpha_matrix.T[-1].sum()
		if proba_seq != 0.0:
			####################
			den = zeros(self.nb_states)
			####################
			num = zeros(shape=(self.nb_states,self.nb_states*len(self.alphabet)))
			for s in range(self.nb_states):
				den[s] = dot(alpha_matrix[s][:-1]*beta_matrix[s][:-1],times/proba_seq).sum()
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
			l = [ tau[s]/den[s] , list_sta, list_obs ]
			new_states.append(MC_state(l,s))
		initial_state = [lst_init[s].sum()/lst_init.sum() for s in range(self.nb_states)]
		return [MC(new_states,initial_state),currentloglikelihood]
		