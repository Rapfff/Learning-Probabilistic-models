import os, sys
from turtle import update
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.CTMC import *
from learning.BW_CTMC import *


class BW_CTMC_Composition(BW_CTMC):
	def __init__(self,h1: CTMC, h2: CTMC) -> None:
		# h1 and h2 don't have any self-loop
		self.hs = [None,h1,h2]
		self.nb_states_hs = [None,len(h1.states),len(h2.states)]
		super().__init__(parallelComposition(h1,h2))

	def _getStateInComposition(self,s:int,model:int,s2:int=0):
		if model == 1:
			return s*self.nb_states_hs[2]+s2
		else:
			return s2*self.nb_states_hs[2]+s

	def _getState1(self,s):
		return s//self.nb_states_hs[2]

	def _getState2(self,s):
		return s%self.nb_states_hs[1]
	
	def _getStates(self,s,model):
		if model == 1:
			return self._getStates1(s)
		else:
			return self._getStates2(s)

	def _getStates1(self,s):
		"""
		Return all the states in the composition that correspond to state
		s in model 1
		"""
		return [s*self.nb_states_hs[2]+i for i in range(self.nb_states_hs[2])]

	def _getStates2(self,s):
		"""
		Return all the states in the composition that correspond to state
		s in model 2
		"""
		return [i*self.nb_states_hs[2]+s for i in range(self.nb_states_hs[1])]


	def processWork(self,sequence: list, times: int, to_update: int):
		other = to_update%2 + 1
		nb_states = self.nb_states_hs[to_update]
		nb_states_other = self.nb_states_hs[other]

		times_seq, obs_seq = self.splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True

		alpha_matrix = self.computeAlphas(obs_seq)
		beta_matrix  = self.computeBetas(obs_seq)
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq <= 0.0:
			return False

		den = []
		num = []
		num_init = []
		for v in range(nb_states):
			den.append(0.0)
			num_init.append(0.0)
			num.append([0.0 for i in range(nb_states*len(self.alphabet))])
			
			ev = self.hs[to_update].e(v)
			
			for u in range(nb_states_other):
				eu = self.hs[other].e(u)
				e  = ev+eu
				uv = self._getStateInComposition(v,to_update,u)
				
				for t in range(len(obs_seq)):
					observation = obs_seq[t]
					if not timed:
						den[-1] += alpha_matrix[uv][t]*beta_matrix[uv][t]
					else:
						den[-1] += alpha_matrix[uv][t]*beta_matrix[uv][t]*times_seq[t]

					for vv in [i for i in range(nb_states) if i != v]:
						uvv = self._getStateInComposition(vv,to_update,u)
						num[-1][vv*len(self.alphabet)+self.alphabet.index(observation)] += alpha_matrix[uv][t]*self.hs[to_update].l(v,vv,observation)*beta_matrix[uvv][t+1]
				
				num[-1] = [i/e for i in num[-1]]
				if not timed:
					den[-1] /= e
				num_init[-1] += alpha_matrix[uv][0]*beta_matrix[uv][0]
			
			num[-1]  = [i*times/proba_seq for i in num[-1]]
			den[-1] *= times/proba_seq
			num_init[-1] *= times/proba_seq
		
		return [den, num, proba_seq, times, num_init]

	def generateHhat(self,traces: list, to_update: int) -> list:
		if to_update == 1:
			nb_states = self.nb_states_hs[1]
		else:
			nb_states = self.nb_states_hs[2]

		den = []
		for s in range(nb_states):
			den.append(0.0)
		tau = []
		for s in range(nb_states):
			tau.append([0 for i in range(nb_states*len(self.alphabet))])
		
		p = Pool(processes = NB_PROCESS)
		tasks = []
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq], to_update,]))
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

		num_init = [0.0 for s in range(nb_states)]
		for i in temp:
			for s in range(nb_states):
				num_init[s] += i[4][s]

		for s in range(nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(nb_states*len(self.alphabet)):
				tau[s][x] = sum([i[1][s][x] for i in temp])

		list_sta = []
		for i in range(nb_states):
			for o in self.alphabet:
				list_sta.append(i)
		list_obs = self.alphabet*nb_states
		new_states = []
		for s in range(nb_states):
			l = [self._newProbabilities(tau[s],den[s]), list_sta, list_obs]
			i = 0
			while i < len(l[0]):
				if l[0][i] == 0.0:
					l[0] = l[0][:i]+l[0][i+1:]
					l[1] = l[1][:i]+l[1][i+1:]
					l[2] = l[2][:i]+l[2][i+1:]
					i -= 1
				i += 1
			if l[0][-1] == 0.0:
				l[0] = l[0][:-1]
				l[1] = l[1][:-1]
				l[2] = l[2][:-1]
					
			new_states.append(CTMC_state(l))

		initial_state = [num_init[s]/sum(traces[1]) for s in range(nb_states)]

		return [CTMC(new_states,initial_state),currentloglikelihood]

	def _learnOneModel(self,traces,to_update,epsilon=0.01,verbose=False,pp='') -> int:
		counter = 0
		prevloglikelihood = 10
		nb_traces = sum(traces[1])
		while True:
			if verbose:
				print(datetime.now(),pp,counter, prevloglikelihood/nb_traces,to_update)
			hhat, currentloglikelihood = self.generateHhat(traces,to_update)
			if to_update == 2:
				self.hs[2] = hhat
			else:
				self.hs[1] = hhat
			
			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = parallelComposition(self.hs[1],self.hs[2])
		return counter

	def learn(self,traces,output_file=None,epsilon=0.01,verbose=False,pp='',to_update=None,limit_min_iteration=10):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [obs1,obs2,...,obsx]
		"""
		if to_update:
			self._learnOneModel(traces,to_update,epsilon,verbose,pp)

		else:
			counter = limit_min_iteration + 1
			to_update = 1
			while counter > limit_min_iteration:
				to_update = 1 +(to_update)%2
				if verbose:
					print(datetime.now(),":","Updating model",to_update,"...")
				counter = self._learnOneModel(traces,to_update,epsilon,verbose,pp)
				if verbose:
					print(counter,"iterations.")
		if output_file:
			self.hs[1].save(output_file+"_1.txt")
			self.hs[2].save(output_file+"_2.txt")
		return self.hs[1], self.hs[2]
