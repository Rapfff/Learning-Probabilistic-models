import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.CTMC import *
from learning.BW_CTMC import *

NB_PROCESS = 11

class BW_CTMC_Composition(BW_CTMC):
	def __init__(self,h1: CTMC, h2: CTMC) -> None:
		# h1 and h2 don't have any self-loop
		self.nb_states_1 = len(h1.states)
		self.nb_states_2 = len(h2.states)
		super().__init__(parallelComposition(h1,h2))

	def _getStateInComposition(self,s1,s2):
		return s1*self.nb_states_2+s2

	def _getState1(self,s):
		return s//self.nb_states_2

	def _getState2(self,s):
		return s%self.nb_states_1

	def _getObs(self,i: int) -> str:
		return self.alphabet[i%len(self.alphabet)]

	def _getTransitionSmallModel(self,s,model=2):
		s1 = self._getState1(s)
		s2 = self._getState2(s)
		res = []
		if model == 2:
			for ss2 in [j for j in range(self.nb_states_2) if j != s2]:
				for o in self.alphabet:
					res.append((self._getStateInComposition(s1,ss2),o))
		else:
			for ss1 in [j for j in range(self.nb_states_1) if j != s1]:
				for o in self.alphabet:
					res.append((self._getStateInComposition(ss1,s2),o))
		return res
	
	def processWork(self,sequence: list, times: int):
		times_seq, obs_seq = self.splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True

		alpha_matrix = self.computeAlphas(obs_seq)
		beta_matrix  = self.computeBetas(obs_seq)
		
		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq != 0.0:
			####################
			den = []
			num = []
			for s in range(self.nb_states):
				den.append(0.0)
				num.append([0.0 for i in range(self.len_to_update)])
				
				for t in range(len(obs_seq)):
					if timed:
						den[-1] += times_seq[t]*alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
					else:
						den[-1] += alpha_matrix[s][t]*beta_matrix[s][t]*times/proba_seq
					
					observation = obs_seq[t]
					for ss in range(len(self.to_update[s])//len(self.alphabet)):
						dest = self.to_update[s][ss*len(self.alphabet)][0]
						p = self.h_tau(s,dest,observation)
						if p != 0.0:
							if timed:
								num[-1][ss*len(self.alphabet)+self.alphabet.index(observation)] += alpha_matrix[s][t]*p*beta_matrix[dest][t+1]*times/proba_seq
							else:
								num[-1][ss*len(self.alphabet)+self.alphabet.index(observation)] += self.h_e(s)*alpha_matrix[s][t]*p*beta_matrix[dest][t+1]*times/proba_seq
							
			####################
			num_init = [alpha_matrix[s][0]*beta_matrix[s][0]*times/proba_seq for s in range(self.nb_states)]
			return [den, num, proba_seq, times, num_init]
		return False

	def generateHhat(self,traces: list) -> list:
		den = []
		for s in range(self.nb_states):
			den.append(0.0)
		tau = []
		for s in range(self.nb_states):
			tau.append([0 for i in range(self.len_to_update)])
		
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
		
		temp = [res.get() for res in tasks if res.get() != False]
		currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

		num_init = [0.0 for s in range(self.nb_states)]
		for i in temp:
			for s in range(self.nb_states):
				num_init[s] += i[4][s]

		for s in range(self.nb_states):
			den[s] = sum([i[0][s] for i in temp])
				
			for x in range(self.len_to_update):
				tau[s][x] = sum([i[1][s][x] for i in temp])

		list_sta = []
		for s in range(self.nb_states):
			list_sta.append([self.to_update[s][i][0] for i in range(self.len_to_update)])
		list_obs = self.alphabet*(self.nb_states_2-1)
		new_states = []
		for s in range(self.nb_states):
			l = [self._newProbabilities(tau[s],den[s],s), list_sta[s], list_obs]
			for k in self.to_keep[s]:
				l[0].append(self.h_tau(s,k[0],k[1]))
				l[1].append(k[0])
				l[2].append(k[1])
			new_states.append(CTMC_state(l))

		initial_state = [num_init[s]/sum(traces[1]) for s in range(self.nb_states)]

		return [CTMC(new_states,initial_state),currentloglikelihood]

	
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
			self.to_update = [self._getTransitionSmallModel(s,2+counter%2) for s in range(self.nb_states)]
			self.len_to_update = len(self.to_update[0])
			self.to_keep = [self._getTransitionSmallModel(s,1+counter%2) for s in range(self.nb_states)]
			
			self.hhat, currentloglikelihood = self.generateHhat(traces)
			
			counter += 1
			self.h = self.hhat
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				
		self.h.save(output_file)
		return self.h
