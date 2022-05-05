import os, sys
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

	def _oneSequence(self,obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,to_update,proba_seq) -> list:
		other = to_update%2 + 1
		nb_states = self.nb_states_hs[to_update]
		nb_states_other = self.nb_states_hs[other]

		den = []
		num = []
		num_init = []
		for v in range(nb_states):
			den.append(0.0)
			num_init.append(0.0)
			num.append([0.0 for i in range(nb_states*len(self.alphabet))])
			
			ev = self.hs[to_update].e(v)
			divider = ev

			for u in range(nb_states_other):
				uv = self._getStateInComposition(v,to_update,u)
				eu = self.hs[other].e(u)
				if not self.disjoints_alphabet:
					divider = eu+ev

				for t in range(len(obs_seq)):
					observation = obs_seq[t]
					if timed:
						den[-1] += alpha_matrix[uv][t]*beta_matrix[uv][t]*times_seq[t]
					else:
						den[-1] += alpha_matrix[uv][t]*beta_matrix[uv][t]/(eu+ev)
					if observation in self.alphabets[to_update]:
						for vv in [i for i in range(nb_states) if i != v]:
							uvv = self._getStateInComposition(vv,to_update,u)
							num[-1][vv*len(self.alphabets[to_update])+self.alphabets[to_update].index(observation)] += alpha_matrix[uv][t]*beta_matrix[uvv][t+1]*self.hs[to_update].l(v,vv,observation)/divider
			
				num_init[-1] += alpha_matrix[uv][0]*beta_matrix[uv][0]
			
			num[-1]  = [i*times/proba_seq for i in num[-1]]
			den[-1] *= times/proba_seq
			num_init[-1] *= times/proba_seq
		return [den, num, num_init]

	def computeAlphasBetas(self,obs_seq, times_seq):
		if not self.disjoints_alphabet:
			return self.computeAlphas(obs_seq, times_seq), self.computeBetas(obs_seq, times_seq)
		else:
			obs_seqs, times_seq = self._splitSequenceObs(obs_seq, times_seq)
			bw = BW_CTMC(self.hs[1])
			alphas1 = bw.computeAlphas(obs_seqs[1], times_seq[0])
			betas1  = bw.computeBetas( obs_seqs[1], times_seq[0])
			bw = BW_CTMC(self.hs[2])
			alphas2 = bw.computeAlphas(obs_seqs[2], times_seq[1])
			betas2  = bw.computeBetas( obs_seqs[2], times_seq[1])
			alpha_matrix = []
			beta_matrix = []
			for s1 in range(self.nb_states_hs[1]):
				for s2 in range(self.nb_states_hs[2]):
					alpha_matrix.append([alphas1[s1][obs_seqs[0][t]]*alphas2[s2][t-obs_seqs[0][t]] for t in range(len(obs_seq)+1)])
					beta_matrix.append( [ betas1[s1][obs_seqs[0][t]]* betas2[s2][t-obs_seqs[0][t]] for t in range(len(obs_seq)+1)])
			return alpha_matrix, beta_matrix


	def processWork(self,sequence: list, times: int, to_update: int):
		times_seq, obs_seq = self.splitTime(sequence)
		if times_seq == None:
			timed = False
		else:
			timed = True
		alpha_matrix, beta_matrix = self.computeAlphasBetas(obs_seq,times_seq)

		proba_seq = sum([alpha_matrix[s][-1] for s in range(self.nb_states)])
		if proba_seq <= 0.0:
			return False

		if to_update:
			res1 = self._oneSequence(obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,to_update,proba_seq)
			res2 = None
		else:
			res1 = self._oneSequence(obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,1,proba_seq)
			res2 = self._oneSequence(obs_seq,times_seq,times,timed,alpha_matrix,beta_matrix,2,proba_seq)
	
		if timed:
			proba_seq = self.h.proba_one_timed_seq(sequence)
		
		return [res1, res2, proba_seq, times]

	def _generateModel(self,temp,nb_traces,to_update):
		#temp = [[den1,num1,num_init1],[den2,num2,num_init2]...]
		nb_states = self.nb_states_hs[to_update]
		alphabet = self.alphabets[to_update]
		den = [0.0 for s in range(nb_states)]
		num_init = [0.0 for s in range(nb_states)]
		tau = []
		for s in range(nb_states):
			tau.append([0 for i in range(nb_states*len(alphabet))])
		
		for i in temp:
			for s in range(nb_states):
				num_init[s] += i[2][s]
				den[s] += i[0][s]
				for x in range(nb_states*len(alphabet)):
					tau[s][x] += i[1][s][x]

		list_sta = []
		for i in range(nb_states):
			for o in alphabet:
				list_sta.append(i)
		list_obs = alphabet*nb_states
		new_states = []
		for s in range(nb_states):
			l = [self._newProbabilities(tau[s],den[s]), list_sta, list_obs]
			l = _removeZeros(l)					
			new_states.append(CTMC_state(l))

		initial_state = [num_init[s]/nb_traces for s in range(nb_states)]

		return CTMC(new_states,initial_state)

	def generateHhat(self,traces: list, to_update: int) -> list:
		p = Pool(processes = NB_PROCESS)
		tasks = []
		
		# temp = []
		# for seq in range(len(traces[0])):
		# 	temp.append(self.processWork(traces[0][seq], traces[1][seq], to_update))
		
		for seq in range(len(traces[0])):
			tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq], to_update,]))
		temp = [res.get() for res in tasks if res.get() != False]
		
		nb_traces = sum(traces[1])
		if to_update == 1:
			self.hs[1] = self._generateModel([i[0] for i in temp],nb_traces,1)
		elif to_update == 2:
			self.hs[2] = self._generateModel([i[0] for i in temp],nb_traces,2)
		else:
			self.hs[1] = self._generateModel([i[0] for i in temp],nb_traces,1)
			self.hs[2] = self._generateModel([i[1] for i in temp],nb_traces,2)

		currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

		return [parallelComposition(self.hs[1],self.hs[2]),currentloglikelihood]

	def learn(self,traces,output_file=None,epsilon=0.01,verbose=False,pp='',to_update=None):
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
		self.alphabets = [None,self.hs[1].observations(),self.hs[2].observations()]
		self.disjoints_alphabet = len(set(self.alphabets[1]).intersection(set(self.alphabets[2]))) == 0
		while True:
			if verbose:
				print(datetime.now(),pp,counter, prevloglikelihood/nb_traces,end='\r')
			hhat, currentloglikelihood = self.generateHhat(traces,to_update)

			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = hhat
		if output_file:
			self.hs[1].save(output_file+"_1.txt")
			self.hs[2].save(output_file+"_2.txt")
		if verbose:
			print()
		return self.hs[1], self.hs[2]

	def _splitSequenceObs(self,seq,tseq):
		res0 = [0]
		res1 = []
		res2 = []
		t1   = []
		t2   = []
		for i,o in enumerate(seq):
			res0.append(res0[-1])
			if o in self.alphabets[1]:
				res1.append(o)
				res0[-1] += 1
				if tseq:
					t1.append(tseq[i])
			elif o in self.alphabets[2]:
				res2.append(o)
				if tseq:
					t2.append(tseq[i])
			else:
				input("ERR0R: "+o+" is not in any alphabet")
		return ((res0,res1,res2),(t1,t2))

def _removeZeros(l):
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
	return l