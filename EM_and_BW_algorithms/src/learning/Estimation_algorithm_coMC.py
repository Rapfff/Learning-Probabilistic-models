import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.coMC import *
from multiprocessing import cpu_count, Pool
from time import time
from tools import correct_proba
import datetime

NB_PROCESS = cpu_count()-1

class Estimation_algorithm_coMC:
	def __init__(self,h):
		"""
		h is a coMC
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.hhat = h
		self.nb_states = len(self.h.states)

	def h_g(self,s1,s2,obs):
		return self.h.g(s1,s2,obs)

	def computeAlphas(self,sequence):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		alpha_matrix = []
		for i in range(self.nb_states):
			if i == self.h.initial_state:
				alpha_matrix.append([1.0])
			else:
				alpha_matrix.append([0.0])

		for k in range(len(sequence)):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(ss,s,sequence[k])
					if p > 0.0:
						summ += alpha_matrix[ss][k]*p
				alpha_matrix[s].append(summ)
		return alpha_matrix

	def computeBetas(self,sequence):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(s,ss,sequence[k])
					if p > 0.0:
						summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ)

		return beta_matrix

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
			num    = []
			den2   = []
			num_mu = []
			num_d  = []
			for s in range(self.nb_states):
				num.append(   [0.0 for i in range(self.nb_states)])
				den2.append(  [0.0 for i in range(self.nb_states)])
				num_mu.append([0.0 for i in range(self.nb_states)])
				num_d.append( [0.0 for i in range(self.nb_states)])

				for t in range(len(sequence)):
					for ss in range(self.nb_states):
						p = self.h_g(s,ss,sequence[t])
						if p != 0.0:
							num[-1][ss]    += alpha_matrix[s][t]*p*beta_matrix[ss][t+1]*times/proba_seq
							p *= alpha_matrix[s][t]*beta_matrix[ss][t+1]*times/proba_seq
							den2[-1][ss]   += p
							num_mu[-1][ss] += p*sequence[t]
							num_d[-1][ss]  += p*(sequence[t]-self.h.states[s].obs_matrix[ss][0])**2
			####################
			return [den,num,proba_seq,times,den2,num_mu,num_d]
		return False


	def learn(self,traces,output_file="output_model.txt",epsilon=0.01,pp=''):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [obs1,obs2,...,obsx]
		"""
		self.h.pprint()
		counter = 0
		prevloglikelihood = 10
		while True:
			print(datetime.datetime.now(),pp,counter, prevloglikelihood)
			den    = []
			a      = []
			den2   = []
			num_mu = []
			num_d  = []
			for s in range(self.nb_states):
				den.append(0.0)
				a.append([0 for i in range(self.nb_states)])
				den2.append([0 for i in range(self.nb_states)])
				num_mu.append([0 for i in range(self.nb_states)])
				num_d.append([0 for i in range(self.nb_states)])

			p = Pool(processes = NB_PROCESS)
			tasks = []
			
			for seq in range(len(traces[0])):
				tasks.append(p.apply_async(self.processWork, [traces[0][seq], traces[1][seq],]))
			
			temp = [res.get() for res in tasks if res.get() != False]
			currentloglikelihood = sum([log(i[2])*i[3] for i in temp])

			for s in range(self.nb_states):
				den[s] = sum([i[0][s] for i in temp])
				
				for ss in range(self.nb_states):
					a[s][ss]      = sum([i[1][s][ss] for i in temp])
					den2[s][ss]   = sum([i[4][s][ss] for i in temp])
					num_mu[s][ss] = sum([i[5][s][ss] for i in temp])
					num_d[s][ss]  = sum([i[6][s][ss] for i in temp])

			list_sta = [i for i in range(self.nb_states)]
			new_states = []
			for s in range(self.nb_states):
				l = [ correct_proba([a[s][i]/den[s] for i in range(len(list_sta))]) , list_sta]
				d = {}
				for ss in range(self.nb_states):
					if l[0][ss] > 0.00001:
						d[ss] = [num_mu[s][ss]/den2[s][ss], num_d[s][ss]/(2*den2[s][ss])]
				new_states.append(coMC_state(l,d))

			self.hhat = coMC(new_states,self.h.initial_state)
			#self.hhat.pprint()
			#print()
			
			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat

		self.h.save(output_file)
		return self.h
