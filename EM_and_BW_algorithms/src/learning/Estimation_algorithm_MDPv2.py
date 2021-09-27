import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.MDP import *
from tools import correct_proba
from random import randint
from multiprocessing import cpu_count, Pool
from time import time
import datetime

class Estimation_algorithm_MDPv2:
	def __init__(self,h,alphabet,actions):
		"""
		h is a MDP
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.hhat = h
		self.alphabet = alphabet
		self.actions = actions
		self.nb_states = len(self.h.states)

	def h_g(self,s1,act,s2,obs):
		return self.h.g(s1,act,s2,obs)

	def hhat_g(self,s1,act,s2,obs):
		return self.hhat.g(s1,act,s2,obs)
	
	def computeForwardBackard(self,sequence,sequence_actions,algo,common=None,alpha_matrix=None,prev_len=None):
		if algo == "alpha":
			return self.computeAlphas(sequence,sequence_actions,common,alpha_matrix,prev_len)
		elif algo == "beta":
			return self.computeBetas(sequence,sequence_actions)

	def computeAlphas(self,sequence,sequence_actions,common,alpha_matrix,prev_len):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		diff = len(sequence) - prev_len
		if diff > 0:
			for s in range(self.nb_states):
				for i in range(diff):
					alpha_matrix[s].append(None)
		if diff < 0:
			for s in range(self.nb_states):
				alpha_matrix[s] = alpha_matrix[s][:diff]

		for k in range(common,len(sequence)):
			action = sequence_actions[k]
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(ss,action,s,sequence[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ

		return ["alpha",alpha_matrix]

	def computeBetas(self,sequence,sequence_actions):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []
		for s in range(self.nb_states):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			action = sequence_actions[k]
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.h_g(s,action,ss,sequence[k])
					summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ)

		return ["beta",beta_matrix]

	def computeDen(self,s,alpha_matrix,beta_matrix,proba_seq):
		res = 0.0
		for t in range(len(alpha_matrix)):
			res += alpha_matrix[t]*beta_matrix[s]
		res /= proba_seq
		return [s, res]

	def computeNum(self,s,ss,o,alpha_matrix,beta_matrix,sequence_actions,tau):
		res = [0.0 for a in self.actions]
		for t in range(len(alpha_matrix)):
			res[self.actions.index(sequence_actions[t])] += alpha_matrix[t]*tau.g(sequence_actions[t],ss,o)*beta_matrix[ss,t+1]
		return [s,ss,o, res]

	def learn(self,traces,output_file="output_model.txt",epsilon=0.01,pp=''):
		"""
		Given a set of sequences of pairs action-observation,
		it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		traces = [[trace1,trace2,...],[number_of_trace1,number_of_trace2,...]]
		trace = [action,obs1,action2,obs2,...,actionx,obsx]
		"""
		self.sequences_sorted = traces[0]
		self.sequences_sorted.sort()
		self.times = [ traces[1][traces[0].index(self.sequences_sorted[seq])] for seq in range(len(self.sequences_sorted))]

		start_time = time()
		
		self.observations = []
		for seq in self.sequences_sorted:
			for i in range(1,len(seq),2):
				if not seq[i] in self.observations:
					self.observations.append(seq[i])

		counter = 0
		prevloglikelihood = 10
		while True:
			#print(datetime.datetime.now(),pp,counter, prevloglikelihood)
			den = [0 for i in range(self.nb_states)]
			tau = [{[0 for i in range(self.nb_states*len(self.observations))] for a in self.actions} for s in range(self.nb_states)]
			currentloglikelihood = 0
			
			for seq in range(len(self.sequences_sorted)):
				sequence_actions = [self.sequences_sorted[seq][i] for i in range(0,len(self.sequences_sorted[seq]),2)]
				sequence_obs = [self.sequences_sorted[seq][i+1] for i in range(0,len(self.sequences_sorted[seq]),2)]

				if seq == 0:
					prev_len = len(sequence_obs)
					common = 0
					alpha_matrix = []

					for s in range(self.nb_states):
						if s == self.h.initial_state:
							alpha_matrix.append([1.0])
						else:
							alpha_matrix.append([0.0])
						alpha_matrix[-1] += [None for i in range(len(sequence_obs))]
				else:
					common = 0
					while common < min(len(self.sequences_sorted[seq-1]),len(self.sequences_sorted[seq])):
						if self.sequences_sorted[seq-1][common] == self.sequences_sorted[seq][common]:
							common += 1
						else:
							break

				#compute FORWARD-BACWARD
				p = Pool(processes = 2)
				tasks = []
				#temp = []
				for j in ["alpha","beta"]:
					tasks.append(p.apply_async(self.computeForwardBackard, [sequence_obs, sequence_actions, j, common, alpha_matrix, prev_len,]))
					#temp.append(self.ghatmultiple(i,j,k))
				p.close()
				temp = [res.get() for res in tasks]
				if temp[0][0] == "alpha":
					alpha_matrix = temp[0][1]
					beta_matrix  = temp[1][1]
				else:
					alpha_matrix = temp[1][1]
					beta_matrix  = temp[0][1]
				del temp
				#########

				proba_seq = self.beta_matrix[self.h.initial_state][0]
				if proba_seq > 0.0:
					currentloglikelihood += log(proba_seq)
					
					#compute DEN
					p = Pool(processes = min(cpu_count()-1,self.nb_states))
					tasks = []
					#temp = []
					for s in range(self.nb_states):
						tasks.append(p.apply_async(self.computeDen, [s, alpha_matrix[s], beta_matrix[s], proba_seq,]))
						#temp.append(self.ghatmultiple(i,j,k))
					p.close()
					temp = [res.get() for res in tasks]
					for i in temp:
						den[i[0]] += i[1]
					#########

					#compute TAU
					for s in range(self.nb_states):
						for ss in range(self.nb_states):
							p = Pool(processes = min(cpu_count()-1, len(self.observations)))
							tasks = []
							#temp = []
							for o in range(self.observations):
								tasks.append(p.apply_async(self.computeNum, [s, ss, o, alpha_matrix[s], beta_matrix[ss], sequence_actions, self.h.states[s],]))
								#temp.append(self.ghatmultiple(i,j,k))
							p.close()
							temp = [res.get() for res in tasks]
							for i in temp:
								for a in range(len(self.actions)):
									tau[i[0]][self.actions[a]][i[1]*len(self.observations)+self.observations.index(i[2])] += i[3][a]
					#########
			
			list_sta = []
			for i in range(self.nb_states):
				for o in self.observations:
					list_sta.append(i)
			list_obs = self.observations*self.nb_states
			new_states = []
			for s in range(self.nb_states):
				dic = {}
				for a in self.actions:
					dic[a] = [ [tau[s][a][i]/den[s] for i in range(len(list_sta))] , list_sta, list_obs ]
			
				new_states.append(MDP_state(dic))

			self.hhat = MDP(new_states,self.h.initial_state)
			
			counter += 1
			if abs(prevloglikelihood - currentloglikelihood) < epsilon:
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat

		self.h.save(output_file)
		print("Duration:", time()-start_time)
		return self.h
