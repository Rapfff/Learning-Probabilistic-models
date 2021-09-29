import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.MCGT import *
from tools import correct_proba
from time import time
import datetime
from multiprocessing import cpu_count, Pool
from math import log


class Estimation_algorithm_MCGT:
	def __init__(self,h,alphabet):
		"""
		h is a MCGT
		alphabet is a list of the possible observations (list of strings)
		"""
		self.h = h
		self.hhat = h
		self.alphabet = alphabet

	def computeAlphas(self, sequence, common, alpha_matrix):
		"""Here we compute all the values alpha(k,t) for a given sequence"""
		for k in range(common,len(sequence)):
			for s in range(len(self.h.states)):
				summ = 0.0
				for ss in range(len(self.h.states)):
					p = self.h.states[ss].g(s,sequence[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ

		return alpha_matrix

	def computeBetas(self, sequence):
		"""Here we compute all the values beta(t,k) for a given sequence"""
		beta_matrix = []

		for s in range(len(self.h.states)):
			beta_matrix.append([1.0])
		
		for k in range(len(sequence)-1,-1,-1):
			for s in range(len(self.h.states)):
				summ = 0
				for ss in range(len(self.h.states)):
					p = self.h.states[s].g(ss,sequence[k])
					if p > 0:
						summ += beta_matrix[ss][1 if ss<s else 0]*p
				beta_matrix[s].insert(0,summ)

		return beta_matrix

	def learn(self,sequences,output_file="output_model.txt",epsilon=0.01,pp=''):
		"""
		Given sequences of observations it adapts the parameters of h in order to maximize the probability to get 
		these sequences of observations.
		sequences = [[sequence1,sequence2,...],[number_of_seq1,number_of_seq2,...]]
		"""
		self.sequences = sequences[0]
		self.sequences.sort()
		self.times = [ sequences[1][sequences[0].index(self.sequences[seq])] for seq in range(len(self.sequences))]

		self.observed = []
		for i in self.sequences:
			for j in i:
				if not j in self.observed:
					self.observed.append(j)
		
		counter = 0
		prevloglikelihood = 10 #it contains the loglikelihood of the previous h
		while True:
			counter += 1
			#print(datetime.datetime.now(),pp,counter, prevloglikelihood)
			new_states = []
			for i in range(len(self.h.states)):
				next_probas = []
				next_states = []
				next_obs    = []
				
				p = Pool(processes = cpu_count()-1)
				tasks = []
				for j in range(len(self.h.states)):
					for k in self.observed:
						tasks.append(p.apply_async(self.ghatmultiple, [i,j,k,]))
				p.close()
				temp = [res.get() for res in tasks]

				next_probas = [ temp[t][2] for t in range(len(temp)) ]
				next_states = [ temp[t][0] for t in range(len(temp)) ]
				next_obs    = [ temp[t][1] for t in range(len(temp)) ]

				for j in temp:
					if len(j) == 4:
						currentloglikelihood = j[3]
				for j in range(len(self.h.states)):		
					for k in [ letter for letter in self.alphabet if not letter in self.observed ]:
						next_probas.append(0)
						next_states.append(j)
						next_obs.append(k)

				next_probas = correct_proba(next_probas)
				new_states.append(MCGT_state([ next_probas, next_states, next_obs]))

			self.hhat = MCGT(new_states,self.h.initial_state)
			
			if abs(prevloglikelihood -currentloglikelihood) < epsilon:#or self.checkEnd() #or time() - start_time > 120
				break
			else:
				prevloglikelihood = currentloglikelihood
				self.h = self.hhat
		
		self.h.save(output_file)
		return self.h

	def ghatmultiple(self,s1,s2,obs):
		num = 0.0
		den = 0.0

		if s1 == 0 and s2 == 0 and self.observed.index(obs) == 0:
			loglikelihood = 0.0
		else:
			loglikelihood = None

		for seq in range(len(self.sequences)):
			sequence = self.sequences[seq]
			times = self.times[seq]
			if seq == 0:
				common = 0
				alpha_matrix = []

				for s in range(len(self.h.states)):
					if s == self.h.initial_state:
						alpha_matrix.append([1.0])
					else:
						alpha_matrix.append([0.0])
					alpha_matrix[-1] += [None for i in range(len(sequence))]
			else:
				common = 0 
				while self.sequences[seq-1][common] == sequence[common]:
					common += 1
			alpha_matrix = self.computeAlphas(sequence,common,alpha_matrix)
			beta_matrix  = self.computeBetas(sequence)			

			bigK = beta_matrix[self.h.initial_state][0]

			for k in range(len(sequence)):
				gamma_s1_k = alpha_matrix[s1][k] * beta_matrix[s1][k] / bigK

				den += gamma_s1_k * times

				if sequence[k] == obs:
					num += alpha_matrix[s1][k]*self.h.states[s1].g(s2,obs)*beta_matrix[s2][k+1]*times/bigK

			if loglikelihood != None:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.h.states))]))

		if den == 0.0:
			#in this case we don't expect to reach s1 (except maybe at the end)
			#so we don't care of this value
			res = 0.0
		else:
			res = num/den

		if loglikelihood == None:
			return (s2,obs,res)
		else:
			return (s2,obs,res, loglikelihood)
