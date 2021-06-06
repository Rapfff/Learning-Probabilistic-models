import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir  = os.path.dirname(parentdir1)
sys.path.append(parentdir)
from src.learning.Estimation_algorithm_MDP import Estimation_algorithm_MDP
from src.models.MDP import MemorylessScheduler, loadMDP
from src.tools import resolveRandom, mergeSets, saveSet
from multiprocessing import cpu_count, Pool
from examples.examples_models import scheduler_uniform
from random import random
from datetime import datetime
"""
class ActiveLearningScheduler:
	def __init__(self,memoryless_scheduler,m):
		self.m = m
		self.nb_states = len(self.m.states)
		self.reset()
		self.memoryless_scheduler = memoryless_scheduler
		
		
	def reset(self):
		self.seq_obs = []
		self.seq_act = []
		self.alpha_matrix = []

		for s in range(self.nb_states):
			if s == self.m.initial_state:
				self.alpha_matrix.append([1.0])
			else:
				self.alpha_matrix.append([0.0])

	def get_action(self):
		#return an action to execute by the agent
		if len(self.seq_act) != 0:
			for s in range(self.nb_states):
				self.alpha_matrix[s].append(None)
				
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.m.g(ss,self.seq_act[-1],s,self.seq_obs[-1])
					summ += self.alpha_matrix[ss][-2]*p
				self.alpha_matrix[s][-1] = summ
		
		t = [self.alpha_matrix[s][-1] for s in range(self.nb_states)]
		tot = sum(t)
		if tot <= 0.0:
			t = [1/len(t) for i in t]
		else:
			t = [i/tot for i in t]
		s_i = resolveRandom(t)
		act = self.memoryless_scheduler.get_action(s_i)
		self.seq_act.append(act)
		return act

	def add_observation(self,obs):
		self.seq_obs.append(obs)
"""

class ActiveLearningScheduler:
	def __init__(self,m,matrix):
		self.m = m
		self.nb_states = len(m.states)
		self.matrix = matrix
		self.reset()

	def reset(self):
		self.seq_obs = []
		self.seq_act = []
		self.alpha_matrix = []

		for s in range(self.nb_states):
			if s == self.m.initial_state:
				self.alpha_matrix.append([1.0])
			else:
				self.alpha_matrix.append([0.0])
	
	def get_action(self):
		#return an action to execute by the agent
		if len(self.seq_act) != 0:
			for s in range(self.nb_states):
				self.alpha_matrix[s].append(None) 
				
			for s in range(self.nb_states):
				summ = 0.0
				for ss in range(self.nb_states):
					p = self.m.g(ss,self.seq_act[-1],s,self.seq_obs[-1])
					summ += self.alpha_matrix[ss][-2]*p
				self.alpha_matrix[s][-1] = summ
		
		t = [self.alpha_matrix[s][-1] for s in range(self.nb_states)]
		tot = sum(t)
		if tot <= 0.0:
			t = [1/len(t) for i in t]
		else:
			t = [i/tot for i in t]
		s_i = resolveRandom(t)
		act = self.m.actions()[resolveRandom(self.matrix[s_i])]
		self.seq_act.append(act)
		return act

	def add_observation(self,obs):
		self.seq_obs.append(obs) 

class Active_Learning_MDP:
	def __init__(self,h,alphabet,actions):
		"""
		h is a MDP
		alphabet is a list of the possible observations (list of strings)
		"""
		self.algo  = Estimation_algorithm_MDP(h,alphabet,actions)
		self.nb_states = len(h.states)

	def learn(self,traces,lr,nb_sequences,max_iteration,epsilon_greedy=0.9,output_folder="active_learning_models",epsilon=0.01,number_steps=None,pp=''):
		self.probas_states = [0.0 for i in range(self.nb_states)]
		total_traces = traces
		
		if number_steps == None:
			number_steps = int(len(traces[0][0])/2)
		elif type(number_steps) == list:
			if len(number_steps) != max_iteration or len(number_steps[0]) != nb_sequences:
				return None

		self.algo.h = loadMDP(output_folder+"/model_0.txt")
		#self.algo.learn(traces,output_folder+"/model_0.txt",epsilon,pp)

		c = 1
		sample_active = []
		while c <= max_iteration :
			print(datetime.now(),lr, epsilon_greedy, pp, c)
			#self.algo.h.pprint()
			if type(number_steps) == int:
				traces = self.addTraces(number_steps,nb_sequences,total_traces,epsilon_greedy)
			else:
				traces = self.addTraces(number_steps[c-1],nb_sequences,total_traces,epsilon_greedy)
			total_traces = mergeSets(total_traces,traces)
			for i in range(len(traces[0])):
				for j in range(traces[1][i]):
					sample_active.append(traces[0][i])

			if lr == "dynamic":
				old_h = self.algo.h
				lr_it = sum(traces[1])/(sum(total_traces[1])-sum(traces[1]))
				self.algo.learn(traces,output_folder+"/temp.txt",epsilon,str(c))
				self.mergeModels(old_h,self.algo.h,lr_it)
				self.algo.h.save(output_folder+"/active_models_"+str(c)+".txt")
			
			elif lr == 0:
				self.algo.learn(total_traces,output_folder+"/active_models_"+str(c)+".txt",epsilon,str(c))
			
			elif type(lr) == type(0.2):
				old_h = self.algo.h
				self.algo.learn(traces,output_folder+"/temp.txt",epsilon,str(c))
				self.mergeModels(old_h,self.algo.h,lr)
				self.algo.h.save(output_folder+"/active_models_"+str(c)+".txt")

			c += 1
		
		fout = open(output_folder+"/sample.txt",'w')
		for seq in sample_active:
			fout.write(str(seq)+'\n')
		fout.close()		
		print()
		return self.algo.h


	def mergeModels(self,old_h,new_h,lr):
		for s in range(self.nb_states):
			for a in new_h.states[s].actions():
				if new_h.states[s].next_matrix[a] == [1.0/len(new_h.states[s].next_matrix[a][0])]:
					if a in old_h.states[s].actions():
						self.algo.h.states[s].next_matrix[a] = old_h.states[s].next_matrix[a]
					elif a in self.algo.h.states[s].actions():
						self.algo.h.states[s].pop(a)

				elif a in old_h.states[s].actions():
					
					for p in range(len(new_h.states[s].next_matrix[a][0])):
						o = new_h.states[s].next_matrix[a][2][p]
						sprime = new_h.states[s].next_matrix[a][1][p]
						self.algo.h.states[s].next_matrix[a][0][p] = (1-lr)*old_h.states[s].g(a,sprime,o)+lr*new_h.states[s].g(a,sprime,o)

			for a in [i for i in old_h.states[s].actions() if not i in new_h.states[s].actions()]:
				self.algo.h.states[s].next_matrix[a] = old_h.states[s].next_matrix[a]

	def addTraces(self,number_steps,nb_sequences,traces,epsilon_greedy):
		memoryless_scheduler = strategy(self.algo.h,traces)
		scheduler_exploit = ActiveLearningScheduler(memoryless_scheduler,self.algo.h)
		scheduler_explore = scheduler_uniform(self.algo.h.actions())

		traces = [[],[]]
		for n in range(nb_sequences):
			if type(number_steps) == int:
				seq_len = number_steps
			elif type(number_steps) == list:
				seq_len = number_steps[n]
			
			if random() > epsilon_greedy:
				seq = self.algo.h.run(seq_len,scheduler_explore)
			else:
				seq = self.algo.h.run(seq_len,scheduler_exploit)

			if not seq in traces[0]:
				traces[0].append(seq)
				traces[1].append(0)
			traces[1][traces[0].index(seq)] += 1
		return traces

def strategy(m,traces):
	#ma solution
	nb_states = len(m.states)

	p = Pool(processes = cpu_count()-1)
	tasks = []
	for seq in range(len(traces[0])):
		tasks.append(p.apply_async(computeProbas, [m,traces[0][seq],traces[1][seq],]))
	p.close()
	temp = [res.get() for res in tasks]

	scheduler = []

	for s in range(nb_states):
		ss = []
		for a in range(len(m.actions())):
			ss.append(sum( [ temp[t][s][a] for t in range(len(temp)) ] ))

		scheduler.append(ss)
		#scheduler.append(m.actions()[ss.index(min(ss))])

	return ActiveLearningScheduler(m,scheduler)
	#return MemorylessScheduler(scheduler)

def computeProbas(m,seq,time):
	#ma solution
	nb_states = len(m.states)
	sequence_actions = [seq[i] for i in range(0,len(seq),2)]
	sequence_obs = [seq[i+1] for i in range(0,len(seq),2)]

	alpha_matrix = []
	for s in range(nb_states):
		if s == m.initial_state:
			alpha_matrix.append([1.0])
		else:
			alpha_matrix.append([0.0])
		alpha_matrix[-1] += [None for i in range(len(sequence_obs))]
	for k in range(len(sequence_obs)):
		action = sequence_actions[k]
		for s in range(nb_states):
			summ = 0.0
			for ss in range(nb_states):
				p = m.g(ss,action,s,sequence_obs[k])
				summ += alpha_matrix[ss][k]*p
			alpha_matrix[s][k+1] = summ

	res = [ [0.0 for a in m.actions()] for i in range(nb_states) ]

	for k in range(len(sequence_actions)):
		tot = sum([alpha_matrix[s][k] for s in range(nb_states)])
		if tot <= 0.0:
			break
		fact = time/tot
		for s in range(nb_states):
			res[s][m.actions().index(sequence_actions[k])] += alpha_matrix[s][k] * fact
	return res