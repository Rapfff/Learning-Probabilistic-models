import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from learning.Estimation_algorithm_MDP import Estimation_algorithm_MDP
from models.MDP import MemorylessScheduler, loadMDP
from tools import resolveRandom, mergeSets
from multiprocessing import cpu_count, Pool

class ActiveLearningScheduler:
	def __init__(self,memoryless_scheduler,m):	
		self.seq_obs = []
		self.seq_act = []
		self.memoryless_scheduler = memoryless_scheduler
		self.m = m

	def reset(self):
		self.seq_obs = []
		self.seq_act = []

	def get_action(self):
		"""return an action to execute by the agent"""
		alpha_matrix = []
		nb_states = len(self.m.states)

		for s in range(nb_states):
			if s == self.m.initial_state:
				alpha_matrix.append([1.0])
			else:
				alpha_matrix.append([0.0])
			alpha_matrix[-1] += [None for i in range(len(self.seq_obs))]
			
		for k in range(len(self.seq_obs)):
			for s in range(nb_states):
				summ = 0.0
				for ss in range(nb_states):
					p = self.m.g(ss,self.seq_act[k],s,self.seq_obs[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ
		
		t = [alpha_matrix[s][-1] for s in range(nb_states)]
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

class Active_Learning_MDP:
	def __init__(self,h,alphabet,actions):
		"""
		h is a MDP
		alphabet is a list of the possible observations (list of strings)
		"""
		self.algo  = Estimation_algorithm_MDP(h,alphabet,actions)
		self.nb_states = len(h.states)

	def learn(self,traces,df,lr,nb_sequences,max_iteration,output_folder="active_learning_models",epsilon=0.01,pp=''):
		self.probas_states = [0.0 for i in range(self.nb_states)]
		total_traces = traces
		number_steps = int(len(traces[0][0])/2)

		self.algo.h = loadMDP(output_folder+"/model_0.txt")
		#self.algo.learn(traces,output_folder+"/model_0.txt",epsilon,pp)

		c = 1
		while c < max_iteration :
			self.algo.h.pprint()
			traces = self.addTraces(number_steps,nb_sequences,total_traces,df)
			total_traces = mergeSets(total_traces,traces)
			
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

	def addTraces(self,number_steps,nb_sequences,traces,df):
		memoryless_scheduler = strategy(self.algo.h,traces,df)
		scheduler = ActiveLearningScheduler(memoryless_scheduler,self.algo.h)
		traces = [[],[]]
		for n in range(nb_sequences):
			seq = self.algo.h.run(number_steps,scheduler)
			if not seq in traces[0]:
				traces[0].append(seq)
				traces[1].append(0)
			traces[1][traces[0].index(seq)] += 1
		return traces

def strategy(m,traces,l):
	nb_states = len(m.states)

	p = Pool(processes = cpu_count()-1)
	tasks = []
	for seq in range(len(traces[0])):
		tasks.append(p.apply_async(computeProbas, [m,traces[0][seq],traces[1][seq],]))
	p.close()
	temp = [res.get() for res in tasks]

	p = []
	for s in range(nb_states):
		p.append( sum([ g[s] for g in temp ]) )
	print("Probas qu'on passe par s dans le training set:",'\n',p)

	old_x = [0.0 for i in range(nb_states)]
	while True:
		memoryless_sched = [] 
		x = []
		for s in range(nb_states):
			t = []
			for a_i in range(len(m.states[s].actions())):
				a = m.states[s].actions()[a_i]
				t.append(0.0)
				for ss in range(nb_states):
					for o in m.observations():
						t[-1] += m.g(s,a,ss,o)*old_x[ss]

			memoryless_sched.append(m.states[s].actions()[t.index(min(t))])
			x.append(min(t)*l + p[s])
		if max([abs(x[i] - old_x[i]) for i in range(nb_states)]) < 0.01:
			break
		else:
			old_x = x
	print(x)
	print(memoryless_sched)
	input()
	return MemorylessScheduler(memoryless_sched)

def computeProbas(m,seq,time):
	#state based
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

	beta_matrix = []
	for s in range(nb_states):
		beta_matrix.append([1.0])
	for k in range(len(sequence_obs)-1,-1,-1):
		action = sequence_actions[k]
		for s in range(nb_states):
			summ = 0.0
			for ss in range(nb_states):
				p = m.g(s,action,ss,sequence_obs[k])
				summ += beta_matrix[ss][1 if ss<s else 0]*p
			beta_matrix[s].insert(0,summ)

	if beta_matrix[m.initial_state][0] != 0.0:
		res = []

		for s in range(nb_states):
			res.append(sum([alpha_matrix[s][k]*beta_matrix[s][k]*time/beta_matrix[m.initial_state][0] for k in range(len(alpha_matrix[0])) ]))

	else:
		res = [0.0 for i in range(nb_states)]

	return res
"""
def computeProbas(m,seq,time):
	#action based
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

	beta_matrix = []
	for s in range(nb_states):
		beta_matrix.append([1.0])
	for k in range(len(sequence_obs)-1,-1,-1):
		action = sequence_actions[k]
		for s in range(nb_states):
			summ = 0.0
			for ss in range(nb_states):
				p = m.g(s,action,ss,sequence_obs[k])
				summ += beta_matrix[ss][1 if ss<s else 0]*p
			beta_matrix[s].insert(0,summ)

	if beta_matrix[m.initial_state][0] != 0.0:
		tot = [0.0 for i in range(nb_states)]
		res = {}

		for a in m.actions():
			ii = [ i for i in range(len(sequence_actions)) if sequence_actions[i] == a]
			res[a] = []
			for s in range(nb_states):
				res[a].append(sum([ alpha_matrix[s][k]*beta_matrix[s][k]*time/beta_matrix[m.initial_state][0] for k in ii ]))
				tot[s] += res[a][-1]

		for s in range(nb_states):
			for a in m.actions():
				if res[a][s] != 0.0:
					res[a][s] /= tot[s]
	else:
		res = {}
		for a in m.actions():
			res[a]= [0.0 for i in range(nb_states)]

	return res
"""