from Estimation_algorithms_MDP_multithreading import Estimation_algorithm_MDP
from MDP import maxReachabilityScheduler
from tools import resolveRandom, mergeSets

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

	def learn(self,traces,omega,learning_rate,nb_sequences,max_iteration,output_file="output_model.txt",limit=0.01,pp=''):
		#self.history = []
		total_traces = traces
		number_steps = int(len(traces[0][0])/2)
		self.algo.problem3(traces,"active_models/active_models_0.txt",limit,pp)

		missing_info_states = self.findMissingInfoStates(omega,total_traces)
		print(len(missing_info_states),"missing_info_states")
		c = 1
		while len(missing_info_states) > 0 and  c < max_iteration :
			old_h = self.algo.h
			for s in missing_info_states:
				traces = self.addTraces(s,number_steps,nb_sequences)
			#self.history.append((len(missing_info_states)*nb_sequences)+sum(self.history))
			self.algo.problem3(traces,"active_models/active_models_"+str(c)+".txt",limit,str(c))
			self.mergeModels(old_h,self.algo.h,learning_rate)
			c += 1
			missing_info_states = self.findMissingInfoStates(omega,total_traces)
			print(len(missing_info_states),"missing_info_states")

		self.h = self.algo.h

	def mergeModels(self,old_h,new_h,lr):
		for s in range(len(new_h.states)):
			for a in new_h.states[s].actions():
				if a in old_h.states[s].actions():
					print(old_h.states[s].next_matrix[a],'\n',new_h.states[s].next_matrix[a])

					for p in range(len(new_h.states[s].next_matrix[a][0])):
						o = new_h.states[s].next_matrix[a][2][p]
						sprime = new_h.states[s].next_matrix[a][1][p]
						new_h.states[s].next_matrix[a][0][p] = (1-lr)*old_h.states[s].g(a,sprime,o)+lr*new_h.states[s].next_matrix[a][0][p]
					
					print(new_h.states[s].next_matrix[a])
					input()
			for a in [i for i in old_h.states[s].actions() if not i in new_h.states[s].actions()]:
				new_h.states[s].next_matrix[a] = old_h.states[s].next_matrix[a]
		self.algo.h = new_h

	def findMissingInfoStates(self,omega,traces):
		probas_states = self.computeProbas(traces)
		training_set_size = sum([traces[1][i]*(len(traces[0][i])+1)/2 for i in range(len(traces[0]))])
		return [ s for s in range(len(probas_states)) if probas_states[s] <= omega*training_set_size]

	def addTraces(self,s,traces,number_steps,nb_sequences):
		memoryless_scheduler = maxReachabilityScheduler(self.algo.h,s)
		scheduler = ActiveLearningScheduler(memoryless_scheduler,self.algo.h)
		traces = [[],[]]
		for n in range(nb_sequences):
			seq = self.algo.h.run(number_steps,scheduler)
			if not seq in traces[0]:
				traces[0].append(seq)
				traces[1].append(0)
			traces[1][traces[0].index(seq)] += 1
		return traces

	def computeProbas(self,traces): #mutlithread it
		sequences_sorted = traces[0].sort()
		times = [ traces[1][traces[0].index(sequences_sorted[seq])] for seq in range(len(sequences_sorted))]

		nb_states = len(self.algo.h.states)
		res = [0.0 for i in range(nb_states)]

		nb_seq = len(sequences_sorted)
		for seq in range(nb_seq):
			sequence_actions = [sequences_sorted[seq][i] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence_obs = [sequences_sorted[seq][i+1] for i in range(0,len(sequences_sorted[seq]),2)]
			if seq == 0:
				common = 0
				alpha_matrix = []

				for s in range(nb_states):
					if s == self.algo.h.initial_state:
						alpha_matrix.append([1.0])
					else:
						alpha_matrix.append([0.0])
					alpha_matrix[-1] += [None for i in range(len(sequence_obs))]
			else:
				common = 0 
				while sequences_sorted[seq-1][common] == sequences_sorted[seq][common]:
					common += 1
			
			alpha_matrix = self.algo.computeAlphas(sequence_obs,sequence_actions,int(common/2),alpha_matrix)
			beta_matrix  = self.algo.computeBetas(sequence_obs,sequence_actions)

			for s in range(nb_states):
				for k in range(len(alpha_matrix[s])):
					res[s] += alpha_matrix[s][k]*beta_matrix[s][k]*times[seq]/beta_matrix[self.algo.h.initial_state][0]

		return res