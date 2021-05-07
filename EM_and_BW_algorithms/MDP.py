from tools import resolveRandom
from math import log
from functools import reduce
from operator import mul

class MemorylessScheduler:
	def __init__(self,actions):
		self.actions = actions

	def get_action(self,current_state):
		return self.actions[current_state]

class FiniteMemoryScheduler:
	def __init__(self,action_matrix,transition_matrix):
		"""
		action_matrix = {scheduler_state: [[proba1,proba2,...],[action1,action2,...]],
					     scheduler_state: [[proba1,proba2,...],[action1,action2,...]],
					     ...}
		transition_matrix = {obs1: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
							 obs2: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
							 ...}
		"""	
		self.s = 0
		self.action_matrix = action_matrix
		self.transition_matrix = transition_matrix

	def reset(self):
		self.s = 0

	def get_action(self):
		"""return an action to execute by the agent"""
		return self.action_matrix[self.s][1][resolveRandom(self.action_matrix[self.s][0])]

	def add_observation(self,obs):
		"""give to the scheduler the new observation seen by the agent"""
		if obs in self.transition_matrix:
			self.s = self.transition_matrix[obs][self.s]

	def get_actions(self,s=None):
		"""return the actions (and their probability) that the agent can execute now"""
		if s==None:
			return self.action_matrix[self.s]
		else:
			return self.action_matrix[s]

	def get_sequence_states(self,seq_obs):
		"""given a sequence of observations, return the sequence of states the scheduler goes through"""
		self.reset()
		res = [0]
		for o in seq_obs:
			self.add_observation(o)
			res.append(self.s)
		return res

	def get_probability(self,action,state=None):
		"""given a scheduler state and an action return the probability to execute this action in this state"""
		if state == None:
			state = self.s
		if not action in self.action_matrix[state][1]:
			return 0
		return self.action_matrix[state][0][self.action_matrix[state][1].index(action)]


class MDP_state:

	def __init__(self,next_matrix):
		"""
		next_matrix = {action1 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_obs,transition2_obs,...]],
					   action2 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_obs,transition2_obs,...]]
					   ...}
		"""
		for action in next_matrix:
			if round(sum(next_matrix[action][0]),2) < 1.0:
				print("Sum of the probabilies of the next_matrix should be 1.0 here it's ",sum(next_matrix[action][0]))
				return False
		self.next_matrix = next_matrix

	def next(self,action):
		if not action in self.next_matrix:
			print("ACTION",action,"is not available in state")
		c = resolveRandom(self.next_matrix[action][0])
		return [self.next_matrix[action][1][c],self.next_matrix[action][2][c]]

	def actions(self):
		return [i for i in self.next_matrix]

	def g(self,action,state,obs):
		if action not in self.actions():
			#print("Not",action,"in",self.actions())
			return 0.0
		for i in range(len(self.next_matrix[action][0])):
			if self.next_matrix[action][1][i] == state and self.next_matrix[action][2][i] == obs:
				return self.next_matrix[action][0][i]
		return 0.0

	def __str__(self):
		res = ""
		for action in self.next_matrix:
			res += str(action)
			res += '\n'
			for proba in self.next_matrix[action][0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.next_matrix[action][1]:
				res += str(state)+' '
			res += '\n'
			for obs in self.next_matrix[action][2]:
				res += str(obs)+' '
			res += '\n'
		res += "*\n"
		return res

class MDP:

	def __init__(self,states,initial_state,name="unknown MDP"):
		self.initial_state = initial_state
		self.states = states
		self.name = name

	def __str__(self):
		return self.name

	def save(self,file_path):
		f = open(file_path,'w')
		f.write(self.name)
		f.write('\n')
		f.write(str(self.initial_state))
		f.write('\n')
		for s in self.states:
			f.write(str(s))
		f.close()

	def saveTerminal(self):
		print(self.name)
		print(str(self.initial_state))
		for s in self.states:
			print(str(s))

	def actions(self):
		res = []
		for s in self.states:
			res += s.actions()
		res = list(set(res))
		res.sort()
		return res

	def actions_state(self,s):
		return self.states[s].actions()

	def observations(self):
		res = []
		for s in self.states:
			for act in s.actions():
				res += s.next_matrix[act][2]
		res = list(set(res))
		res.sort()
		return res

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0

	def g(self,s1,action,s2,obs):
		return self.states[s1].g(action,s2,obs)
			
	def run(self,number_steps,scheduler,with_action=True):
		#output = [self.states[self.initial_state].observation]
		res = []
		#actions = []
		current = self.initial_state
		scheduler.reset()

		current_len = 0

		while current_len < number_steps:
			action = scheduler.get_action()
			#actions.append(action)
			while action not in self.states[current].next_matrix:
				action = scheduler.get_action()

			if with_action:
				res.append(action)
			next_state, observation = self.states[current].next(action)

			#output.append(observation)
			res.append(observation)
			scheduler.add_observation(observation)

			current = next_state
			current_len += 1
		return res

	def pprint(self):
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='',end='')
			if i == self.initial_state:
				print("(initial_state)")
			else:
				print()
			for action in self.states[i].next_matrix:
				for j in range(len(self.states[i].next_matrix[action][0])):
					if self.states[i].next_matrix[action][0][j] > 0.0:
						print("s",i," - (",action,") -> s",self.states[i].next_matrix[action][1][j]," : ",self.states[i].next_matrix[action][2][j],' : ',self.states[i].next_matrix[action][0][j],sep='')
		print()

	#-------------------------------------------
	def logLikelihoodTraces(self,sequences):
		sequences_sorted = sequences[0]
		sequences_sorted.sort()
		loglikelihood = 0.0

		alpha_matrix = self.initAlphaMatrix(int(len(sequences[0][0])/2))
		for seq in range(len(sequences_sorted)):
			sequence_actions = [sequences_sorted[seq][i] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence_obs = [sequences_sorted[seq][i+1] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence = sequences_sorted[seq]
			times = sequences[1][sequences[0].index(sequence)]
			common = 0
			if seq > 0:
				while sequences_sorted[seq-1][common] == sequence[common]:
					common += 1
			common = int(common/2)
			alpha_matrix = self.computeAlphaMatrix(sequence_obs,sequence_actions,common,alpha_matrix)
			if sum([alpha_matrix[s][-1] for s in range(len(self.states))]) <= 0.0:
				print(sequences_sorted[seq])
				return None
			else:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.states))]))

		return loglikelihood/sum(sequences[1])

	def computeAlphaMatrix(self,sequence_obs,sequence_actions,common,alpha_matrix):
		for k in range(common,len(sequence_obs)):
			action = sequence_actions[k]
			for s in range(len(self.states)):
				summ = 0.0
				for ss in range(len(self.states)):
					p = self.states[ss].g(action,s,sequence_obs[k])
					summ += alpha_matrix[ss][k]*p
				alpha_matrix[s][k+1] = summ
		return alpha_matrix

	def initAlphaMatrix(self,len_seq):
		alpha_matrix = []
		for s in range(len(self.states)):
			if s == self.initial_state:
				alpha_matrix.append([1.0])
			else:
				alpha_matrix.append([0.0])
			alpha_matrix[-1] += [None for i in range(len_seq)]
		return alpha_matrix

	def probasSequences(self,sequences):
		#given sequences = [seq1,seq2...] /!\ all sequences should be pairwise different
		#return probas   = [prob_seq1,prob_seq2,...]
		sequences_sorted = sequences
		sequences_sorted.sort()
		alpha_matrix = self.initAlphaMatrix(int(len(sequences[0])/2))
		probas = []
		for seq in range(len(sequences_sorted)):
			sequence_actions = [sequences_sorted[seq][i] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence_obs = [sequences_sorted[seq][i+1] for i in range(0,len(sequences_sorted[seq]),2)]
			sequence = sequences_sorted[seq]
			common = 0
			if seq > 0:
				while sequences_sorted[seq-1][common] == sequence[common]:
					common += 1
			common = int(common/2)
			alpha_matrix = self.computeAlphaMatrix(sequence_obs,sequence_actions,common,alpha_matrix)
			probas.append(sum([alpha_matrix[s][-1] for s in range(len(self.states))]))
		return probas	
	#-------------------------------------------

	def saveToMathematica(self,output_file):
		f = open(output_file,'w',encoding="utf-8")
		f.write('L={"start","')
		f.write('","'.join(self.observations()))
		f.write('"};\n')
		f.write('A={"')
		f.write('","'.join(self.actions()))
		f.write('"};\n')
		f.write('S={')
		f.write(','.join([str(i) for i in range(len(self.states))]))
		f.write('};\n')
		f.write('iota={ {"start",'+str(self.initial_state)+'} -> 1};\n')

		f.write('tau={')
		flag = False
		for s1 in range(len(self.states)):
			for a in self.actions():
				if flag:
					f.write(',')
				flag = True
				f.write('\n\t{"'+str(a)+'",'+str(s1)+'} -> { ')
				if not a in self.actions_state(s1):
					f.write('{"error",'+str(s1)+'} -> 1')
					#add error with prob 0 for other states
					#add 0 probtransition for all other obs
				else:
					ss = ""
					for s2 in range(len(self.states)):
						for o in self.observations():
							ss += '{"'+str(o)+'",'+str(s2)+'} -> '+str(self.g(s1,a,s2,o))+', '
							#f.write('{"'+str(o)+'",'+str(s2)+'} -> '+str(self.g(s1,a,s2,o))+', ')
						#add error with 0 prob
					ss = ss[:-2]
					f.write(ss)
				f.write('}')
		f.write('\n};\n')

		f.write("M = MDP[S,iota,tau];")
		f.close()


def KLDivergence(m1,m2,test_set):
	pm1 = m1.probasSequences(test_set)
	tot_m1 = sum(pm1)
	pm2 = m2.probasSequences(test_set)
	res = 0.0
	for seq in range(len(test_set)):
		if pm2[seq] <= 0.0:
			print(test_set[seq])
			return None
		if tot_m1 > 0.0 and pm1[seq] > 0.0:
			res += (pm1[seq]/tot_m1)*log(pm1[seq]/pm2[seq])
	return res

def maxReachabilityScheduler(m,s):
	#Return the memoryless scheduler that maximizes the probability to reach 
	# state s in MDP m
	observations = m.observations()
	bad_states = [i for i in range(len(m.states))]
	bad_states.remove(s)
	good_states = [s]
	f = True
	while f:
		f = False
		for ss in bad_states:
			next_ss = False
			for o in observations:
				for a in m.states[ss].actions():
					for sss in good_states:
						if m.states[ss].g(a,sss,o) > 0:
							bad_states.remove(ss)
							good_states.append(ss)
							next_ss = True
							f = True
							break
					if next_ss:
						break
				if next_ss:
					break
	old_x = [0 if i!=s else 1 for i in range(len(m.states))]
	while True:
		x = []
		for ss in range(len(m.states)):
			if ss == s:
				x.append(1)
			elif ss in bad_states:
				x.append(0)
			else:
				t = [0.0 for i in range(len(m.states[ss].actions())) ]
				for a_i in range(len(m.states[ss].actions())):
					a = m.states[ss].actions()[a_i]
					for o in observations:
						for sss in good_states:
							t[a_i] += m.states[ss].g(a,sss,o)*old_x[sss]
				x.append(max(t))
		if max([abs(x[i] - old_x[i]) for i in range(len(x))]) < 0.001:
			break
		else:
			old_x = x

	memoryless_sched = [] 
	for ss in range(len(m.states)):
		t = [0.0 for i in range(len(m.states[ss].actions())) ]
		for a_i in range(len(m.states[ss].actions())):
			a = m.states[ss].actions()[a_i]
			for o in observations:
				for sss in good_states:
					t[a_i] += m.states[ss].g(a,sss,o)*old_x[sss]
		memoryless_sched.append(m.states[ss].actions()[t.index(max(t))])

	return MemorylessScheduler(memoryless_sched)

def loadMDP(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = int(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l:
		d = {}
		while l != "*\n":
			a = l[:-1]
			p = [float(i) for i in  f.readline()[:-2].split(' ')]
			s = [int(i) for i in  f.readline()[:-2].split(' ')]
			t = f.readline()[:-2].split(' ')
			d[a] = [p,s,t]
			l = f.readline()
		states.append(MDP_state(d))
		l = f.readline()
	return MDP(states,initial_state,name)

def MDPFileToMathematica(file_path,output_file):
	m = loadMDP(file_path)
	m.saveToMathematica(output_file)

def prismToMathematica(file_path,output_file):
	m = loadPrismMDP(file_path)
	m.saveToMathematica(output_file)

def loadPrismMDP(file_path):
	f = open(file_path)
	f.readline()
	f.readline()
	l = f.readline()
	l = l.split(' ')

	states = []
	init = int(l[-1][:-2])
	for i in range(int(l[2][4:-1])+1):
		states.append({})

	l = f.readline()
	while l[:-1] != "endmodule":
		act = l[1]
		state = int(l[l.find('=')+1:l.find('-')-1])
		l = (' '+f.readline()).split('+')
		states[state][act] = []
		states[state][act].append([ float(i[1:i.find(':')-1]) for i in l ]) #add proba
		states[state][act].append([ int(i[i.find('=')+1:i.find(')')]) for i in l ]) #add state

		l = f.readline()

	map_s_o = {}
	l = f.readline()

	while l:
		l = l[:-2]
		if not "goal" in l:
			obs = l[l.find('"')+1:l.rfind('"')]
			obs = obs[0].upper() + obs[1:]
			l = l.split('|')
			s = [int(i[i.rfind('=')+1:]) for i in l]
			for ss in s:
				map_s_o[ss] = obs
		l = f.readline()

	for state in range(len(states)):
		for a in states[state]:
			states[state][a].append( [ map_s_o[states[state][a][1][i]] for i in range(len(states[state][a][1])) ] )


	states = [MDP_state(i) for i in states]

	m = MDP(states,init,file_path[:-6])
	#m.pprint()
	#m.save(file_path[:-6]+".txt")
	return m
