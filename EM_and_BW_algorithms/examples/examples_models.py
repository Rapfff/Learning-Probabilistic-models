import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from HMM import *
from MCGT import *
from MCGS import *
from MDP import *
from tools import randomProbabilities

# ---- HMM ----------------------------

def modelHMM1():
	h_s0 = HMM_state([[1.0],['$']],[[0.3,0.4,0.3],[1,2,3]])
	h_s1 = HMM_state([[1.0],['a']],[[1.0],[4]])
	h_s2 = HMM_state([[1.0],['b']],[[1.0],[4]])
	h_s3 = HMM_state([[1.0],['c']],[[1.0],[4]])
	h_s4 = HMM_state([[1.0],['d']],[[1.0],[4]])
	return HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0,"HMM1")


def modelHMM1_equiprobable():
	h_s0 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s1 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s2 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s3 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s4 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	return HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0,"HMM1_equiprobable")

def modelHMM2():
	h_s0 = HMM_state([[0.5,0.5],['x','y']],[[0.3,0.7,0.0],[1,2,3]])
	h_s1 = HMM_state([[1.0],['a']],[[1.0],[3]])
	h_s2 = HMM_state([[1.0],['b']],[[1.0],[3]])
	h_s4 = HMM_state([[1.0],['d']],[[1.0],[3]])
	return HMM([h_s0,h_s1,h_s2,h_s4],0,"HMM2")

def modelHMM2_random():
	h_s0 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(4),[0,1,2,3]])
	h_s1 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(4),[0,1,2,3]])
	h_s2 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(4),[0,1,2,3]])
	h_s3 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(4),[0,1,2,3]])
	return HMM([h_s0,h_s1,h_s2,h_s3],0,"HMM2_random")

def modelHMM2_equiprobable():
	h_s0 = HMM_state([[0.25,0.25,0.25,0.25,0.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	h_s1 = HMM_state([[0.25,0.25,0.25,0.25,0.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	h_s2 = HMM_state([[0.25,0.25,0.25,0.25,0.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	h_s3 = HMM_state([[0.0,0.0,0.0,0.0,1.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	return HMM([h_s0,h_s1,h_s2,h_s3],0,"HMM2_equiprobable")


def modelHMM3():
	h_s0 = HMM_state([[1.0],['$']],[[0.5,0.5],[1,2]])
	h_s1 = HMM_state([[0.8,0.2],['a','b']],[[1.0],[0]])
	h_s2 = HMM_state([[0.2,0.8],['a','b']],[[1.0],[0]])
	return HMM([h_s0,h_s1,h_s2],0,"HMM3")

def modelHMM4():
	h_s0 = HMM_state([[0.4,0.6],['x','y']],[[0.5,0.5],[1,2]])
	h_s1 = HMM_state([[0.8,0.2],['a','b']],[[1.0],[3]])
	h_s2 = HMM_state([[0.1,0.9],['a','b']],[[1.0],[4]])
	h_s3 = HMM_state([[0.5,0.5],['x','y']],[[0.8,0.1,0.1],[0,1,2]])
	h_s4 = HMM_state([[1.0],['y']],[[1.0],[3]])
	return HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0,"HMM4")

def modelHMM_random(number_states, alphabet):
	states = []
	for s in range(number_states):
		states.append(HMM_state([randomProbabilities(len(alphabet)),alphabet],[randomProbabilities(number_states),list(range(number_states))]))
	return HMM(states,0)

# ---- MCGT ---------------------------

def modelMCGT1():
	g_s0 = MCGT_state([[0.15,0.15,0.2,0.2,0.15,0.15],[1,1,2,2,3,3],['x','y','x','y','x','y']])
	g_s1 = MCGT_state([[1.0],[4],['a']])
	g_s2 = MCGT_state([[1.0],[4],['b']])
	g_s3 = MCGT_state([[1.0],[4],['c']])
	g_s4 = MCGT_state([[0.5,0.5],[4,4],['d','e']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0,"MCGT1")

def modelMCGT2():
	g_s0 = MCGT_state([[0.2,0.2,0.3,0.3],[1,1,2,2],['x','y','x','y']])
	g_s1 = MCGT_state([[1.0],[3],['a']])
	g_s2 = MCGT_state([[1.0],[3],['b']])
	g_s3 = MCGT_state([[1.0],[3],['d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3],0,"MCGT2")

def modelMCGT3():
	g_s0 = MCGT_state([[1/3,1/3,1/3],[1,2,3],['$','$','$']])
	g_s1 = MCGT_state([[1.0],[4],['a']])
	g_s2 = MCGT_state([[1.0],[4],['b']])
	g_s3 = MCGT_state([[1.0],[4],['c']])
	g_s4 = MCGT_state([[1.0],[4],['d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0,"MCGT3")

def modelMCGT4():
	g_s0 = MCGT_state([[0.5,0.5],[1,2],['x','y']])
	g_s1 = MCGT_state([[0.4,0.1,0.35,0.15],[3,3,4,4],['a','b','a','b']])
	g_s2 = MCGT_state([[0.3,0.2,0.1,0.4],[1,1,4,4],['b','a','a','b']])
	g_s3 = MCGT_state([[0.5,0.5],[4,5],['c','c']])
	g_s4 = MCGT_state([[1.0],[5],['d']])
	g_s5 = MCGT_state([[1.0],[5],['e']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5],0,"MCGT4")

def modelMCGT_game():
	s_dice = MCGT_state([[1/3,1/3,1/3],[0,0,1],['1-2','3-4','5-6']])
	s_cards= MCGT_state([[10/13,3/13],[1,2],["Number","Face"]])
	s_win  = MCGT_state([[1.0],[2],["Win"]])
	return MCGT([s_dice,s_cards,s_win],0,"MCGT_games")

def modelMCGT_REBER():
	g_s0 = MCGT_state([[1.0],[1],['B']])
	g_s1 = MCGT_state([[0.5,0.5],[2,3],['T','P']])
	g_s2 = MCGT_state([[0.6,0.4],[2,4],['S','X']])
	g_s3 = MCGT_state([[0.7,0.3],[3,5],['T','V']])
	g_s4 = MCGT_state([[0.5,0.5],[3,6],['X','S']])
	g_s5 = MCGT_state([[0.5,0.5],[4,6],['P','V']])
	g_s6 = MCGT_state([[1.0],[6],['E']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5,g_s6],0,"MCGT_REBER")

def modelMCGT_REBERlight():
	g_s0 = MCGT_state([[0.5,0.5],[1,2],['T','P']])
	g_s1 = MCGT_state([[1.0],[3],['X']])
	g_s2 = MCGT_state([[1.0],[4],['V']])
	g_s3 = MCGT_state([[0.5,0.5],[2,5],['X','S']])
	g_s4 = MCGT_state([[0.5,0.5],[3,5],['P','V']])
	g_s5 = MCGT_state([[1.0],[5],['E']])
	g_s6 = MCGT_state([[1.0],[0],['B']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5,g_s6],6,"MCGT_REBER")

def modelMCGT1_assist():
	g_s0 = MCGT_state([randomProbabilities(6),[1,1,2,2,3,3],['x','y','x','y','x','y']])
	g_s1 = MCGT_state([randomProbabilities(4),[1,2,3,4],['a','a','a','a']])
	g_s2 = MCGT_state([randomProbabilities(4),[1,2,3,4],['b','b','b','b']])
	g_s3 = MCGT_state([randomProbabilities(4),[1,2,3,4],['c','c','c','c']])
	g_s4 = MCGT_state([randomProbabilities(4),[1,2,3,4],['d','d','d','d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0,"MCGT1_assist")

def modelMCGT_equiprobable(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MCGT_state([[1/len(obs)]*len(obs),s,obs]))
	return MCGT(states,0,"MCGT_equiprobable_"+str(nb_states)+"states")

def modelMCGT_random(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MCGT_state([randomProbabilities(len(obs)),s,obs]))
	return MCGT(states,0,"MCGT_random_"+str(nb_states)+"states")

def modelMCGT1_equiprobable():
	return modelMCGT_equiprobable(5,['x','y','a','b','c','d'])


# ---- MCGS ---------------------------

def modelMCGS1():
	m_s0  = MCGS_state('',[[0.5,0.5],[5,6]])
	m_s0x = MCGS_state('x',[[0.3,0.4,0.3],[1,2,3]])
	m_s0y = MCGS_state('y',[[0.3,0.4,0.3],[1,2,3]])
	m_s1  = MCGS_state('',[[1.0],[7]])
	m_s1a = MCGS_state('a',[[1.0],[4]])
	m_s2  = MCGS_state('',[[1.0],[8]])
	m_s2b = MCGS_state('b',[[1.0],[4]])
	m_s3  = MCGS_state('',[[1.0],[9]])
	m_s3c = MCGS_state('c',[[1.0],[4]])
	m_s4  = MCGS_state('',[[1.0],[10]])
	m_s4d = MCGS_state('d',[[1.0],[4]])
	return MCGS([m_s0,m_s1,m_s2,m_s3,m_s4,m_s0x,m_s0y,m_s1a,m_s2b,m_s3c,m_s4d],[[1.0],[0]])


# ---- MDP ----------------------------

def modelMDP_random(nb_states,alphabet,actions):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		dic = {}
		for act in actions:
			dic[act] = [randomProbabilities(len(obs)),s,obs]
		states.append(MDP_state(dic))
	return MDP(states,0,"MDP_random_"+str(nb_states)+"states")

def modelMDP1_fullyobservable():
	m_s0 = MDP_state({'a': [[1.0],[1],['1']], 'b': [[1.0],[2],['2']]})
	m_s1 = MDP_state({'a': [[0.8,0.2],[2,3],['2','3']], 'b': [[0.9,0.1],[3,4],['3','4']]})
	m_s2 = MDP_state({'a': [[0.9,0.1],[6,4],['6','4']], 'b': [[1.0],[2],['2']]})
	m_s3 = MDP_state({'a': [[0.9,0.1],[1,2],['1','2']], 'b': [[1.0],[1],['1']]})
	m_s4 = MDP_state({'a': [[1.0],[5],['5']], 'b': [[1.0],[2],['4']]})
	m_s5 = MDP_state({'a': [[1.0],[0],['0']], 'b': [[1.0],[0],['0']]})
	m_s6 = MDP_state({'a': [[1.0],[4],['4']], 'b': [[1.0],[4],['4']]})
	return MDP([m_s0,m_s1,m_s2,m_s3,m_s4,m_s5,m_s6],0,"MDP1_fullyobservable")

def modelMDP2():
	# x*y grid
	# starting point : (x_init,y_init)
	# p : prob of deplacement failure
	x,y = 4,3
	x_init,y_init = 0,0
	p = 0.2

	target_obs = []
	for i in range(y):
		target_obs.append([])	
		if i == 0:
			preobs = 't'
		elif i == y-1:
			preobs = 'b'
		else:
			preobs = ''
		
		for j in range(x):
			if j == 0:
				target_obs[-1].append(preobs+'l')
			elif j == x-1:
				target_obs[-1].append(preobs+'r')
			elif preobs == '':
				target_obs[-1].append('n')
			else:
				target_obs[-1].append(preobs)


	states = []
	pos = 0
	for i in range(y):
		for j in range(x):
			dic = {}
			if i == 0:
				dic['t'] = [ [1.0], [pos], [target_obs[i][j]] ]
			else:
				dic['t'] = [ [1.0-p,p], [pos - x, pos], [target_obs[i-1][j],target_obs[i][j]] ]

			if i == y-1:
				dic['b'] = [ [1.0],[pos], [target_obs[i][j]] ]
			else:
				dic['b'] = [ [1.0-p,p],[pos + x, pos], [target_obs[i+1][j],target_obs[i][j]] ]

			if j == 0:
				dic['l'] = [ [1.0],[pos], [target_obs[i][j]] ]
			else:
				dic['l'] = [ [1.0-p,p],[pos - 1, pos], [target_obs[i][j-1],target_obs[i][j]] ]

			if j == x-1:
				dic['r'] = [ [1.0],[pos], [target_obs[i][j]] ]
			else:
				dic['r'] = [ [1.0-p,p],[pos + 1, pos], [target_obs[i][j+1],target_obs[i][j]] ]

			states.append(MDP_state(dic))
			pos += 1

	return MDP(states, y_init*x+x_init,"MDP2")

def modelMDP3():
	m_s0 = MDP_state({'a': [[1.0],[0],['0']], 'b': [[1.0],[1],['1']]})
	m_s1 = MDP_state({'a': [[0.8,0.2],[0,1],['0','1']], 'b': [[0.9,0.1],[1,0],['1','0']]})
	return MDP([m_s0,m_s1],0,"MDP3")

def modelMDP4():
	m_s0 = MDP_state({'a': [[0.25,0.75],[2,1],['C','B']], 'b': [[1.0],[0],['A']]})
	m_s1 = MDP_state({'b': [[1.0],[0],['B']]})
	m_s2 = MDP_state({'a': [[1.0],[2],['C']]})
	return MDP([m_s0,m_s1,m_s2],0,"MDP4")

def modelMDP5():
	m_s0 = MDP_state({'a': [[1.0],[1],['A']], 'b': [[1.0],[2],['A']]})
	m_s1 = MDP_state({'a': [[0.8,0.2],[0,3],['A','C']], 'b': [[1.0],[2],['A']]})
	m_s2 = MDP_state({'a': [[0.5,0.5],[0,1],['A','B']], 'b': [[1.0],[3],['C']]})
	m_s3 = MDP_state({'a': [[1.0],[3],['A']], 'b': [[1.0],[0],['A']]})
	return MDP([m_s0,m_s1,m_s2,m_s3],0,"MDP5")


def scheduler_random(actions):
	return FiniteMemoryScheduler({0:[[1/len(actions)]*len(actions),actions]},{})

def scheduler_always_same(action):
	return FiniteMemoryScheduler({0:[[1.0],[action]]},{})

# -------------------------------------
