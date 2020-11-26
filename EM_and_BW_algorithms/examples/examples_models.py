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
	return HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0)


def modelHMM1_equiprobable():
	h_s0 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s1 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s2 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s3 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	h_s4 = HMM_state([[0.2,0.2,0.2,0.2,0.2],['$','a','b','c','d']],[[0.2,0.2,0.2,0.2,0.2],[0,1,2,3,4]])
	return HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0)

def modelHMM2():
	h_s0 = HMM_state([[0.5,0.5],['x','y']],[[0.3,0.7,0.0],[1,2,3]])
	h_s1 = HMM_state([[1.0],['a']],[[1.0],[3]])
	h_s2 = HMM_state([[1.0],['b']],[[1.0],[3]])
	h_s4 = HMM_state([[1.0],['d']],[[1.0],[3]])
	return HMM([h_s0,h_s1,h_s2,h_s4],0)

def modelHMM2_random():
	h_s0 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(5),[0,1,2,3]])
	h_s1 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(5),[0,1,2,3]])
	h_s2 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(5),[0,1,2,3]])
	h_s3 = HMM_state([randomProbabilities(5),['x','y','a','b','d']],[randomProbabilities(5),[0,1,2,3]])
	return HMM([h_s0,h_s1,h_s2,h_s3],0)

def modelHMM2_equiprobable():
	h_s0 = HMM_state([[0.25,0.25,0.25,0.25,0.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	h_s1 = HMM_state([[0.25,0.25,0.25,0.25,0.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	h_s2 = HMM_state([[0.25,0.25,0.25,0.25,0.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	h_s3 = HMM_state([[0.0,0.0,0.0,0.0,1.0],['x','y','a','b','d']],[[0.25,0.25,0.25,0.25],[0,1,2,3]])
	return HMM([h_s0,h_s1,h_s2,h_s3],0)


def modelHMM3():
	h_s0 = HMM_state([[1.0],['$']],[[0.5,0.5],[1,2]])
	h_s1 = HMM_state([[0.8,0.2],['a','b']],[[1.0],[0]])
	h_s2 = HMM_state([[0.2,0.8],['a','b']],[[1.0],[0]])
	return HMM([h_s0,h_s1,h_s2],0)

def modelHMM4():
	h_s0 = HMM_state([[0.4,0.6],['x','y']],[[0.5,0.5],[1,2]])
	h_s1 = HMM_state([[0.8,0.2],['a','b']],[[1.0],[3]])
	h_s2 = HMM_state([[0.1,0.9],['a','b']],[[1.0],[4]])
	h_s3 = HMM_state([[0.5,0.5],['x','y']],[[0.8,0.1,0.1],[0,1,2]])
	h_s4 = HMM_state([[1.0],['y']],[[1.0],[3]])
	return HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0)

# ---- MCGT ---------------------------

def modelMCGT1():
	g_s0 = MCGT_state([[0.15,0.15,0.2,0.2,0.15,0.15],[1,1,2,2,3,3],['x','y','x','y','x','y']])
	g_s1 = MCGT_state([[1.0],[4],['a']])
	g_s2 = MCGT_state([[1.0],[4],['b']])
	g_s3 = MCGT_state([[1.0],[4],['c']])
	g_s4 = MCGT_state([[1.0],[4],['d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0)

def modelMCGT2():
	g_s0 = MCGT_state([[0.2,0.2,0.3,0.3],[1,1,2,2],['x','y','x','y']])
	g_s1 = MCGT_state([[1.0],[3],['a']])
	g_s2 = MCGT_state([[1.0],[3],['b']])
	g_s3 = MCGT_state([[1.0],[3],['d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3],0)

def modelMCGT3():
	g_s0 = MCGT_state([[1/3,1/3,1/3],[1,2,3],['$','$','$']])
	g_s1 = MCGT_state([[1.0],[4],['a']])
	g_s2 = MCGT_state([[1.0],[4],['b']])
	g_s3 = MCGT_state([[1.0],[4],['c']])
	g_s4 = MCGT_state([[1.0],[4],['d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0)

def modelMCGT4():
	g_s0 = MCGT_state([[0.5,0.5],[1,2],['x','y']])
	g_s1 = MCGT_state([[0.4,0.1,0.35,0.15],[3,3,4,4],['a','b','a','b']])
	g_s2 = MCGT_state([[0.3,0.2,0.1,0.4],[1,1,4,4],['b','a','a','b']])
	g_s3 = MCGT_state([[0.5,0.5],[4,5],['c','c']])
	g_s4 = MCGT_state([[1.0],[5],['d']])
	g_s5 = MCGT_state([[1.0],[5],['e']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5],0)

def modelMCGT1_assist():
	g_s0 = MCGT_state([randomProbabilities(6),[1,1,2,2,3,3],['x','y','x','y','x','y']])
	g_s1 = MCGT_state([randomProbabilities(4),[1,2,3,4],['a','a','a','a']])
	g_s2 = MCGT_state([randomProbabilities(4),[1,2,3,4],['b','b','b','b']])
	g_s3 = MCGT_state([randomProbabilities(4),[1,2,3,4],['c','c','c','c']])
	g_s4 = MCGT_state([randomProbabilities(4),[1,2,3,4],['d','d','d','d']])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0)

def modelMCGT_equiprobable(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MCGT_state([[1/len(obs)]*len(obs),s,obs]))
	return MCGT(states,0)

def modelMCGT_random(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MCGT_state([randomProbabilities(len(obs)),s,obs]))
	return MCGT(states,0)

def modelMCGT1_equiprobable():
	return modelMCGT_equiprobable(5,['x','y','a','b','c','d'])

def modelMCGT1_random():
	s = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4]
	obs = ['x','y','a','b','c','d']*5
	g_s0 = MCGT_state([randomProbabilities(len(obs)),s,obs])
	g_s1 = MCGT_state([randomProbabilities(len(obs)),s,obs])
	g_s2 = MCGT_state([randomProbabilities(len(obs)),s,obs])
	g_s3 = MCGT_state([randomProbabilities(len(obs)),s,obs])
	g_s4 = MCGT_state([randomProbabilities(len(obs)),s,obs])
	return MCGT([g_s0,g_s1,g_s2,g_s3,g_s4],0)


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

def modelMDP1():
	m_s0 = MDP_state({'a': [[1.0],[1]], 'b': [[1.0],[2]]},"s0")
	m_s1 = MDP_state({'a': [[0.8,0.2],[2,3]], 'b': [[0.9,0.1],[3,4]]},"s1")
	m_s2 = MDP_state({'a': [[0.9,0.1],[6,4]]},"s2")
	m_s3 = MDP_state({'a': [[0.9,0.1],[1,2]], 'b': [[1.0],[1]]},"s3")
	m_s4 = MDP_state({'a': [[1.0],[5]]},"s4")
	m_s5 = MDP_state({'a': [[1.0],[0]], 'b': [[1.0],[0]]},"s5")
	m_s6 = MDP_state({'a': [[1.0],[4]], 'b': [[1.0],[4]]},"s6")
	return MDP([m_s0,m_s1,m_s2,m_s3,m_s4,m_s5,m_s6],0)

# -------------------------------------
