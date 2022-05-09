import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.models.HMM import *
from src.models.MC import *
from src.models.CTMC import *
from src.models.MDP import *
from src.models.coMC import *
from src.models.coHMM import *
from src.tools import randomProbabilities
from random import randint, uniform
from math import sqrt

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

def modelHMM_random(number_states, alphabet, random_initial_state=False):
	states = []
	for s in range(number_states):
		states.append(HMM_state([randomProbabilities(len(alphabet)),alphabet],[randomProbabilities(number_states),list(range(number_states))]))

	if random_initial_state:
		init = randomProbabilities(number_states)
	else:
		init = 0
	return HMM(states,init)

# ---- coHMM ---------------------------
def modelCOHMM_random(nb_states,random_initial_state=False,min_mu=0.0,max_mu=2.0,min_std=0.5,max_std=2.0):
	#mu between -2 and 2
	#sd between 0 and 2
	s = [i for i in range(nb_states)]
	states = []
	for i in range(nb_states):
		d = [round(uniform(min_mu,max_mu),3),round(uniform(min_std,max_std),3)]
		states.append(coHMM_state([randomProbabilities(nb_states),s],d))
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return coHMM(states,init,"coHMM_random_"+str(nb_states)+"_states")

def modelCOHMM_nox(nb_states=5,random_initial_state=True,min_mu=-0.2,max_mu=0.5,min_std=0.05,max_std=4.5,self_loop_prob=0.8):
	m = modelCOHMM_random(nb_states,random_initial_state,min_mu,max_mu,min_std,max_std)
	for s in range(nb_states):
		r = 1-m.states[s].next_matrix[0][s]
		m.states[s].next_matrix[0] = [(m.states[s].next_matrix[0][j]/r)*(1-self_loop_prob) if j != s else self_loop_prob for j in range(nb_states)]
	m.name = "coHMM_random_nox"
	return m

def modelCOHMM1():
	s0 = coHMM_state([[0.2,0.8],[0,1]],[0.0,1.0])
	s1 = coHMM_state([[1.0],[0]],[0.5,1.5])
	return coHMM([s0,s1],0,"coHMM1")

# ---- MC -----------------------------

def modelMC1():
	g_s0 = MC_state([[0.15,0.15,0.2,0.2,0.15,0.15],[1,1,2,2,3,3],['x','y','x','y','x','y']])
	g_s1 = MC_state([[1.0],[4],['a']])
	g_s2 = MC_state([[1.0],[4],['b']])
	g_s3 = MC_state([[1.0],[4],['c']])
	g_s4 = MC_state([[0.5,0.5],[4,4],['d','e']])
	return MC([g_s0,g_s1,g_s2,g_s3,g_s4],0,"MCGT1")

def modelMC2():
	g_s0 = MC_state([[0.2,0.2,0.3,0.3],[1,1,2,2],['x','y','x','y']])
	g_s1 = MC_state([[1.0],[3],['a']])
	g_s2 = MC_state([[1.0],[3],['b']])
	g_s3 = MC_state([[1.0],[3],['d']])
	return MC([g_s0,g_s1,g_s2,g_s3],0,"MCGT2")

def modelMC3():
	g_s0 = MC_state([[1/3,1/3,1/3],[1,2,3],['$','$','$']])
	g_s1 = MC_state([[1.0],[4],['a']])
	g_s2 = MC_state([[1.0],[4],['b']])
	g_s3 = MC_state([[1.0],[4],['c']])
	g_s4 = MC_state([[1.0],[4],['d']])
	return MC([g_s0,g_s1,g_s2,g_s3,g_s4],0,"MCGT3")

def modelMC4():
	g_s0 = MC_state([[0.5,0.5],[1,2],['x','y']])
	g_s1 = MC_state([[0.4,0.1,0.35,0.15],[3,3,4,4],['a','b','a','b']])
	g_s2 = MC_state([[0.3,0.2,0.1,0.4],[1,1,4,4],['b','a','a','b']])
	g_s3 = MC_state([[0.5,0.5],[4,5],['c','c']])
	g_s4 = MC_state([[1.0],[5],['d']])
	g_s5 = MC_state([[1.0],[5],['e']])
	return MC([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5],0,"MCGT4")

def modelMC5(p1=0.4,p2=0.3,p3=0.2,p4=0.05):
	g_s0 = MC_state([[p1,p2,p3,1-p1-p2-p3],[0,0,1,1],['a','b','a','b']])
	g_s1 = MC_state([[p4,1-p4],[0,0],['a','b']])
	return MC([g_s0,g_s1],0,"MCGT5")

def modelMC_game():
	s_dice = MC_state([[1/3,1/3,1/3],[0,0,1],['1-2','3-4','5-6']])
	s_cards= MC_state([[10/13,3/13],[1,2],["Number","Face"]])
	s_win  = MC_state([[1.0],[2],["Win"]])
	return MC([s_dice,s_cards,s_win],0,"MCGT_games")

def modelMC_map():
	size = 9
	initial = 4
	actions= {'n':-3,'s':3,'e':1,'w':-1}
	materials = ['S','M','G',
				 'M','G','C',
				 'G','S','M']
	error_probs = {'C':0.0, 'G': 0.2, 'M':0.4, 'S':0.25}
	states = []
	for s in range(size):
		p = []
		ss = []
		o = []
		for l in actions.keys():
			dest = s+actions[l]
			if (dest<0 or
			    dest>=size or
				(l == 'e' and dest//sqrt(size) != s//sqrt(size)) or
				(l == 'e' and dest//sqrt(size) != s//sqrt(size))):
				p.append(1.0/4)
				ss.append(s)
				o.append(l)
			else:
				errors_cells = []	
				if (l == 'e' or l == 'w'):
					if dest-int(sqrt(size)) >= 0:
						errors_cells.append(dest-int(sqrt(size)))
					if dest+int(sqrt(size)) < size:
						errors_cells.append(dest+int(sqrt(size)))
				if (l == 'n' or l == 's'):
					if dest%int(sqrt(size)) > 0:
						errors_cells.append(dest-1)
					if dest%int(sqrt(size)) < 2:
						errors_cells.append(dest+1)
				p.append((1-error_probs[materials[dest]])/4)
				ss.append(dest)
				o.append(l)
				p += [(error_probs[materials[dest]]/len(errors_cells))/4 for i in errors_cells]
				ss+= errors_cells
				o += [l]*len(errors_cells)
		states.append(MC_state([p,ss,o]))
	
	return MC(states,initial,name="map")
	
def modelMC_REBER():
	g_s0 = MC_state([[1.0],[1],['B']])
	g_s1 = MC_state([[0.5,0.5],[2,3],['T','P']])
	g_s2 = MC_state([[0.6,0.4],[2,4],['S','X']])
	g_s3 = MC_state([[0.7,0.3],[3,5],['T','V']])
	g_s4 = MC_state([[0.5,0.5],[3,6],['X','S']])
	g_s5 = MC_state([[0.5,0.5],[4,6],['P','V']])
	g_s6 = MC_state([[1.0],[6],['E']])
	return MC([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5,g_s6],0,"MCGT_REBER")


def modelMC1_assist():
	g_s0 = MC_state([randomProbabilities(6),[1,1,2,2,3,3],['x','y','x','y','x','y']])
	g_s1 = MC_state([randomProbabilities(4),[1,2,3,4],['a','a','a','a']])
	g_s2 = MC_state([randomProbabilities(4),[1,2,3,4],['b','b','b','b']])
	g_s3 = MC_state([randomProbabilities(4),[1,2,3,4],['c','c','c','c']])
	g_s4 = MC_state([randomProbabilities(4),[1,2,3,4],['d','d','d','d']])
	return MC([g_s0,g_s1,g_s2,g_s3,g_s4],0,"MCGT1_assist")

def modelMC_equiprobable(nb_states,alphabet):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MC_state([[1/len(obs)]*len(obs),s,obs]))
	return MC(states,0,"MCGT_equiprobable_"+str(nb_states)+"_states")

def modelMC_random(nb_states,alphabet,random_initial_state=False):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MC_state([randomProbabilities(len(obs)),s,obs]))
	
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MC(states,init,"MCGT_random_"+str(nb_states)+"_states")

# ---- CTMC----------------------------

def modelCTMC_random(nb_states: int, alphabet: list, min_waiting_time : int, max_waiting_time: int, self_loop: bool = True) -> CTMC:
	#lambda between 0 and 1
	s = []
	for j in range(nb_states):
		s.append([])
		for i in range(nb_states):
			if self_loop or i != j:
				s[j] += [i] * len(alphabet)
	if self_loop:
		obs = alphabet*nb_states
	else:
		obs = alphabet*(nb_states-1)

	states = []
	for i in range(nb_states):
		av_waiting_time = randint(min_waiting_time,max_waiting_time)
		states.append(CTMC_state([[p/av_waiting_time for p in randomProbabilities(len(obs))],s[i],obs]))

	return CTMC(states,0,"CTMC_random_"+str(nb_states)+"_states")

def modelCTMC_REBER():
	g_s0 = CTMC_state([[1.0],[1],['B']])
	g_s1 = CTMC_state([[0.5,0.5],[2,3],['T','P']])
	g_s2 = CTMC_state([[0.3,0.2],[2,4],['S','X']])
	g_s3 = CTMC_state([[0.175,0.075],[3,5],['T','V']])
	g_s4 = CTMC_state([[0.05,0.05],[3,6],['X','S']])
	g_s5 = CTMC_state([[0.0625,0.0625],[4,6],['P','V']])
	g_s6 = CTMC_state([[0.02],[6],['E']])
	return CTMC([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5,g_s6],0,"CTMC_REBER")


def modelCTMC1():
	s0 = CTMC_state([[0.05,0.45,0.5],[0,1,1],['a','a','b']])
	s1 = CTMC_state([[0.005,0.005],[0,1],['a','b']])
	return CTMC([s0,s1],0,"CTMC1")

def modelCTMC2(suffix=''):
	s0 = CTMC_state([[0.3/5,0.5/5,0.2/5],[1,2,3], ['r'+suffix,'g'+suffix,'r'+suffix]])
	s1 = CTMC_state([[0.08,0.25,0.6,0.07],[0,2,2,3], ['r'+suffix,'r'+suffix,'g'+suffix,'b'+suffix]])
	s2 = CTMC_state([[0.5/4,0.2/4,0.3/4],[1,3,3], ['b'+suffix,'g'+suffix,'r'+suffix]])
	s3 = CTMC_state([[0.95/2,0.04/2,0.01/2],[0,0,2], ['r'+suffix,'g'+suffix,'r'+suffix]])
	return CTMC([s0,s1,s2,s3],0,"CTMC2")

def modelCTMC3(suffix=''):
	s0 = CTMC_state([[0.65/4,0.35/4],[1,3],['g'+suffix,'b'+suffix]])
	s1 = CTMC_state([[0.6/3,0.1/3,0.3/3],[0,3,3],['g'+suffix,'g'+suffix,'b'+suffix]])
	s2 = CTMC_state([[0.25/5,0.6/5,0.15/5],[0,0,1],['r'+suffix,'g'+suffix,'b'+suffix]])
	s3 = CTMC_state([[1.0/10],[2],['g'+suffix]])
	return CTMC([s0,s1,s2,s3],0,"CTMC3")

# ---- coMC ---------------------------
def modelCOMC_random(nb_states,random_initial_state=False,min_mu=0.0,max_mu=2.0,min_std=0.5,max_std=2.0):
	#mu between -2 and 2
	#sd between 0 and 2
	s = [i for i in range(nb_states)]
	states = []
	for i in range(nb_states):
		d = {}
		for j in range(nb_states):
			d[j] = [round(uniform(min_mu,max_mu),3),round(uniform(min_std,max_std),3)]
		states.append(coMC_state([randomProbabilities(nb_states),s],d))
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return coMC(states,init,"coMC_random_"+str(nb_states)+"_states")

def modelCOMC1():
	s0 = coMC_state([[0.2,0.8],[0,1]],{0:[0.0,1.0],1:[1.0,1.0]})
	s1 = coMC_state([[1.0],[0]],{0:[0.5,1.5]})
	return coMC([s0,s1],0,"coMC1")


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
	return MDP(states,0,"MDP_random_"+str(nb_states)+"_states")

def modelMDP1_fullyobservable():
	m_s0 = MDP_state({'a': [[1.0],[1],['1']], 'b': [[1.0],[2],['2']]})
	m_s1 = MDP_state({'a': [[0.8,0.2],[2,3],['2','3']], 'b': [[0.9,0.1],[3,4],['3','4']]})
	m_s2 = MDP_state({'a': [[0.9,0.1],[6,4],['6','4']], 'b': [[1.0],[2],['2']]})
	m_s3 = MDP_state({'a': [[0.9,0.1],[1,2],['1','2']], 'b': [[1.0],[1],['1']]})
	m_s4 = MDP_state({'a': [[1.0],[5],['5']], 'b': [[1.0],[2],['2']]})
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

def modelMDP6():
	m_s0 = MDP_state({'a': [[1.0],[1],['A']], 'b': [[1/3,2/3],[2,4],['A','A']]})
	m_s1 = MDP_state({'a': [[0.5,1/9,7/18],[1,3,2],['A','A','A']]})
	m_s2 = MDP_state({'a': [[1.0],[2],['A']]})
	m_s3 = MDP_state({'a': [[1.0],[3],['A']]})
	m_s4 = MDP_state({'a': [[0.25,0.75],[5,6],['A','A']]})
	m_s5 = MDP_state({'a': [[1.0],[6],['A']], 'b': [[1/3,2/3],[2,7],['A','A']]})
	m_s6 = MDP_state({'a': [[3/5,2/5],[6,5],['A','A']]})
	m_s7 = MDP_state({'a': [[0.5,0.5],[2,3],['A','A']]})
	return MDP([m_s0,m_s1,m_s2,m_s3,m_s4,m_s5,m_s6,m_s7],0,"MDP6")

def modelMDP_smallstreet(p=0.75):
	m_s_same = MDP_state({'m': [[p,1-p],[0,1],['M','S']], 's': [[p,1-p],[1,0],['M','S']]})
	m_s_diff = MDP_state({'m': [[1.0],[1],['OK']],        's': [[1.0],[1],['OK']]})
	return MDP([m_s_same,m_s_diff],0,"smallstreet")


def modelMDP_midstreet(p=0.75):
	m_s_rr = MDP_state({'m': [[p,1-p],[1,3],['L','R']], 's': [[p,1-p],[2,0],['L','R']]})
	m_s_ll = MDP_state({'m': [[p,1-p],[0,2],['R','L']], 's': [[p,1-p],[3,1],['R','L']]})
	m_s_rl = MDP_state({'m': [[1.0],[2],['OK']],        's': [[1.0],[2],['OK']]})
	m_s_lr = MDP_state({'m': [[1.0],[3],['OK']],        's': [[1.0],[3],['OK']]})
	return MDP([m_s_rr,m_s_ll,m_s_rl,m_s_lr],0,"midstreet")

def modelMDP_bigstreet(p=0.75):
	m_s_rr = MDP_state({'m': [[p,1-p],[1,2],['L','R']], 's': [[p,1-p],[2,0],['L','R']]})
	m_s_ll = MDP_state({'m': [[p,1-p],[0,2],['R','L']], 's': [[p,1-p],[2,1],['R','L']]})
	m_s_di = MDP_state({'m': [[1.0],[3],['HIT']],       's': [[1.0],[4],['OK']]})
	m_s_de = MDP_state({'m': [[1.0],[3],['HIT']],       's': [[1.0],[3],['HIT']]})
	m_s_vi = MDP_state({'m': [[1.0],[4],['OK']],        's': [[1.0],[4],['OK']]})
	return MDP([m_s_rr,m_s_ll,m_s_di,m_s_de,m_s_vi],0,"bigstreet")

def scheduler_uniform(actions):
	return FiniteMemoryScheduler({0:[[1/len(actions)]*len(actions),actions]},{})

def scheduler_always_same(action):
	return FiniteMemoryScheduler({0:[[1.0],[action]]},{})

# -------------------------------------
