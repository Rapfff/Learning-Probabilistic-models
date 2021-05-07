from MDP import *
from tools import randomProbabilities

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
