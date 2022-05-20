import pyjaja as ja
from random import seed
from datetime import datetime
from os import remove

def test_HMM():
	def modelHMM4():
		h_s0 = ja.HMM_state([[0.4,0.6],['x','y']],[[0.5,0.5],[1,2]],0)
		h_s1 = ja.HMM_state([[0.8,0.2],['a','b']],[[1.0],[3]],1)
		h_s2 = ja.HMM_state([[0.1,0.9],['a','b']],[[1.0],[4]],2)
		h_s3 = ja.HMM_state([[0.5,0.5],['x','y']],[[0.8,0.1,0.1],[0,1,2]],3)
		h_s4 = ja.HMM_state([[1.0],['y']],[[1.0],[3]],4)
		return ja.HMM([h_s0,h_s1,h_s2,h_s3,h_s4],0,"HMM4")
	print("\nHMM")
	model = modelHMM4() 
	print(model)
	model.save("test_save.txt")
	model = ja.loadHMM("test_save.txt")
	s = model.generateSet(100,1/2,"geo",5)
	m3 = ja.BW_HMM().fit(s,nb_states=5)
	print(m3.logLikelihood(s))
	print(model.logLikelihood(s))

def test_MC():
	def modelMC4():
		g_s0 = ja.MC_state([[0.5,0.5],[1,2],['x','y']],0)
		g_s1 = ja.MC_state([[0.4,0.1,0.35,0.15],[3,3,4,4],['a','b','a','b']],1)
		g_s2 = ja.MC_state([[0.3,0.2,0.1,0.4],[1,1,4,4],['b','a','a','b']],2)
		g_s3 = ja.MC_state([[0.5,0.5],[4,5],['c','c']],3)
		g_s4 = ja.MC_state([[1.0],[5],['d']],4)
		g_s5 = ja.MC_state([[1.0],[5],['e']],5)
		return ja.MC([g_s0,g_s1,g_s2,g_s3,g_s4,g_s5],0,"MCGT4")
	print("\nMC")
	model = modelMC4()
	model.save("test_save.txt")
	model = ja.loadMC("test_save.txt")
	print(model)
	s = model.generateSet(100,10)
	m1 = ja.BW_MC().fit(s,nb_states=6)
	print(m1.logLikelihood(s))
	m2 = ja.Alergia().fit(s,0.000005)
	print(m2)
	print(m2.logLikelihood(s))

def test_GOHMM():
	def modelGOHMM2():
		s0 = ja.GOHMM_state([[0.9,0.1],[0,1]],[3.0,5.0],0)
		s1 = ja.GOHMM_state([[0.05,0.9,0.04,0.01],[0,1,2,4]],[0.5,1.5],1)
		s2 = ja.GOHMM_state([[0.05,0.8,0.14,0.01],[1,2,3,4]],[0.2,0.7],2)
		s3 = ja.GOHMM_state([[0.05,0.95],[2,3]],[0.0,0.3],3)
		s4 = ja.GOHMM_state([[0.1,0.9],[1,4]],[2.0,4.0],4)
		return ja.GOHMM([s0,s1,s2,s3,s4],[0.1,0.7,0.0,0.0,0.2],"GOHMM2")
	print("\nGOHMM")
	model = modelGOHMM2() 
	model.save("test_save.txt")
	model = ja.loadGOHMM("test_save.txt")
	print(model)
	s = model.generateSet(100,1/2,"geo",6)
	m3 = ja.BW_GOHMM().fit(s,nb_states=5,random_initial_state=True)
	print(m3)
	print(m3.logLikelihood(s))
	print(model.logLikelihood(s))

def test_MDP():
	def modelMDP_bigstreet(p=0.75):
		m_s_rr = ja.MDP_state({'m': [[p,1-p],[1,2],['L','R']], 's': [[p,1-p],[2,0],['L','R']]},0)
		m_s_ll = ja.MDP_state({'m': [[p,1-p],[0,2],['R','L']], 's': [[p,1-p],[2,1],['R','L']]},1)
		m_s_di = ja.MDP_state({'m': [[1.0],[3],['HIT']],       's': [[1.0],[4],['OK']]},2)
		m_s_de = ja.MDP_state({'m': [[1.0],[3],['HIT']],       's': [[1.0],[3],['HIT']]},3)
		m_s_vi = ja.MDP_state({'m': [[1.0],[4],['OK']],        's': [[1.0],[4],['OK']]},4)
		return ja.MDP([m_s_rr,m_s_ll,m_s_di,m_s_de,m_s_vi],[0.5,0.5,0.0,0.0,0.0],"bigstreet")
	print("\nMDP")
	model = modelMDP_bigstreet()
	model.save("test_save.txt")
	model = ja.loadMDP("test_save.txt")
	print(model)
	scheduler = ja.UniformScheduler(model.actions())
	s = model.generateSet(500,10,scheduler)
	m1 = ja.BW_MDP().fit(s,nb_states=5,random_initial_state=True)
	print(m1)
	print(m1.logLikelihood(s))
	print(model.logLikelihood(s))


test_MDP()
remove("test_save.txt")