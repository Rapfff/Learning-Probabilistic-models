from tools import resolveRandom, correct_proba, find_gcd
from itertools import combinations_with_replacement
from math import pi, sin, cos, log

class MCGT_state:

	def __init__(self,next_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_symbol,transition2_symbol,...]]
		"""
		if round(sum(next_matrix[0]),2) < 1.0 and sum(next_matrix[0]) != 0:
			print("Sum of the probabilies of the next_matrix should be 1 or 0 here it's ",sum(next_matrix[0]))
			#return False
		self.next_matrix = next_matrix

	def next(self):
		c = resolveRandom(self.next_matrix[0])
		return [self.next_matrix[1][c],self.next_matrix[2][c]]

	def g(self,state,obs):
		for i in range(len(self.next_matrix[0])):
			if self.next_matrix[1][i] == state and self.next_matrix[2][i] == obs:
				return self.next_matrix[0][i]
		return 0.0

class MCGT:

	def __init__(self,states,initial_state):
		self.initial_state = initial_state
		self.states = states

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0
			
	def run(self,number_steps):
		output = ""
		current = self.initial_state

		while len(output) < number_steps:
			[next_state, symbol] = self.states[current].next()
			output += symbol
			current = next_state

		return output

	def pprint(self):
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			for j in range(len(self.states[i].next_matrix[0])):
				if self.states[i].next_matrix[0][j] > 0.0:
					print("s",i," - (",self.states[i].next_matrix[2][j],") -> s",self.states[i].next_matrix[1][j]," : ",self.states[i].next_matrix[0][j],sep='')
		print()
	
	def logLikelihood(self,sequences):
		res = 0
		for i in range(len(sequences[0])):
			p = self.probabilityObservations(sequences[0][i])
			if p == 0:
				return -256
			res += log(p) * sequences[1][i]
		return res / sum(sequences[1])

	def allStatesPathIterative(self, start, obs_seq):
		"""return all the states path from start that can generate obs_seq"""
		res = []
		for i in range(len(self.states[start].next_matrix[1])):
			if self.states[start].next_matrix[0][i] > 0 and self.states[start].next_matrix[2][i] == obs_seq[0]:
				if len(obs_seq) == 1:
					res.append([start,self.states[start].next_matrix[1][i]])
				else:
					t = self.allStatesPathIterative(self.states[start].next_matrix[1][i],obs_seq[1:])
					for j in t:
						res.append([start]+j)
		return res
	
	def allStatesPath(self,obs_seq):
		"""return all the states path that can generate obs_seq"""
		res = []
		for j in self.allStatesPathIterative(self.initial_state,obs_seq):
			res.append(j)
		return res

	def allStatesPathLength(self,size):
		"""returns all states path of length <size>+1 """
		combi = combinations_with_replacement([x for x in range(len(self.states))], size)
		res = []
		for i in list(combi)[:-1]:
			res.append([self.initial_state]+list(i))
		return res

	def checkListOrLTL(self,lstprops):
		res = 0.0
		for i in lstprops:
			#res += self.checkLTL(i)
			res += self.probabilityObservations(i)
		return res


	def probabilityStatesObservations(self,states_path, obs_seq):
		"""return the probability to get this states_path generating this observations sequence"""
		if states_path[0] != self.initial_state:
			return 0.0
		else:
			res = 1.0
		for i in range(len(states_path)-1):
			if res == 0.0:
				return 0.0
			res *= self.states[states_path[i]].g(states_path[i+1],obs_seq[i])
		return res
		#but sum_k alpha(seq[:k],states_path[k])*beta(seq[k:],states_path[k]) =? proba(states_path,seq)

	def probabilityObservations(self,obs_seq):
		res = 0
		for p in self.allStatesPath(obs_seq):
			res += self.probabilityStatesObservations(p,obs_seq)
		return res


	def UPPAAL_convert(self,outputfile="mcgt.xml"):
		radius_uppaal_states = 300
		radius_uppaal_branchpoints = 250
		suffix = ".xml"

		if(not outputfile.endswith(suffix)):
			outputfile+=suffix

		#File header
		header  = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
		header += "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>\n"
		header += "<nta>\n"

		#Global declarations
		gl_decl = "\t<declaration>// Place global declarations here.</declaration>\n"

		#Local declarations
		lc_decl = ""

		#Template data, containing locational data of the model, for Uppaal's visual editor
		template1  = "\t<template>\n"
		template1 += "\t\t<name x=\"5\" y=\"5\">Template</name>\n"
		template1 += lc_decl

		#states
		states_positions = [UPPAAL_position(i,len(self.states),radius_uppaal_states) for i in range(len(self.states))]
		for i in range(len(self.states)):
			template1 += UPPAAL_addState(i,states_positions[i])

		template2  = ""

		#transitions
		counter_id_uppaal = len(self.states)
		for source in range(len(self.states)):
			
			if len(self.states[source].next_matrix) != 0:
				next_mat = UPPAAL_probas(self.states[source].next_matrix)
				
				if len(next_mat[0]) == 1: #non-stochastic
					if next_mat[1][0] == source:
						template2 += UPPAAL_addSelfLoop(source,states_positions[source],next_mat[2][0])
					else:
						text_coords = UPPAAl_textPosition(states_positions[source],states_positions[next_mat[1][0]])
						template2 += UPPAAL_addTransition(source,next_mat[1][0],1,next_mat[2][0],text_coords)
				
				else:					  #stochastic => add a branchpoint
					branchpoint_position = UPPAAL_position(source,len(self.states),radius_uppaal_branchpoints)
					template1 += UPPAAL_addBranchpoint(counter_id_uppaal,branchpoint_position)
					template2 += "\t\t<transition>\n\t\t\t<source ref=\"id"+str(source)+"\"/>\n\t\t\t<target ref=\"id"+str(counter_id_uppaal)+"\"/>\n\t\t</transition>\n"

					for transition in range(len(next_mat[0])): #shape if self loop?
						text_coords = UPPAAl_textPosition(branchpoint_position,states_positions[next_mat[1][transition]])
						template2 += UPPAAL_addTransition(counter_id_uppaal,next_mat[1][transition],next_mat[0][transition],next_mat[2][transition],text_coords)

					counter_id_uppaal += 1

		template1 += "\t\t<init ref=\"id"+str(self.initial_state)+"\"/>\n"
		template2 += "\t</template>\n"

		#System declarations
		sys_decl  = "\t<system>// Place template instantiations here.\n"
		sys_decl += "Process = Template();\n"
		sys_decl += "// List one or more processes to be composed into a system.\n"
		sys_decl += "system Process;\n"
		sys_decl += "</system>\n"

		#Queries for verifier
		queries  = "\t<queries>\n"
		queries += "\t\t<query>\n"
		queries += "\t\t\t<formula></formula>\n"
		queries += "\t\t\t<comment></comment>\n"
		queries += "\t\t</query>\n"
		queries += "\t</queries>\n"

		document = open(outputfile, "w")
		document.write(header+gl_decl+template1+template2+sys_decl+queries+"</nta>\n")

		document.close()



def HMMtoMCGT(h):
	states_g = []
	for sh in h.states:
		transitions = [[],[],[]]
		for sy in range(len(sh.output_matrix[0])):
			for ne in range(len(sh.next_matrix[0])):
				transitions[0].append(sh.output_matrix[0][sy]*sh.next_matrix[0][ne])
				transitions[1].append(sh.next_matrix[1][ne])
				transitions[2].append(sh.output_matrix[1][sy])
		states_g.append(MCGT_state(transitions))
	return MCGT(states_g,h.initial_state)

#UPPAAL PACK

def UPPAAL_addState(idd,coords):
	res  = "\t\t<location id=\"id"+str(idd)+"\""
	res += " x=\""+str(coords[0])+"\"" 
	res += " y=\""+str(coords[1])+"\">\n"
	res += "\t\t\t<name x=\""+str(coords[0])+"\" y=\""+str(coords[1])+"\">id"+str(idd)+"</name>\n"
	res += "\t\t</location>\n"
	return res

def UPPAAL_addBranchpoint(idd,coords):
	res  = "\t\t<branchpoint id=\"id"+str(idd)+"\""
	res += " x=\""+str(coords[0])+"\""
	res += " y=\""+str(coords[1])+"\">"
	res += "</branchpoint>\n"
	return res

def UPPAL_addSelfLoop(source,src_coords,symbol):
	radius_diff = 10
	loop_extremum_coords = UPPAAl_textPosition(src_coords,[0,0], 10)
	res  = "\t\t<transition>\n"
	res += "\t\t\t<source ref=\"id"+str(source)+"\"/>\n"
	res += "\t\t\t<target ref=\"id"+str(source)+"\"/>\n"
	res += "\t\t\t<nail x=\""+str(loop_extremum_coords[0])+"\" y=\""+str(loop_extremum_coords[0])+"\">"
	res += "\t\t\t<label kind=\"comments\" x=\""+str(loop_extremum_coords[0])+"\" y=\""+str(loop_extremum_coords[1])+"\">"+symbol+"</label>\n"
	res += "\t\t</transition>\n"
	return res
	
def UPPAAL_addTransition(source,target,prob,symbol,text_coords):
	res  = "\t\t<transition>\n"
	res += "\t\t\t<source ref=\"id"+str(source)+"\"/>\n"
	res += "\t\t\t<target ref=\"id"+str(target)+"\"/>\n"
	res += "\t\t\t<label kind=\"comments\" x=\""+str(text_coords[0])+"\" y=\""+str(text_coords[1]-10)+"\">"+symbol+"</label>\n"
	if prob != 1:
		res += "\t\t\t<label kind=\"probability\" x=\""+str(text_coords[0])+"\" y=\""+str(text_coords[1]+10)+"\">"+str(prob)+"</label>\n"
	res += "\t\t</transition>\n"
	return res

def UPPAAl_textPosition(src_coords, trgt_coords, offset=40):
	if (src_coords[0] - trgt_coords[0]) != 0:
		diff = (src_coords[1] - trgt_coords[1])/(src_coords[0] - trgt_coords[0])
		if src_coords[0] > trgt_coords[0]:
			x = src_coords[0] - offset
			y = int(src_coords[1] - offset*diff)
		else:
			x = src_coords[0] + offset
			y = int(src_coords[1] + offset*diff)
	else:
		x = src_coords[0] + 10
		if src_coords[1] > trgt_coords[1]:
			y = src_coords[1] - offset
		else:
			y = src_coords[1] + offset

	return [x,y]

def UPPAAL_position(idd,number_states,radius=150):
	"""The state are in circle.
	The best way to place the states is to determine a planar layout of a graph with least edges crossing (NP hard)
	Some heuristics exist, but for now ... """
	ox = 0 #arbitrary
	oy = 0 #arbitrary

	return [int(ox + radius*cos(idd*2*pi/number_states)), int(oy + radius*sin(idd*2*pi/number_states))]

def UPPAAL_probas(next_mat,digits=3):
	"""Compute new simpler proba for UPPAAL"""
	"""Remove all the transition with proba == 0 """
	
	#round
	for i in range(len(next_mat[0])):
		next_mat[0][i] = round(next_mat[0][i],digits)*(10**digits)
	#we want sum(probas) = 1
	if sum(next_mat[0]) != (10**digits):
		next_mat[0][0] -= sum(next_mat[0])-1.0

	#simplification
	gcd = find_gcd(next_mat[0])
	for i in range(len(next_mat[0])):
		next_mat[0][i] = int(next_mat[0][i]/gcd)

	#remove proba = 0
	i = 0
	while i < len(next_mat[0]):
		if next_mat[0][i] <= 0.0:
			next_mat[0] = next_mat[0][:i] + next_mat[0][i+1:]
			next_mat[1] = next_mat[1][:i] + next_mat[1][i+1:]
			next_mat[2] = next_mat[2][:i] + next_mat[2][i+1:]
			i -= 1
		i += 1
	return next_mat
