from tools import resolveRandom

class MDP_state:

	def __init__(self,next_matrix, obs):
		"""
		next_matrix = {action1 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...]],
					   action2 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...]],
					   ...}
		"""
		for action in next_matrix:
			if round(sum(next_matrix[action][0]),2) < 1.0:
				print("Sum of the probabilies of the next_matrix should be 1.0 here it's ",sum(next_matrix[0]))
				return False
		self.next_matrix = next_matrix
		self.observation = obs

	def next(self,action):
		if not action in self.next_matrix:
			print("ACTION",action,"is not available in state",self.observation)
		c = resolveRandom(self.next_matrix[action][0])
		return self.next_matrix[action][1][c]

	def actions(self):
		return [i for i in self.next_matrix]


class MDP:

	def __init__(self,states,initial_state):
		self.initial_state = initial_state
		self.states = states

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0
			
	def run(self,number_steps,scheduler):
		"""
		scheduler(path, current) where path <string> is the path from now, current <MDP_state> is the current state
		"""
		output = [self.states[self.initial_state].observation]
		actions = []
		current = self.initial_state

		while len(output) < number_steps+1:
			action = scheduler(output, self.states[current])
			next_state = self.states[current].next(action)
			output.append(self.states[next_state].observation)
			actions.append(action)
			current = next_state

		return [output,actions]

	def pprint(self):
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			for action in self.states[i].next_matrix:
				for j in range(len(self.states[i].next_matrix[action][0])):
					if self.states[i].next_matrix[action][0][j] > 0.0:
						print("s",i," - (",action,") -> s",self.states[i].next_matrix[action][1][j]," : ",self.states[i].next_matrix[action][0][j],sep='')
			print("observation  --",self.states[i].observation)
		print()
	
	def probabilityStatesObservations(self,states_path, obs_seq):
		"""return the probability to get this states_path generating this observations sequence"""
		return None

	def probabilityObservations(self,obs_seq):
		"""return the probability to get this observations sequence"""
		return None


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
