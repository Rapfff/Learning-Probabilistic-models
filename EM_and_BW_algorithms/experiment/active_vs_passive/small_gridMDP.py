import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from random import randint
from tools import resolveRandom

class SmallGrid:

	def __init__(self):
		#set of labels = ['Concrete','Grass','Wall','Mud','Sand']
		#set of actions= ['N','S','W','E']
		self.reset()
		self.map =	[  [ 'S','M','G' ],
					   [ 'M','G','C' ],
					   [ 'G','S','M' ]
					]
		self.error_probs = {'C':0.0, 'G': 0.2, 'M':0.4, 'S':0.25}
		self.dimX = len(self.map[0])
		self.dimY = len(self.map)

	def get_state_obs(self):
		if   self.map[self.y][self.x] == 'C':
			return "Concrete"
		elif self.map[self.y][self.x] == 'G':
			return "Grass"
		elif self.map[self.y][self.x] == 'W':
			return "Wall"
		elif self.map[self.y][self.x] == 'M':
			return "Mud"
		elif self.map[self.y][self.x] == 'S':
			return "Sand"

	def reset(self):
		self.x = 1
		self.y = 1

	def observations(self):
		return ['Concrete','Grass','Wall','Mud','Sand']

	def actions(self):
		return ['N','S','W','E']

	def availableCells(self,nx,ny):
		cells = []
		if nx == self.x:
			for dx in [-1,+1]:
				if nx+dx < self.dimX and nx+dx >= 0:
					if self.map[ny][nx+dx] != 'W':
						cells.append((nx+dx,ny))
		else:
			for dy in [-1,+1]:
				if ny+dy < self.dimY and ny+dy >= 0:
					if self.map[ny+dy][nx] != 'W':
						cells.append((nx,ny+dy))
		return cells

	def probabilisticMove(self,nx,ny):
		#move the player regarding to the probabilities
		if self.map[ny][nx] == 'C':
			self.x, self.y = nx, ny
			return
		
		cells = self.availableCells(nx,ny)
		if len(cells) == 0:
			self.x, self.y = nx, ny
			return
		
		cells.append((nx,ny))
		p = self.error_probs[self.map[ny][nx]]
		n = cells[resolveRandom([p/(len(cells)-1)]*(len(cells)-1)+[1-p])]
		self.x, self.y = n

	def move(self,action):
		if action == 'N':
			nx, ny = self.x, self.y-1
		elif action == 'S':
			nx, ny = self.x, self.y+1
		elif action == 'W':
			nx, ny = self.x-1, self.y
		else:
			nx, ny = self.x+1, self.y
		
		# ----- OUT -----
		if ny < 0 or ny == self.dimY or nx < 0 or nx == self.dimX or self.map[ny][nx] == 'W':
			return "Wall"

		self.probabilisticMove(nx,ny)

		#return(self.x,self.y)
		return self.get_state_obs()

	def run(self,number_steps,scheduler,with_action=True):
		self.reset()
		scheduler.reset()

		res = []
		current_len = 0
		while current_len < number_steps:
			action = scheduler.get_action()

			if with_action:
				res.append(action)
			observation = self.move(action)

			res.append(observation)
			scheduler.add_observation(observation)
			current_len += 1
		return res

	def printGrid(self):
		print('_'*(self.dimX*2+1))
		for y in range(self.dimY):
			print('|',end='')
			if y!= self.y:
				print('|'.join(self.map[y]),end="|\n")
			else:
				print('|'.join([self.map[y][i] if i != self.x else 'X' for i in range(self.dimX)]),end="|\n")
			if y < self.dimY-1:
				print('|'+'_'*(self.dimX*2-1)+'|')
		print('_'*(self.dimX*2+1))
		
