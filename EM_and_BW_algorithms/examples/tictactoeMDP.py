from random import randint
class TicTacToe:

	def __init__(self):
		#set of labels = [1,2,3,4,5,6,7,8,9,error,win,loose,draw]
		#set of actions= [1,2,3,4,5,6,7,8,9]
		self.reset()


	def reset(self):
		self.cells = [0,0,0,
					  0,0,0,
					  0,0,0]
		self.game_state = "playing"

	def printGrid(self):
		for i in range(9):
			if i % 3 == 0:
				print()
			print(self.cells[i],end=' ')
		print('\n')


	def observations(self):
		return ['1','2','3','4','5','6','7','8','9',"error","win","loose","draw"]

	def actions(self):
		return ['1','2','3','4','5','6','7','8','9']

	def checkVictory(self,last_token,player):
		# vertical line
		if (self.cells[(last_token+3)%9] == player) and (self.cells[(last_token+6)%9] == player):
			return True
		# horizontal line
		if (self.cells[int(last_token//3)*3+(last_token+1)%3] == player) and (self.cells[int(last_token//3)*3+(last_token+2)%3] == player):
			return True
		#top left to bottom right
		if (last_token%4 == 0):
			if self.cells[0] == player and self.cells[4] == player and self.cells[8] == player:
				return True
		#top right to bottom left
		if (last_token%2 == 0):
			if self.cells[2] == player and self.cells[4] == player and self.cells[6] == player:
				return True
		return False

	def roundMove(self,postition):
		postition = int(postition)
		postition -= 1

		if self.game_state != "playing":
			return self.game_state
		
		if self.cells[postition] != 0:
			return "error"
		self.cells[postition] = 1
		
		if self.checkVictory(postition,1):
			self.game_state = "win"
			return "win"
		
		if self.cells.count(0) == 0:
			self.game_state = "draw"
			return "draw"
		
		opponent = int(randint(0,self.cells.count(0)-1))
		i = -1
		while opponent > -1:
			i += 1
			if self.cells[i] == 0:
				opponent -=1
		self.cells[i] = 2
		
		if self.checkVictory(i,2):
			self.game_state = "loose"
			return "loose"

		return str(i+1)

	def run(self,number_steps,scheduler,with_action=True):
		self.reset()
		scheduler.reset()

		res = []
		current_len = 0

		while current_len < number_steps:
			action = scheduler.get_action()

			if with_action:
				res.append(action)
			observation = self.roundMove(action)

			res.append(observation)
			scheduler.add_observation(observation)
			current_len += 1
		return res
