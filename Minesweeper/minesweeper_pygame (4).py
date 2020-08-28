import pygame
import math
import numpy as np
import random
import time
import mss

class Minesweeper(object):
    def __init__(self, ROWS = 16, COLS = 16, SIZEOFSQ = 100, MINES = 10, display = False, OUT = 'FULL', interval = 0,rewards = {"win" : 1, "loss" : -1, "progress" : 0.9, "noprogress" : -0.3, "YOLO" : -0.3}):

        """A base environment class for open ai gym.

        Attributes
        ----------
        ROWS, COLS, MINES : int 
            The number of ROWS,COLS,MINES in the game setup.
        actions : ndarray
            The current action space.
        display : bool
            Control Flag to display game.
        rewards : dict
            reward stuctre of the game setup.
        won : int
            number of wins in the current session
        lost : int
            number of loses in the current session
        """

        self.ROWS = ROWS
        self.COLS = COLS
        self.MINES = MINES
        self.OUT = OUT
        self.actions = np.array(range(ROWS*COLS))
        self.display = display
        self.rewards = rewards
        self.won = 0
        self.lost = 0

        self.interval = interval


        self.grid = np.zeros((self.ROWS, self.COLS), dtype=object)
        self.state = np.zeros((self.ROWS, self.COLS), dtype=object)
        self.state_last = np.copy(self.state)
        self.nbMatrix = np.zeros((ROWS, COLS), dtype=object)

        self.computeNeighbors() #Fills out the nbMatrix


        self.nb_actions = 0


        if display: #Loads pygame

            #Scales to the screen resolution
            with mss.mss() as sct:
                img = np.array(sct.grab(sct.monitors[1]))
                self.SIZEOFSQ = int(SIZEOFSQ * img.shape[1] / 3840)
                SIZEOFSQ = self.SIZEOFSQ

            pygame.init()
            pygame.font.init()

            self.myfont = pygame.font.SysFont("monospace-bold", SIZEOFSQ)
            self.screen = pygame.display.set_mode((COLS * SIZEOFSQ, ROWS * SIZEOFSQ))

        self.initGame()


    def initGame(self):
        if self.display:
            self.initPattern()
            pygame.display.set_caption('Minesweeper |            Won: {}   Lost: {}'.format(self.won, self.lost))
        self.grid = self.initBoard(startcol = 5, startrow = 5)
        self.state = np.ones((self.ROWS, self.COLS), dtype=object) * 'U'
        self.state_last = np.copy(self.state)


        self.action((5,5))  #Hack alert, to start off with non empty board. Can be removed but then agent has to learn
                            #what to do when the board starts out empty. 

    def initBoard(self, startcol, startrow):
        """ Initializes the board """

        COLS = self.COLS
        ROWS = self.ROWS
        grid = np.zeros((self.ROWS, self.COLS), dtype=object)
        mines = self.MINES

        #Randomly place bombs
        while mines > 0:
            (row, col) = (random.randint(0, ROWS-1), random.randint(0, COLS-1))
            #if (col,row) not in findNeighbors(startcol, startrow, grid) and grid[col][row] != 'B' and (col, row) not in (startcol, startrow):
            if (row,col) not in self.findNeighbors(startrow, startcol) and (row,col) != (startrow, startcol) and grid[row][col] != 'B':
                grid[row][col] = 'B'
                mines = mines - 1


        #Get rest of board when bombs have been placed
        for col in range(COLS):
            for row in range(ROWS):
                if grid[row][col] != 'B':
                    totMines = self.sumMines(col, row, grid)
                    if totMines > 0:
                        grid[row][col] = totMines
                    else:
                        grid[row][col] = 'E'


        return grid

    def computeNeighbors(self):
        """ Computes the neighbor matrix for quick lookups"""

        for row in range(self.ROWS):
            for col in range(self.COLS):
                self.nbMatrix[row][col] = self.findNeighbors(row, col)


    def findNeighbors(self, rowin, colin):
        """ Takes col, row and grid as input and returns as list of neighbors
        """
        COLS = self.grid.shape[1]
        ROWS = self.grid.shape[0]
        neighbors = []
        for col in range(colin-1, colin+2):
            for row in range(rowin-1, rowin+2):
                if (-1 < rowin < ROWS and 
                    -1 < colin < COLS and 
                    (rowin != row or colin != col) and
                    (0 <= col < COLS) and
                    (0 <= row < ROWS)):
                    neighbors.append((row,col))

        return neighbors


    def sumMines(self, col, row, grid):
        """ Finds amount of mines adjacent to a field.
        """
        mines = 0
        neighbors = self.findNeighbors(row, col)
        for n in neighbors:
            if grid[n[0],n[1]] == 'B':
                mines = mines + 1
        return mines


    def printState(self):
        """Prints the current state"""
        grid = self.state
        COLS = grid.shape[1]
        ROWS = grid.shape[0]
        for row in range(0,ROWS):
            print(' ')
            for col in range(0,COLS):
                print(grid[row][col], end=' ')


    def printBoard(self):
        """Prints the board """
        grid = self.grid
        COLS = grid.shape[1]
        ROWS = grid.shape[0]
        for row in range(0,ROWS):
            print(' ')
            for col in range(0,COLS):
                print(grid[row][col], end=' ')


    def reveal(self, col, row, checked, press = "LM"):
        """Finds out which values to show in the state when a square is pressed
           Checked : np.array((row,col)) to check which squares has already been checked
                     If the field is not a bomb we want to reveal it, if the field is empty
                     we want to find it's neighbors and reveal them too if they are not a bomb.   
        """
        if press == "LM":
            if checked[row][col] != 0:
                return
            checked[row][col] = checked[row][col] + 1
            if self.grid[row][col] != 'B':

                if self.display:
                    self.drawSpirit(col, row, self.grid[row][col])

                #Reveal to state space
                self.state[row][col] = self.grid[row][col]

                if self.grid[row][col] == 'E':
                    neighbors = self.findNeighbors(row, col)
                    for n in neighbors:
                        if not checked[n[0],n[1]]: 
                            self.reveal(n[1], n[0], checked)
        
        elif press == "RM":
            #Draw flag, not used for agent
            pass

            if self.display:
                self.drawSpirit(col, row, "flag")



    def action(self, a):
        """ External action, taken by human or agent
            a: tuple - row and column of the tile to act on
         """

        self.nb_actions += 1
        row, col = a[0], a[1]

        #If press a bomb game over, start new game and return bad reward, -10 in this case
        if self.grid[row][col] == "B":
            self.lost += 1
            self.initGame()
            return({"s" : np.copy(self.state), "r" :  self.rewards['loss'], "d" : True})

        #Take action and reveal new state
        self.reveal(col, row , np.zeros_like(self.grid))
        if self.display == True:
            pygame.display.flip()


        #Winning condition
        if np.sum(self.state == "U") == self.MINES:
            self.won += 1
            self.initGame()
            return({"s" : np.copy(self.state), "r" :  self.rewards['win'], "d" : True})

        #Get the reward for the given action
        reward = self.compute_reward(a)


        #return the state and the reward
        return({"s" : np.copy(self.state), "r" : reward, "d" : False})

    def compute_reward(self, a):
        """Computes the reward for a given action"""

        #Reward = 1 if we get less unknowns, 0 otherwise 
        if (np.sum(self.state_last == 'U') - np.sum(self.state == 'U')) > 0:
            reward =  self.rewards['progress']
        else:
            reward =  self.rewards['noprogress']

        #YOLO -> it it clicks on a random field with unknown neighbors
        tot = 0
        for n in self.nbMatrix[a[0],a[1]]:
            if self.state_last[n[0],n[1]] == 'U':
                tot += 1
        if tot == len(self.nbMatrix[a[0],a[1]]):
            reward = self.rewards['YOLO']

        self.state_last = np.copy(self.state)
        return(reward)

    def drawSpirit(self, col, row, type):
        """
        Draws a spirit at pos col, row of type = [E (empty), B (bomb), 1, 2, 3, 4, 5, 6, F (flag)]
        """
        myfont = self.myfont
        SIZEOFSQ = self.SIZEOFSQ

        if type == 'E':
            if self.checkpattern(col,row):
                c = (242, 244, 247)
            else:
                c = (247, 249, 252)
            pygame.draw.rect(self.screen, c, pygame.Rect(col*SIZEOFSQ, row*SIZEOFSQ, SIZEOFSQ, SIZEOFSQ))

        else:
            self.drawSpirit(col, row, 'E')
            if type == 1:
                text = myfont.render("1", 1, (0, 204, 0))
            elif type == 2:
                text = myfont.render("2", 1, (255, 204, 0))
            elif type == 3:
                text = myfont.render("3", 1, (204, 0, 0))
            elif type == 4:
                text = myfont.render("4", 1, (0, 51, 153))
            elif type == 5:
                text = myfont.render("5", 1, (255, 102, 0))
            elif type == 6:
                text = myfont.render("6", 1, (255, 102, 0))
            elif type == 'flag':
                text = myfont.render("F", 1, (255, 0, 0))

            #Get the text rectangle and center it inside the rectangles
            textRect = text.get_rect()
            textRect.center = (col*self.SIZEOFSQ + int(0.5*self.SIZEOFSQ)),(row*self.SIZEOFSQ + int(0.5*self.SIZEOFSQ))
            self.screen.blit(text, textRect)
            


    def checkpattern(self, col, row):
        #Function to construct the checked pattern in pygame
        if row % 2:
            if col % 2: #If unequal
                return True
            else: #if equal
                return False
        else: 
            if col % 2: #If unequal
                return False
            else: #if equal
                return True

    def initPattern(self):
        #Initialize pattern:

        c1 = (4, 133, 223) #color1
        c2 = (4, 145, 223) #color2
        rects = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.checkpattern(col, row):
                    c = c1
                else:
                    c = c2

                rects.append(pygame.draw.rect(self.screen, c, pygame.Rect(col*self.SIZEOFSQ, row*self.SIZEOFSQ, self.SIZEOFSQ, self.SIZEOFSQ)))

        pygame.display.flip()

    def get_state(self):
        return np.copy(self.state)

    def stateConverter(self, state):
        """ Converts 2d state to one-hot encoded 3d state
            input: state (rows x cols)
            output: state3d (row x cols x 10) (if full)
                            (row x cols x 2) (if condensed)
                            (row x cols x 1) (if image)

        """
        rows, cols = state.shape
        if self.OUT == "FULL":
            res = np.zeros((rows,cols,10), dtype = int)
            for i in range(0,8):
                res[:,:,i] = state == i+1 #1-7
            res[:,:,8] = state == 'U'
            res[:,:,9] = state == 'E'
           
            return(res)
        elif self.OUT == "CONDENSED":
            #Outputs a condensed representation of nxmx2
            #First layer is the value of the intergers
            #Second layer is true if field is empty, 0 otherwise

            res = np.zeros((rows, cols, 2))
            filtr = ~np.logical_or(state == "U", state == "E") #Not U or E
            res[filtr,0] = state[filtr] / 4
            res[state == "U", 1] = 1
            return(res)

        elif self.OUT == "IMAGE":
            #Outputs an image
            res = np.zeros((rows, cols,1))
            res[state == "U", 0] = -1
            res[state == "E", 0] = 0
            filtr = ~np.logical_or(state == "U", state == "E") #Not U or E
            res[filtr, 0] = state[filtr] / 8
            return(res)


    def get_validMoves(self):
        
        return(self.state == "U") #All unknowns are valid moves

    # OpenAI GYM Wrapper API
    def step(self, a):
        
        """
        Takes an action and returns observations, rewards, done flag, info (Similar to OpenAI Gym ennv)
        """
        a = np.unravel_index(a, (self.ROWS,self.COLS))
        d = self.action(a)
        d["s"] = self.stateConverter(d["s"])
        
        return d["s"], d["r"], d["d"], None

    def reset(self):
        
        """
        Resets the current episode
        """
        self.initGame()
        
        return self.stateConverter(self.state)

    def get_nbactions(self):

        return(self.nb_actions)

    def action_space(self) :
        """
        Returns the Current possible actions in a numpy array
        """

        self.actions = np.ravel_multi_index(np.nonzero(self.get_validMoves()), (self.ROWS, self.COLS))
        
        return np.copy(self.actions)

# This block is for testing purposes only, plis comment it out if necessary

if __name__ == "__main__":

    display = True
    game = Minesweeper(display=display)
    game.printState()

    start = time.time()
    i = 0
    while True:
        if display :
            pygame.event.get()
        # print(game.action_space())
        # action = int(input("Enter an Action : ")) 
        # s, r, d, _ = game.step(action)
        # game.printState()
        # print("\nReward = {}".format(r))

        #Test how fast it can run by performing 10000 actions:

        i += 1
        print(i)
        action = random.choice(game.action_space())
        game.step(action)

        if i >= 10000:
            break

    print("Games Won : ", game.won)
    print("Games Lost : ", game.lost)
    print("Took: " + str(time.time()-start) + "seconds")


