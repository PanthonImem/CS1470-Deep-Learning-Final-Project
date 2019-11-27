from tkinter import *
import time
import random
import numpy as np
import unit_env_test2 as test
import Action2
import string
def showgrid(grid):
    for row in range(len(grid)):
        print(grid[row])
class overcook_env:
    class Agent:
        def __init__(self, id, starting_pos):
            self.id = id
            self.y = starting_pos[0]
            self.x = starting_pos[1]
            self.holding = None
            self.holding_list = ['Raw Salmon', 'Salmon Sashimi']
        def move(self, dir, info):
            max_y, max_x, grid = info
            xdir = [0,+1,+1,+1,0,-1,-1,-1]
            ydir = [-1,-1,0,+1,+1,+1,0,-1]
            new_x = self.x + xdir[dir]
            new_y = self.y + ydir[dir]
            if((new_x>=0)
                & (new_x< max_x)
                & (new_y>=0)
                & (new_y<max_y)
                ):
                if((grid[new_y][new_x] == 0)):
                    grid[new_y][new_x] = 1
                    grid[self.y][self.x] = 0
                    self.x = new_x
                    self.y = new_y
            else:
                print('Move invalid')
            showgrid(grid)
            return grid
        def info(self):
            print('Agent Info:')
            print('X, Y: ', self.x, self.y)
            print('Holding: ', self.holding)
            print('------')
    class Object:
        def __init__(self, id, pos, type):
            self.id = id
            self.y = pos[0]
            self.x = pos[1]
            self.type = type
        def get_item(self):
            if(self.type == 'Dispenser'):
                return 'Raw Salmon'
        def info(self):
            print('Object Info:')
            print('ID: ', self.id)
            print('X: ', self.x)
            print('Y: ', self.y)
            print('Type: ', self.type)
    def __init__(self, time_limit, grid, order):
        self.grid = grid
        self.parse_grid(self.grid)
        self.time_limit = time_limit #in milliseconds
        self.order = order
        self.time = 0
        self.cumulative_reward = 0
    def parse_grid(self,grid):
        objlist = []
        self.height = len(grid)
        self.width = len(grid[0])
        obj_id = 0
        ag_id = 0
        itemdict = {2:'Dispenser', 3:'Serving Counter', 4:'Cutting Board'}
        for row in range(self.height):
            for col in range(self.width):
                if(grid[row][col]== 1 ):
                    self.agent = self.Agent(ag_id, (row,col))
                    ag_id += 1
                elif(grid[row][col] > 1):
                    objlist.append(self.Object(obj_id, (row,col), itemdict[grid[row][col]]))
                    obj_id += 1
        self.objectls = objlist

    def reset(self):
        self.time = 0
        self.cumulative_reward = 0
        self.agent = self.Agent(0, (self.height/2, self.width/2))
        return self.get_curr_state()
    def update_ui(self):
        pass
    def step(self, action):
        reward = -1
        #update agent position
        self.grid = self.agent.move(action.dir, (self.height, self.width, self.grid))
        #move and update grid
        #update done
        done = False
        self.time = self.time+1
        if(self.time >= self.time_limit):
            done = True
        #interact with closest object
        obj, dist = self.get_closest_object()
        if(action.interact == 1 and dist < 1.5):   #1.5>sqrt(2)
            #If object is dispenser, get ingredient
            if(obj.type == 'Dispenser'):
                self.agent.holding = obj.get_item()
                reward = 20
            #If object is serving counter, get reward based on correctness
            elif(obj.type == 'Serving Counter'):
                if(self.agent.holding is not None):
                    if(self.agent.holding == self.order):
                        reward = 100
                    else:
                        reward = -50
                    self.agent.holding = None
            #If object is Cutting Board, turn raw salmon to salmon sashimi
            elif(obj.type == 'Cutting Board'):
                if(self.agent.holding is not None):
                    if(self.agent.holding == 'Raw Salmon'):
                        self.agent.holding = 'Salmon Sashimi'
        self.cumulative_reward += reward
        return self.get_curr_state(), reward, done

    """
    Get internal game state. Use this to get initial game state
    """
    def get_curr_state(self):
        return (self.time ,self.grid, self.order)
    """
    Helper function for determining the closest object.
    Returns the closest object and distance
    """
    def get_closest_object(self):
        mindist = 9999999
        minobj = None
        for object in self.objectls:
            dist = np.sqrt((object.x-self.agent.x)**2+(object.y-self.agent.y)**2)
            if(dist<mindist):
                mindist = dist
                minobj = object
        return minobj, mindist

    def get_dim_state(self):
        return 2, (len(self.agent.holding_list) + 1)

    def hold_to_int(self):
        dict =  {name: i+1 for (i,name) in enumerate(self.agent.holding_list)}
        dict[None] = 0
        return dict
    def int_to_hold(self):
        dict =  {i+1: name for (i,name) in enumerate(self.agent.holding_list)}
        dict[0] = None
        return dict

class stage_1(overcook_env):
    def __init__(self):
        grid = [[0,1,0,0,0,0],
                [0,0,0,0,0,2],
                [0,0,0,0,0,0],
                [3,0,0,0,0,0]]
        super().__init__(210, grid, 'Raw Salmon')
        return
class stage_2(overcook_env):
    def __init__(self):
        grid = [[0,4,0,0,0,2],
                [0,0,0,0,1,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,3]]
        super().__init__(210, grid, 'Salmon Sashimi')
        return
if __name__ == '__main__':
    tester = test.unit_env_test()
    tester.test_stage_1()
    tester.test_stage_2()
