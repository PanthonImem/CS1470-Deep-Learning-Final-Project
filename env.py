from tkinter import *
import time
import random
import numpy as np
import unit_env_test as test
class overcook_env:
    class Agent:
        def __init__(self, id, starting_pos):
            self.id = id
            self.y = starting_pos[0]
            self.x = starting_pos[1]
            self.holding = None
        def move(self, dir, lim):
            ylim, xlim = lim
            xdir = [0,+0.707,+1,+0.707,0,-0.707,-1,-0.707]
            ydir = [-1,-0.707,0,+0.707,+1,+0.707,0,-0.707]
            new_x = int(self.x + 20 * xdir[dir])
            new_y = int(self.y + 20 * ydir[dir])
            if(new_x>=0 and new_x< xlim and new_y>0 and new_y<ylim):
                self.x = new_x
                self.y = new_y
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
    def __init__(self, height, width, time_limit, agent, objectlist, order):
        self.height = height
        self.width = width
        self.objectlist = objectlist
        self.time_limit = time_limit #in milliseconds
        self.agent = agent
        self.og_pos = (self.agent.y, self.agent.x)
        self.order = order
        self.time = 0
        self.cumulative_reward = 0
    def reset(self):
        self.time = 0
        self.cumulative_reward = 0
        self.agent = self.Agent(0, self.og_pos)
        return self.get_curr_state()
    """
    Action = 8 means interact
    """
    def step(self, action):

        reward = -1
        #update agent position
        if(action >=0 and action <= 7):
            self.agent.move(action,(self.height, self.width))
        #update done
        done = False
        self.time = self.time+1
        if(self.time >= self.time_limit):
            done = True
        #interact with closest object
        obj, dist = self.get_closest_object()
        if(action == 8 and dist < 50):
            #If object is dispenser, get ingredient
            if(obj.type == 'Dispenser' and self.agent.holding == None):
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
                        reward = 35
        self.cumulative_reward += reward
        return self.get_curr_state(), reward, done
    """
    Get internal game state. Use this to get initial game state
    """
    def get_curr_state(self):
        itemdict = {None:2, 'Raw Salmon':3, 'Salmon Sashimi':4}
        retls = []
        retls.append(self.agent.y)
        retls.append(self.agent.x)
        for i in range(len(itemdict)):
            retls.append(0)
        retls[itemdict[self.agent.holding]]
        return retls
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
class stage_1(overcook_env):
    def __init__(self):
        self.objectls = self.gen_stage()
        self.agent = self.Agent(0, (200, 300))
        super().__init__(600, 800, 210, self.agent, self.objectls, 'Raw Salmon')
        return
    def gen_stage(self):
        objectls = []
        objectls.append(self.Object(0, (200,200), 'Dispenser'))
        objectls.append(self.Object(1, (200,400), 'Serving Counter'))
        return objectls
class stage_2(overcook_env):
    def __init__(self):
        self.objectls = self.gen_stage()
        self.agent = self.Agent(0, (200, 300))
        super().__init__(600, 800, 210, self.agent, self.objectls, 'Salmon Sashimi')
        return
    def gen_stage(self):
        objectls = []
        objectls.append(self.Object(0, (200,200), 'Dispenser'))
        objectls.append(self.Object(1, (220,300), 'Cutting Board'))
        objectls.append(self.Object(2, (200,400), 'Serving Counter'))
        return objectls
if __name__ == '__main__':
    tester = test.unit_env_test()
    tester.test_stage_1()
    tester.test_stage_2()
