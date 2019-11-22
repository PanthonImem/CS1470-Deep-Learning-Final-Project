from tkinter import *
import time
import random
import numpy as np
import unit_env_test as test
import Action
class overcook_env:
    class Agent:
        def __init__(self, id, starting_pos):
            self.id = id
            self.x = starting_pos[0]
            self.y = starting_pos[1]
            self.holding = None
        def move(self, dir, mov):
            xdir = [0,+1,0,-1]
            ydir = [+1,0,-1,0]
            self.x = int(self.x + (mov+1) * 5 * xdir[dir])
            self.y = int(self.y + (mov+1) * 5 * ydir[dir])
        def info(self):
            print('Agent Info:')
            print('X, Y: ', self.x, self.y)
            print('Holding: ', self.holding)
            print('------')
    class Object:
        def __init__(self, id, pos, type):
            self.id = id
            self.x = pos[0]
            self.y = pos[1]
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
        self.order = order
        self.time = 0
        self.cumulative_reward = 0
    def reset(self):
        self.time = 0
        self.cumulative_reward = 0
        self.agent = Agent(0, (height/2, width/2))
    def update_ui(self):
        pass
    def step(self, action):
        reward = -1
        #update agent position
        self.agent.move(action.dir, action.mov)
        #update done
        done = False
        self.time = self.time+1
        if(self.time >= self.time_limit):
            done = True
        #interact with closest object
        obj, dist = self.get_closest_object()
        if(action.interact == 1 and dist < 40):
            #If object is dispenser, get ingredient
            if(obj.type == 'Dispenser'):
                self.agent.holding = obj.get_item()
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
        return (self.time ,self.agent, self.objectls, self.order)
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
        self.agent = self.Agent(0, (200, 400))
        super().__init__(600, 800, 210, self.agent, self.objectls, 'Raw Salmon')
        return
    def gen_stage(self):
        objectls = []
        objectls.append(self.Object(0, (200,200), 'Dispenser'))
        objectls.append(self.Object(1, (200,600), 'Serving Counter'))
        return objectls
class stage_2(overcook_env):
    def __init__(self):
        self.objectls = self.gen_stage()
        self.agent = self.Agent(0, (200, 400))
        super().__init__(600, 800, 210, self.agent, self.objectls, 'Salmon Sashimi')
        return
    def gen_stage(self):
        objectls = []
        objectls.append(self.Object(0, (200,200), 'Dispenser'))
        objectls.append(self.Object(1, (200,400), 'Cutting Board'))
        objectls.append(self.Object(2, (200,600), 'Serving Counter'))
        return objectls
if __name__ == '__main__':
    tester = test.unit_env_test()
    tester.test_stage_1()
    tester.test_stage_2()
