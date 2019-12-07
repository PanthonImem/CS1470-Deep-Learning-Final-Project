import random
import os
import numpy as np
import unit_env_test as test
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mgimg
from PIL import Image

class Agent(object):
    """
    Class describing agent in the game
        
    params:
     - id - id (useful for multi agent)
     - x,y - position of the agent
     - holding - None if holding nothing, else String of object that it's holding 
    """
    def __init__(self, id, starting_pos):
        self.id = id 
        self.y = starting_pos[0]
        self.x = starting_pos[1]
        self.holding = None #TODO: make it object in the future
    
    """
    check collision of the agent after a move and update the position if there is no collision

    params
     - dir - direction from 0-7 (0=South, Southeast, North, ... (counterclockwise))
     
     returns
     - new position of agent if it's not collide with anything
    """
    def move(self, dir):
            
        ydir = [-1,-0.707,0,+0.707,+1,+0.707,0,-0.707, 0]
        xdir = [0,+0.707,+1,+0.707,0,-0.707,-1,-0.707, 0]

        new_x = int(self.x + 20 * xdir[dir])
        new_y = int(self.y + 20 * ydir[dir])
       
        return new_x, new_y
    
    """"
    reset agent to a position

    params
    - ori_pos - position of the agent after reset

    """
    def reset(self, ori_pos):
        self.y = ori_pos[0]
        self.x = ori_pos[1]
        self.holding = None
    """
    Display information of the agent
    """
    def info(self):
        print('Agent Info:')
        print('X, Y: ', self.x, self.y)
        print('Holding: ', self.holding)
        print('------')

class GameObject(object):

    """
    Class describing the object in the game
        
    params
    - id - id of the object
    - x,y - position of the object
    - type - name of the object
    """
    def __init__(self, id, pos, type, size = 30, interact_range = 50):
        self.id = id
        self.y = pos[0]
        self.x = pos[1]
        self.type = type
        self.size = size
        self.int_range = interact_range

    """
    Interaction of the item with the agent
    
    params
     - agent - game agent
     - env - game environment
    
    return 
     - reward for that action
    """
    def interact(self, agent, env):
        return 0 
    """
    handle object collision
    
    params
     - x, y - position to check collision

    returns
     - reward
     - new_pos depending on whether the update successful
    """
    def collision(self, x, y, agent):
        dist = self.dist(x,y)
        if(dist < self.size):
            return -10, agent.x, agent.y
        else:

            return 0, x, y

    """
    Calculate distance to obejct
    """
    def dist(self, x, y):
         return np.sqrt((self.x-x)**2+(self.y-y)**2)
    
    """
    Get information about the object
    """
    def info(self):
        print('Object Info:')
        print('ID: ', self.id)
        print('X: ', self.x)
        print('Y: ', self.y)
        print('Type: ', self.type)


    

class Dispenser(GameObject):
    def __init__(self, id, pos, size = 30, interact_range = 50, food = 'Raw Salmon'):
       super().__init__(id, pos, 'Food_Dispenser', size)
       self.food = food
    
    def interact(self, agent, env):
        dist = self.dist(agent.x, agent.y)
        if dist < self.int_range:
            if agent.holding == None:
                agent.holding = self.food
                return 200
            else:
                return 0
        else:
            return 0
        
    def collision(self, x, y, agent):
        dist = self.dist(x,y)
        if(dist < self.size):
            return 0, agent.x, agent.y
        else:
            return 0, x, y

class ServingCounter(GameObject):
    def __init__(self, id, pos, size = 30, interact_range = 50):
       super().__init__(id, pos, 'Serving_Counter', size)
    
    def interact(self, agent, env):
        dist = self.dist(agent.x, agent.y)
        if dist < self.int_range:
            if agent.holding == env.order:
                agent.holding = None
                return 1000
            elif (agent.holding is not None):
                agent.holding = None
                return -200
            else :
                return -5
        else:
            return 0
        
    def collision(self, x, y, agent):
        dist = self.dist(x,y)
        if(dist < self.size):
            return 0, agent.x, agent.y
        else:
            return 0, x, y

        
class CuttingBoard(GameObject):
    def __init__(self, id, pos, size = 30, interact_range = 50):
       super().__init__(id, pos, 'Cutting_Board', size)
    
    def interact(self, agent, env):
        dist = self.dist(agent.x, agent.y)
        if dist < self.int_range:
            if agent.holding == 'Raw Salmon':
                agent.holding = 'Salmon Sashimi'
                return 350
            else:
                return 0
        else:
            return 0

class Frame(GameObject):
    def __init__(self, id, width, height):
       super().__init__(id, (0,0), 'Frame', 0)
       self.width = width
       self.height = height
    

    def collision(self, x, y, agent):
        if(x >= 0 and x < self.width and y >= 0 and y < self.height):
            # check object collision
            return 0, x, y
        else:
            return -10, agent.x, agent.y
    
    def dist(self, x, y):
        return float('inf')

class Wall(GameObject):
    def __init__(self, id, x1, x2):
       super().__init__(id, (0,0), 'Wall', 0)
       self.y1, self.x1 = x1
       self.y2, self.x2 = x2
       self.normal = np.array((self.y2 - self.y1, self.x1 - self.x2))
    
    """
    def collision(self, x, y, agent):
        rel1 = np.dot(np.array((x - self.x1, y - self.y1)), self.normal)
        rel2 = np.dot(np.array((agent.x - self.x1, agent.y - self.y1)), self.normal)
        if(rel1 * rel2 <= 0):
            # check object collision
            return -10, agent.x, agent.y
        else:
            return 0,x,y
    """
    def collision(self, x, y, agent):
        # m1 = (self.y1-self.y2)/(self.x1-self.x2+1e-10)
        # m2 = (y-agent.y)/(x-agent.x+1e-10)
        # b1 = self.y2 - m1*self.x2
        # b2 = y - (m2*x)

        # x_int  = (b2-b1)/(m2-m1+1e-10)
        # y_int = m1 * x_int + b1

        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        x_max = max(self.x1, self.x2)
        y_max = max(self.y1, self.y2)

        det = (x - agent.x)*(self.y2 - self.y1) - (y - agent.y)*(self.x2 - self.x1)
        if det != 0:
            t1 = ((agent.x - self.x1)*(self.y2 - self.y1) - (agent.y - self.y1)*(self.x2 - self.x1))/det
            x_int = agent.x - t1 * (x - agent.x)
            y_int = agent.y - t1 * (y - agent.y)
            if ((x_int > x_min and x_int < x_max) or (y_int > y_min and y_int < y_max)):
                return -10, agent.x, agent.y
        
        return 0,x,y
        

        
    def dist(self, x, y):
        return float('inf')


class Overcook(object):
    """
    Class describe environment for overcook game

    params:
     - height - height limit of the game frame
     - width - width limit of the game frame
     - time_limit - maximum time before the game end
     - agent - TODO: delete this
     - objectlist - Object in the game
     - order - order served in order to gain point
    """
    
    
    def __init__(self, height, width, time_limit, agent, objectlist, order):
        self.height = height
        self.width = width
        self.state_dim = 2
        self.objectlist = objectlist
        self.objectlist.append(Frame(id, width, height))
        self.time_limit = time_limit #in milliseconds
        self.agent = agent
        self.og_pos = (self.agent.y, self.agent.x)
        self.order = order
        self.time = 0
        self.cumulative_reward = 0
        self.num_action = 9
        self.possible_holding = [None,'Raw Salmon','Salmon Sashimi']
        self.history = []
        self.rewards = []
        self.holdings = []
    """
    reset the game
    """
    def reset(self):
        self.time = 0
        self.cumulative_reward = 0
        self.agent.reset(self.og_pos)
        self.history = []
        self.holdings = []
        return self.get_curr_state()
    

    
    """
    Go the the next state base on current state and action 
    
    params
     - action - an integer from 0 - 7 (move) or 8 (interact) represent the action 
    return 
     - state of the system 
     - reward value
     - boolean telling whether the game ended
    """
    def step(self, action):
        # starting reward
        reward = -1
        
        #update agent position
        new_x, new_y = self.agent.move(action)
        
        for object in self.objectlist:
            r, new_x, new_y = object.collision(new_x, new_y, self.agent)
            reward += r
        
        self.agent.x = new_x
        self.agent.y = new_y

        #update done
        done = False
        self.time = self.time + 1
        if (self.time >= self.time_limit):
            done = True
       
        #interact with closest object
        obj, _ = self.get_closest_object()
        if(action == 8):
            reward += obj.interact(self.agent, self)
        
        self.cumulative_reward += reward
        #self.show_game_stage()
        self.history.append([self.agent.x, self.agent.y])
        self.rewards.append(reward)
        self.holdings.append(self.agent.holding)
        return self.get_curr_state(), reward, done
    """
    Get internal game state. Use this to get initial game state
    """
    def get_curr_state(self):
        itemdict = {None:0, 'Raw Salmon':1, 'Salmon Sashimi':2}
        retls = []
        retls.append(self.agent.y)
        retls.append(self.agent.x)

        retls2 = []
        for _ in range(len(itemdict)):
            retls2.append(0)
        retls2[itemdict[self.agent.holding]] = 1
        return (retls, retls2)
    """
    Helper function for determining the closest object.
    Returns the closest object and distance
    """
    def get_closest_object(self):
        mindist = 9999999
        minobj = None
        for object in self.objectlist:
            dist = object.dist(self.agent.x, self.agent.y)
            if(dist<mindist):
                mindist = dist
                minobj = object
        return minobj, mindist
    """
    Print gane stage
    """
    def show_game_stage(self):
        color_dict = {'Dispenser':'blue', 'Serving Counter':'brown', 'Cutting Board':'green'}
        plt.scatter(self.agent.x, self.agent.y, s= 100, c = 'red')
        if(self.agent.holding == 'Raw Salmon'):
            plt.scatter(self.agent.x, self.agent.y, s= 10, c = 'orange')
        elif(self.agent.holding == 'Salmon Sashimi'):
            plt.scatter(self.agent.x, self.agent.y, s= 10, c = 'orange')
            plt.scatter(self.agent.x, self.agent.y, s= 3, c = 'magenta')
        for object in self.objectlist:
            plt.scatter(object.x, object.y, s= 900, c = color_dict[object.type])
            plt.text(object.x, object.y, object.type , fontsize=9, horizontalalignment='center')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.show()

"""
state example
"""

class stage_1(Overcook):
    def __init__(self):
        self.objectls = self.gen_stage()
        self.agent = Agent(0, (200, 300))
        super().__init__(400, 500, 210, self.agent, self.objectls, 'Raw Salmon')
        return
    def gen_stage(self):
        objectls = []
        objectls.append(Dispenser(0, (200,200), food = 'Raw Salmon', size = 20))
        objectls.append(ServingCounter(1, (200,400), size = 20))
        return objectls
class stage_2(Overcook):
    def __init__(self):
        self.objectls = self.gen_stage()
        self.agent = Agent(0, (200, 300))
        super().__init__(400, 500, 210, self.agent, self.objectls, 'Salmon Sashimi')
        return
    def gen_stage(self):
        objectls = []
        objectls.append(Dispenser(0, (200,200), food = 'Raw Salmon'))
        objectls.append(CuttingBoard(1, (260,300)))
        objectls.append(ServingCounter(2, (200,400)))
        return objectls
# class stage_3(Overcook):
#      def __init__(self):
#          self.objectls = self.gen_stage()
#          self.agent = Agent(0, (300, 100))
#          super().__init__(400, 400, 210, self.agent, self.objectls, 'Salmon Sashimi')
#          return
#      def gen_stage(self):
#          objectls = []
#          objectls.append(Dispenser(0, (250,150), food = 'Raw Salmon'))
#          objectls.append(CuttingBoard(1, (250,250)))
#          objectls.append(ServingCounter(2, (150,250)))
#          objectls.append(Wall(3, (50,200),(350,200)))
#          objectls.append(Wall(4, (200,50),(200,350)))

        #  return objectls
class stage_3(Overcook):
     def __init__(self):
         self.objectls = self.gen_stage()
         self.agent = Agent(0, (225, 75))
         super().__init__(300, 300, 210, self.agent, self.objectls, 'Salmon Sashimi')
         return
     def gen_stage(self):
         objectls = []
         objectls.append(Dispenser(0, (200, 100), food = 'Raw Salmon'))
         objectls.append(CuttingBoard(1, (200,200)))
         objectls.append(ServingCounter(2, (100,200)))
         objectls.append(Wall(3, (25,150),(145,150)))
         objectls.append(Wall(4, (150,25),(150,145)))
         objectls.append(Wall(5, (155,150),(275,150)))
         objectls.append(Wall(6, (150,155),(150,275)))

         return objectls
    
def animate_game(env, save = False):

    fig = plt.figure()
    ax = plt.axes(xlim=(0, env.width), ylim=(0, env.height))

    objs = []
    for object in env.objectlist:
        objs += ax.plot(object.x, object.y, 'o', markersize = 10, label = object.type)

    agent, = ax.plot([], [], 'o',lw=2, markersize = 10, label = 'agent')

    agentWith, = ax.plot([], [], 'o',lw=2, markersize = 10, label = 'agent(holding)')

    T_text = ax.text(0.05, 1.01, ' ', transform=ax.transAxes, fontsize = 16, color = 'k')
    

    # initialization function: plot the background of each frame
    def init():
        agent.set_data(0,0)
        agent.set_label('agent')
        for i, obj in  enumerate(objs):
            obj.set_data(env.objectlist[i].x,env.objectlist[i].y)
            obj.set_label(env.objectlist[i].type)
        agentWith.set_data([],[])
        agentWith.set_label('agent(holding)')
        T_text.set_text('')
        return agent, agentWith, objs, T_text

    # animation function.  This is called sequentially
    def animate(t, save = False):
        
        if env.holdings[t] == None:
            agent.set_data(env.history[t][0], env.history[t][1])
            agent.set_label('agent')
            agentWith.set_data([],[])
            agentWith.set_label('agent(holding)')
        else:
            agent.set_data([],[])
            agent.set_label('agent')
            agentWith.set_data(env.history[t][0], env.history[t][1])
            agentWith.set_label('agent(holding)')
        
        for i, obj in  enumerate(objs):
            obj.set_data(env.objectlist[i].x,env.objectlist[i].y)
            obj.set_label(env.objectlist[i].type)
        T_text.set_text('t = {} reward = {}'.format(t, env.rewards[t]))
        plt.legend()
        return agent, agentWith,  objs, T_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=200, interval=50, blit=False)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save:
        anim.save('trajectory.mp4', fps=20, extra_args=['-vcodec', 'libx264'], dpi = 300)

    
    plt.show()

def render(env, save_path = None):

    fig = plt.figure()
    ax = plt.axes(xlim=(0, env.width), ylim=(0, env.height))

    dir_path = os.path.dirname(os.path.realpath(__file__))

    objs = []
    for object in env.objectlist:
        if object.type != 'Frame' and object.type != 'Wall':
            img = Image.open(os.path.join(dir_path, 'graphics', object.type+'.png'))
            img.thumbnail((50, 50), Image.ANTIALIAS) 
            img = np.array(img)
            imgObj = ax.imshow(img,extent=[object.x - 25, object.x +img.shape[1] - 25, object.y - 25, object.y+img.shape[0] - 25], zorder=1)
            objs.append(imgObj)

        if object.type == 'Wall':
            ax.plot([object.x1, object.x2], [object.y1, object.y2])
    
    agent_img = Image.open(os.path.join(dir_path, 'graphics', 'Agent.png'))
    agent_img.thumbnail((50, 50), Image.ANTIALIAS) 
    agent_img = np.array(agent_img)
    agent = ax.imshow(agent_img,extent=[env.history[0][0]-25, env.history[0][0] +img.shape[1]-25, env.history[0][1]-25, env.history[0][1]+img.shape[0]-25], zorder=1)

    T_text = ax.text(0.05, 1.01, ' ', transform=ax.transAxes, fontsize = 16, color = 'k')
    
    
    # animation function.  This is called sequentially
    def animate(t, save = False):
        
       
        agent.set_extent([env.history[t][0] - 25, env.history[t][0] +img.shape[1] - 25, env.history[t][1] - 25, env.history[t][1]+img.shape[0] - 25])
        T_text.set_text('t = {} reward = {}'.format(t, env.rewards[t]))
        return agent,  objs, T_text

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,frames=200, interval=50, blit=False)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save_path is not None:
        anim.save(save_path, fps=20, extra_args=['-vcodec', 'libx264'], dpi = 300)

    
    plt.show()
def visualize(rewards, technique, save_path = None):
    avg_rewards = np.convolve(rewards, np.ones(50)/50, 'valid')
    plt.plot(rewards, 'blue', label = 'last episode')
    plt.plot(avg_rewards, 'red', label = 'avg. last 50 episodes')
    plt.title('Reward per episode: '+technique)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi = 300)

    plt.show()

if __name__ == '__main__':
    tester = test.unit_env_test()
    tester.test_stage_1()
    tester.test_stage_2()
