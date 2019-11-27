import random
import numpy as np
import env2 as Env
import Action2
class unit_env_test:
    def __init__(self):
        return
    def test_summary(self, env, rewardls):
        time, grid, order = env.get_curr_state()
        print('Time Limit: ', env.time_limit)
        print('Time Taken: ', time)
        print('Order: ', order)
        print('Reward History:', rewardls)
    def test_stage_1(self):
        print('Run Unit Test 1')
        print('Agent run to Dispenser, grab a raw fish, run to counter and deliver')
        env = Env.stage_1()
        time, grid, order = env.get_curr_state()
        rewardls = []
        #test1
        action = Action2.Action(5,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(3,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(3,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(3,1)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(7,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(7,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(7,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(5,0)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        action = Action2.Action(5,1)
        obv, reward, done = env.step(action)
        rewardls.append(reward)

        self.test_summary(env, rewardls)
        #if(rewardls[-1] == 100):
            #print('Test 1 Pass!')
    def test_stage_2(self):
        print('Run Unit Test 2')
        print('Agent run to Dispenser, grab a raw fish, run to cutting board,')
        print('cut fish, run to counter and deliver')
        env = Env.stage_2()
        time, grid, order = env.get_curr_state()
        print('Order', order)
        rewardls = []
        #test1
        for i in range(9):
            action = Action.Action(2,3,0)
            obv, reward, done = env.step(action)
            rewardls.append(reward)
        action = Action.Action(2,3,1)
        obv, reward, done = env.step(action)
        rewardls.append(reward)
        for i in range(9):
            action = Action.Action(0,3,0)
            obv, reward, done = env.step(action)
            rewardls.append(reward)
        action = Action.Action(0,3,1)
        obv, reward, done = env.step(action)
        for i in range(9):
            action = Action.Action(0,3,0)
            obv, reward, done = env.step(action)
            rewardls.append(reward)
        action = Action.Action(0,3,1)
        obv, reward, done = env.step(action)
        rewardls.append(reward)
        self.test_summary(env, rewardls)
        if(rewardls[-1] == 100):
            print('Test 2 Pass!')
