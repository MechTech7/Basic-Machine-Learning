import numpy as np
#this is an implementation of the k-armed reinforcement learning bandit problem
#the rewards are stationary
class Problem:
    def __init__(self, arm_count = 10):
        self.arm_count = arm_count
        #this is a list of all of the true rewards for the levers
        self.lever_rewards = self.gaussian_list(self.arm_count)
    def gaussian_list(self, count):
        op_list = []
        for i in range(count):
            op_list.append(np.random.randn())
        return op_list
    def return_reward(self, action_index):
        return self.lever_rewards[action_index]
class Learner:
    def __init__(self, problem, e_value):
        self.problem = problem
        #this has to be a number between 0 and 1
        self.e_value = e_value
        self.rewards = [0.5] * len(problem.lever_rewards)
        self.step = 1
    def perform_action(self):
        print("before action: " + str(self.rewards))
        optimal_index = self.get_greatest_lever()
        action = optimal_index
        if self.e_result() == True:
            #explore: choose a lever that isn't the optimal one
            print("gone exploring")
            r_len = len(self.rewards)
            reduced = range(r_len)
            reduced.remove(optimal_index)
            action = np.random.choice(reduced)

        self.step += 1
        reward = self.problem.return_reward(action)
        print("reward: " + str(reward))
        self.update_rewards(action, reward)
        print("after action: " + str(self.rewards))
    def update_rewards(self, action_index, new_reward):
        first_part = (self.rewards[action_index] * (self.step - 1) + new_reward)
        self.rewards[action_index] = first_part / self.step
        
    def get_greatest_lever(self):
        greatest_index = 0
        for i in range(len(self.rewards)):
            if self.rewards[i] > greatest_index:
                greatest_index = i
        return greatest_index
    def e_result(self):
        options = [True, False]
        op = np.random.choice(options, p = [self.e_value, 1 - self.e_value])
        return op

p = Problem(arm_count=5)
a = Learner(p, 0.1)

for i in xrange(10000):
    a.perform_action()
print("true rewards: " + str(p.lever_rewards))
