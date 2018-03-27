import numpy

#this is a really basic markov descision process for an agent to move across a grid to a goal

#parts:
#1. the Agent: estimates reward possibility, decides action,

'''
-------
|0|1|2|
|3|4|5|
-------

0, 1, 2, 3, 4
5, 6, 7, 8, 9
10, 11, 12, 13

TODO: Define a system for grid layout. Each position has it's own number

'''
class Grid:
    def __init__(self, x_dim=3, y_dim=2):
        self.x_len = x_dim
        self.y_len = y_dim
        self.generate_grid()

        #sets the goal state to the bottom right corner
        self.goal_state = (x_dim * y_dim) - 1

        #the agent starts off in the top left
        self.current_position = 0

    def generate_grid(self):
        self.grid = []
        count = 0
        for i in range(self.y_len):
            row_arr = []
            for j in range(self.x_len):
                row_arr.append(count)
                count += 1
            self.grid.append(row_arr)
    def print_stacked_grid(self):
        for i in range(self.y_len):
            print(self.grid[i])
    def reward(self, state):
        if state == self.goal_state:
            return 10
        #a minus 1 reward is returned for every state that is not the goal state
        return 0
    def get_state_count(self):
        count = self.x_len * self.y_len
        return count
    def get_possible_states(self, state):
        #the agent can move 1 left, 1 right, 1 up, 1 down, or stay in the same place
        pos = state#self.current_position
        row_len = self.x_len
        col_len = self.y_len
        states = [pos - 1, pos + 1, pos - row_len, pos + row_len, pos]
        #validate whether each state in the states array is a possible move
        op = list(states)
        print("state: " + str(state))
        for i in states:
            valid = (i / self.x_len) == (pos / row_len) or (i % row_len) == (pos % row_len)
            if valid == False or i < 0 or i > (row_len * col_len) - 1:
                print("removed: " + str(i))
                op.remove(i)

        print(op)
        return op

class Agent:
    def __init__(self, inp_env):
        self.environment = inp_env
        self.utilities = []
        self.initialize_utilities()
    def initialize_utilities(self):
        #each state has it's own reward
        print("DAMN")
        state_count = self.environment.get_state_count()
        self.utilities = [0] * state_count
    def navigate_grid(self):
        visited_arr = []
        reward = 0
        while reward != 10:
            pos = self.environment.current_position
            possible = self.environment.get_possible_states(pos)
            best = possible[0]
            top_util = -1
            for i in possible:
                util = self.utilities[i]
                if util > top_util:
                    top_util = util
                    best = i
                    print(i)
            self.environment.current_position = best
            reward = self.environment.reward(best)
            visited_arr.append(best)
            print(best)
        print("route: " + str(visited_arr))

    def iterate_utilities(self):
        #this method works except for when the reward for points that aren't the reward is negative 1
        for _ in xrange(100):
            utilities = list(self.utilities)
            for i in range(len(utilities)):
                neighbors = self.environment.get_possible_states(i)
                added = 0
                for j in neighbors:
                    update_amount = self.environment.reward(j) + 0.1 * utilities[j]
                    added += update_amount

                    print("state: " + str(i) + " neighbor: " + str(j) + " neighbor_util: " + str(utilities[j]))
                    print("updated by: " + str(update_amount))
                utilities[i] = added

            self.utilities = list(utilities)
    def get_utility(self, state):
        return self.utilities[i]


        #Finish working on the Agent

viro = Grid(10, 30)
print(viro.current_position)
viro.print_stacked_grid()
print("----------------------------------------------------")
#print(viro.get_possible_states(viro.current_position))
#print(viro.reward(8))

#TODO: states farther from the reward have the same utility
gent = Agent(viro)

gent.iterate_utilities()
print(gent.utilities)
gent.navigate_grid()


viro.print_stacked_grid()
