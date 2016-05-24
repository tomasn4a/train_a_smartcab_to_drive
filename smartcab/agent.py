import numpy as np
import random
import itertools
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import operator
import math

# np.random.seed(2)
# random.seed(2)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.Q_table = defaultdict(lambda: defaultdict(int)) # table containing Q values
        self.state_prev = [] # we will store the previous state in this variable
        self.counter = 0 # a counter that keeps track of time steps
        self.rewards = [] # an array of the rewards
        self.actions = [] # an array of actions
        self.max_deadline = [] # this variable stores the max deadline, to be updated later
        self.rew_total = [] # an array of the rewards per trial
        self.trial_count = 0 # a counter to keep track of the trial number
        self.steps = []
        self.global_counter = 0

    def reset(self, destination=None):
        random.seed(self.trial_count + 1)
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state_prev = []
        self.steps.append(self.counter)
        self.counter = 0
        self.rew_total.append(sum(self.rewards))
        self.rewards = []
        self.actions = []
        self.trial_count += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        #######################################################################

        # TODO: Update state

        # STATE SPACE 3
        self.state = [('next', self.next_waypoint),('light', inputs['light'])]
        if inputs['left'] == None or inputs['left'] == 'left' or inputs['left'] == 'right':
            self.state.append(('left', 'N|L|R'))
        else:
            self.state.append(('left', 'F'))
        if inputs['oncoming'] == None or inputs['oncoming'] == 'left':
            self.state.append(('forward', 'N|L'))
        else:
            self.state.append(('forward', 'R|F'))

        # # STATE SPACE 2
        # self.state = [('next', self.next_waypoint),('light', inputs['light']),('left',inputs['left']),('right',inputs['right'])]

        # NOTE: If we wanted we could an extra dimension to our state spaces by
        #       taking the deadline parameter into account as below:
        # Set lambda parameter for deadline
        # ld = 5
        # if deadline <= 5:
        #     self.state.append(('deadline','close'))
        # else:
        #     self.state.append(('deadline', 'far'))

        #######################################################################

        # TODO: Select action according to your policy

        # MODEL: Using Boltzmann's explotarion
        tau = 0.4 # Boltzmann's temperature
        if self.Q_table[str(self.state)] == {}:
            action = random.choice(self.env.valid_actions)
        else:
            possible_actions = []
            probs = []
            for k in self.env.valid_actions:
                probs.append(math.exp((self.Q_table[str(self.state)][k])/tau))
                sum_probs = sum(probs)
                probs_norm = [x/sum_probs for x in probs]
            action = np.random.choice(self.env.valid_actions, p=probs_norm)
        self.actions.append(action)

        # # BENCHMARK: Always choose the best action
        # if self.next_waypoint == 'left':
        #     if inputs['light'] == 'red':
        #         action = None
        #     else:
        #         if inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right':
        #             action = None
        #         else:
        #             action = 'left'
        # elif self.next_waypoint == 'forward':
        #     if inputs['light'] == 'green':
        #         action = 'forward'
        #     else:
        #         action = None
        # else:
        #     if inputs['light'] == 'red':
        #         if inputs['left'] == 'forward':
        #             action = None
        #         else:
        #             action = 'right'
        #     else:
        #         action = 'right'
        # self.actions.append(action)

        # # RANDOM: Always choose a random action
        # action = random.choice(self.env.valid_actions)
        # self.actions.append(action)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.rewards.append(reward)

        #######################################################################

        #TODO: Learn policy based on state, action, reward
        gamma = 0.
        alpha = pow((1-(self.global_counter/3000)),2) # decaying learning rate 1
        #alpha = pow(np.cos(np.pi*self.global_counter/6000),2) # decaying learning rate 2
        if self.counter == 0:
            self.max_deadline.append(deadline)
        else:
            if deadline < 0: # we don't allow negative alphas during training, when deadline isn't enforced
                alpha = 0.
            Q_value_update = self.rewards[self.counter-1] + gamma*max(list(self.Q_table[str(self.state)].values()) or [0])
            Q_value_current = self.Q_table[str(self.state_prev)][self.actions[self.counter-1]]
            self.Q_table[str(self.state_prev)][self.actions[self.counter-1]] = (1-alpha)*Q_value_current + alpha*Q_value_update
        self.state_prev = self.state

        self.counter+=1
        self.global_counter += 1

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    print a.max_deadline
    print a.steps
    print a.rew_total

if __name__ == '__main__':
    run()
