from gym import Env, spaces
import numpy as np
import gym
from gym.spaces import Box, Discrete

# after selecting item, it is removed from the original problem  instance and reduced capacity is generated
#state: v_i, w_i + 4 (total value, total weight, capacity, number of elements)

class KnapsackEnv(gym.Env):
    def __init__(self, N0, T, item_distributions, m):
        self.N0 = N0  # Initial capacity of the knapsack
        self.T = T    # Deadline or max number of timesteps
        self.item_distributions = item_distributions  # A function to generate items
        self.m = m  # Number of past items to remember
        self.p = 0 #penalty for rejecting an item or then capacity does not enough to insert

        self.action_space = Discrete(2)
        low_bounds = [0] * (6 + 2 * m)
        high_bounds = [N0, N0, float('inf'), T, float('inf'), float('inf')] + [float('inf')] * (2 * m)
        self.observation_space = Box(low=np.array(low_bounds), high=np.array(high_bounds), dtype=np.float32)

        self.reset()

    def reset(self):
        self.timestep = 0
        self.used_capacity = 0
        self.total_value = 0
        self.last_m_weights = [0] * self.m
        self.last_m_values = [0] * self.m
        self.current_item = self.item_distributions()
        return self.get_state()

    def get_state(self):
        available_capacity = self.N0 - self.used_capacity
        return [available_capacity, self.used_capacity, self.total_value,
                self.timestep, self.current_item.weight, self.current_item.value] + self.last_m_weights + self.last_m_values

    def step(self, action):
        # Action 0 is reject, action 1 is accept
        reward = 0
        if action == 1 and self.can_accept(self.current_item):
            self.used_capacity += self.current_item.weight
            reward = self.current_item.value
            self.total_value += reward

            # Update history
            self.last_m_weights.pop(0)
            self.last_m_values.pop(0)
            self.last_m_weights.append(self.current_item.weight)
            self.last_m_values.append(self.current_item.value)
        else:
            reward = -self.p

        # Generate the next item and move to the next timestep
        self.current_item = self.item_distributions()
        self.timestep += 1
        done = self.timestep >= self.T


        return self.get_state(), reward, done, {}

    def can_accept(self, item):
        return self.used_capacity + item.weight <= self.N0