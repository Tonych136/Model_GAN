"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pickle
from numpy import linalg as LA
from cartpole import CartPoleEnv

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class ModelParamEnv(gym.Env):
    def __init__(self, gravity = 9.8, masscart = 1.0, masspole = 0.1, 
                        length = 0.5, force_mag = 10.0):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.force_mag = force_mag

        self.state = np.asarray([gravity, masscart, masspole, length, force_mag])

        self.action_space = spaces.Box( low=np.array([-0.1]*5), 
                                        high=np.array([0.1]*5), dtype=np.float32)
        record = open('record.pickle', 'rb')
        data = pickle.load(record)
        record.close()
        self.action_record = data['actions']
        self.state_record = data['states']

        self.status = 0
        ob, reward, _, _ = self.step(np.zeros(5))
        self.status = reward
        self.observation_space = convert_observation_to_space(ob)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.state = self.state + np.asarray(action)
        env = CartPoleEnv(self.state[0],self.state[1],
                          self.state[2],self.state[3],self.state[4])
 
        episode_count = len(self.action_record)
        model_diff = 0
        for i in range(episode_count):
            ob = env.reset()
            traj_state = []
            for j in range(len(self.action_record[i])):
            # The traj that done is better tricky here
                action = self.action_record[i][j]
                ob, reward, done, _ = env.step(action)
                traj_state.append(ob)
                if done:
                    break
            if not done:
                model_diff = model_diff + 1 # penalty for not done
            model_diff = model_diff + self._traj_diff(np.asarray(traj_state), self.state_record[i])
        reward = - model_diff - self.status
        self.status = - model_diff
        done = False
        return np.array(self.state), reward, done, {}

    def reset(self, gravity = 9.8, masscart = 1.0, masspole = 0.1, 
                    length = 0.5, force_mag = 10.0):
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.length = length
        self.force_mag = force_mag

        self.state = np.asarray([gravity, masscart, masspole, length, force_mag])
        self.status = 0
        ob, reward, _, _ = self.step(np.zeros(5))
        self.status = reward
        return np.array(self.state)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _traj_diff(self, traj_state1, traj_state2):
        total_distance = 0
        for i in range(min(len(traj_state1), len(traj_state2))):
            total_distance = total_distance + LA.norm(traj_state1[i] - traj_state2[i])
        for i in range(min(len(traj_state1), len(traj_state2)), 
                        max(len(traj_state1), len(traj_state2))):
            if len(traj_state1) <= len(traj_state2):
                total_distance = total_distance + LA.norm(traj_state2[i])
            else:
                total_distance = total_distance + LA.norm(traj_state1[i])
        return total_distance


