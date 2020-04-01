import argparse
import sys
from cartpole import CartPoleEnv

import gym
from gym import wrappers, logger
import numpy as np
import pickle 

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = CartPoleEnv(gravity = 8, masscart = 1.3, masspole = 0.2, 
                        length = 0.7, force_mag = 8.0)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    action_record = []
    state_record = []
    for i in range(episode_count):
        ob = env.reset()
        #action = agent.act(ob, reward, done)
        traj_action = []
        traj_state = []
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            traj_action.append(action)
            traj_state.append(ob)
            print(done)
            #env.render()
            if done:
                break
        action_record.append(traj_action)
        state_record.append(traj_state)
    action_record = np.asarray(action_record)
    state_record = np.asarray(state_record)
    record = {'actions': action_record, 'states':state_record}

    record_file = open('record.pickle', 'wb')
    pickle.dump(record, record_file)
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()