#!/usr/bin/env python3

import numpy as np
import random
import os.path
import sys

from stable_baselines.common.vec_env import DummyVecEnv
from crowd_sim.envs.policy.record_expert import generate_expert_traj

from collections import OrderedDict
import argparse
import configparser
import json

import gym
import crowd_sim
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.policy_factory import policy_factory

ENV_NAME = 'CrowdSim-v0'

TUNING = False
NN_TUNING = False

class ExpertNavigation():
    def __init__(self, argv, params):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env_config', type=str, default='configs/env.config')
        parser.add_argument('--policy_config', type=str, default='configs/policy.config')
        parser.add_argument('--policy', type=str, default='orca')
        parser.add_argument('--model_dir', type=str, default=None)
        parser.add_argument('--il', default=False, action='store_true')
        parser.add_argument('--gpu', default=False, action='store_true')
        parser.add_argument('--visualize', default=True, action='store_true')
        parser.add_argument('--phase', type=str, default='test')
        parser.add_argument('--test_case', type=int, default=None)
        parser.add_argument('--square', default=False, action='store_true')
        parser.add_argument('--circle', default=False, action='store_true')
        parser.add_argument('--hallway', default=False, action='store_true')
        parser.add_argument('--video_file', type=str, default=None)
        parser.add_argument('--traj', default=False, action='store_true')
        parser.add_argument('-d', '--draw_screen', default=False, action='store_true')
        
        args = parser.parse_args()
        #args = vars(parsed_args)
        
        print(args)
        
        params = dict()
        success_reward = None
        potential_reward_weight = None
        collision_penalty = None
        time_to_collision_penalty = None
        personal_space_penalty = None          
        slack_reward = None
        energy_cost = None
        learning_rate = 0.001
        params['nn_layers'] = nn_layers= [64, 64]
        gamma = 0.9
        decay = 0
        batch_norm = 'no'

        # configure policy
        policy = policy_factory[args.policy]()
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        policy.configure(policy_config)

        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)

        draw_screen = True if args.draw_screen else None
        
        env = gym.make('CrowdSim-v0', success_reward=success_reward, collision_penalty=collision_penalty, time_to_collision_penalty=time_to_collision_penalty,
                       discomfort_dist=None, discomfort_penalty_factor=None, potential_reward_weight=potential_reward_weight, slack_reward=slack_reward,
                       energy_cost=slack_reward, draw_screen=draw_screen, expert_policy=True)
        
        print("Gym environment created.")
        
        env.seed(321)
        np.random.seed(321)
        
        self.robot = Robot(env_config, 'robot')
        self.robot.set_policy(policy)
        
        env.set_robot(self.robot)
        env.configure(env_config)

        self.human_num = env_config.getint('sim', 'human_num')

        tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/orca_' + str(self.human_num) + "_" + self.string_to_filename(json.dumps(params))
        
        try:
            os.mkdir(tb_log_dir)
        except:
            pass
        
        save_weights_file = tb_log_dir + '/orca_weights_final.npz'

        generate_expert_traj(self.orca_expert, save_weights_file, env, n_episodes=10000)

#         if args.test:
#             print("Testing!")
#             model = SAC.load(args.weights)
#             obs = env.reset()
#             while True:
#                 action, _states = model.predict(obs)
#                 obs, rewards, dones, info = env.step(action)
#             os.exit(0)
# 
#         model.learn(total_timesteps=1000000)
#         model.save(tb_log_dir + "/stable_baselines")
#         print(">>>>> End testing <<<<<", self.string_to_filename(json.dumps(params)))
#         print("Final weights saved at: ", tb_log_dir + "/stable_baselines.pkl")
#         
#         print("TEST COMMAND: python3 py3_learning.py --test --weights ", tb_log_dir + "/stable_baselines.pkl")
    
    def orca_expert(self, _obs, _obs_crowdnav):
        action = self.robot.orca_act(_obs_crowdnav)
        return action
    
    def string_to_filename(self, input):
        output = input.replace('"', '').replace('{', '').replace('}', '').replace(' ', '_').replace(',', '_')
        return output

if __name__ == '__main__':
    def pathstr(v): return os.path.abspath(v)
    
    ExpertNavigation(sys.argv, dict())


