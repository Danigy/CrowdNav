#!/usr/bin/env python3

import numpy as np
import random
import os.path
import sys
import datetime
import gym
import crowd_sim
from crowd_sim.envs.utils.info import *
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import SAC

from collections import OrderedDict
import argparse
import configparser
import json

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory

ENV_NAME = 'CrowdSim-v0'

N_SENSORS = 0
N_ACTIONS = 3
            
N_OBSTACLES = 0
N_PEDESTRIANS = 3
PERSONAL_SPACE_DISTANCE = 0.3
MAX_STEPS = 1000
HOLONOMIC = False
                
TUNING = False
NN_TUNING = False

class SimpleNavigation():
    def __init__(self, argv, params):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--test', default=False, action='store_true')
        parser.add_argument('-w', '--weights', type=pathstr, required=False, help='Path to weights file')
        parser.add_argument('-d', '--draw_screen', default=False, action='store_true')
        parser.add_argument('--env_config', type=str, default='configs/env.config')
        parser.add_argument('--policy', type=str, default='cadrl')
        parser.add_argument('--policy_config', type=str, default='configs/policy.config')
        parser.add_argument('--train_config', type=str, default='configs/train.config')
        
        args = parser.parse_args()
        #args = vars(parsed_args)
        
        print(args)
        
        if NN_TUNING:
            gamma = params['gamma']
            params['decay'] = decay = 0
            params['batch_norm'] = 'no'
            success_reward = 1
            potential_reward_weight = 10
            collision_penalty = -10
            potential_collision_reward_weight = 0
            personal_space_penalty = -100
            freespace_reward_weight = 0.0  
            slack_reward = -0.01
            learning_rate = 0.0005
        elif TUNING:
            success_reward = params['success']
            potential_reward_weight = params['potential']
            collision_penalty = params['collision']
            personal_space_penalty = params['personal']
            freespace_reward_weight = 0.0
            potential_collision_reward_weight = 0.0
            slack_reward = -0.01
            learning_rate = 0.001
            
            #personal_space_cost = 0.0
            #slack_reward = -0.01
            #learning_rate = 0.001
            if not NN_TUNING:
                nn_layers= [512, 256, 128]
                gamma = 0.99
                decay = 0
        else:
            params = dict()
            params['success'] = success_reward = 1
            params['potential'] = potential_reward_weight = 10
            params['collision'] = collision_penalty = -10
            params['personal'] = personal_space_penalty = -100            
            params['freespace'] = freespace_reward_weight = 0
            params['slack'] = slack_reward = -0.01
            potential_collision_reward_weight = 0
            params['learning_rate'] = learning_rate = 0.001

            params['nn_layers'] = nn_layers= [512, 256, 128]
            gamma = 0.99
            decay = 0
            batch_norm = 'no'

        # Create the Gym environment
        env = gym.make(ENV_NAME)
        
        # configure policy
        policy = policy_factory[args.policy]()
        if not policy.trainable:
            parser.error('Policy has to be trainable')
        if args.policy_config is None:
            parser.error('Policy config has to be specified for a trainable network')
        policy_config = configparser.RawConfigParser()
        policy_config.read(args.policy_config)
        policy.configure(policy_config)

        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(args.env_config)
        
        env = gym.make('CrowdSim-v0')
        print("Gym environment created.")
        
        env.seed(123)
        np.random.seed(123)
        
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        
        env.set_robot(robot)
        env.configure(env_config)

        env = DummyVecEnv([lambda: env])

        if TUNING or NN_TUNING:
            tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_' + self.string_to_filename(json.dumps(params))
            save_weights_file = tb_log_dir + '/dqn_' + ENV_NAME + '_weights_' + self.string_to_filename(json.dumps(args)) +'.h5f'

        else:
            tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_' + self.string_to_filename(json.dumps(params))
#            tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/dqn_npeds_' + str(N_PEDESTRIANS) + '_n_obs_' + str(N_OBSTACLES) + '_' + str(env.observation_space.shape) + '_' + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
            save_weights_file = tb_log_dir + '/sac' + ENV_NAME + '_weights_final' + '.h5f'

        weights_path = os.path.join(tb_log_dir, "model_weights.{epoch:02d}.h5")
 
        model = SAC(CustomPolicy, env, verbose=1, tensorboard_log=tb_log_dir, learning_rate=learning_rate,  buffer_size=50000)
        
        if args.test:
            print("Testing!")
            model = SAC.load(args.weights)
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
            os.exit(0)

        print("Holonomic?", HOLONOMIC)
        model.learn(total_timesteps=500000)
        model.save(tb_log_dir + "/stable_baselines")
        print(">>>>> End testing <<<<<", self.string_to_filename(json.dumps(params)))
        print("Final weights saved at: ", tb_log_dir + "/stable_baselines.pkl")

    
    def create_base_lidar_model(self, input_size, output_size, nn_layers):
        base_model = OrderedDict()
        base_model['input']        = Input(shape=(1, input_size))
        
        # First layer.
        base_model['dense_1']      = Dense(nn_layers[0], activation='relu')(base_model['input'])
        #base_model['batch_norm_1'] = BatchNormalization()(base_model['dense_1'])
        #base_model['dropout_1']    = Dropout(rate=0.2)(base_model['batch_norm_1'])

        # Second layer.
        base_model['dense_2']      = Dense(nn_layers[1], kernel_initializer='lecun_uniform', activation='relu')(base_model['dense_1'])
        #base_model['batch_norm_2'] = BatchNormalization()(base_model['dense_2'])
        #base_model['dropout_2']    = Dropout(rate=0.2)(base_model['batch_norm_2'])
        
        base_model['dense_3']      = Dense(nn_layers[2], activation='relu')(base_model['dense_2'])
        
        # Output
        output = Dense(output_size, activation='linear')(base_model['dense_3'])
        output = Reshape(target_shape=(output_size,))(output)

        model = Model(inputs=base_model['input'], outputs=output)

        return model

    def create_base_lidar_model1(self, input_size, output_size):
        base_model = OrderedDict()
        base_model['input']        = Input(shape=(1, input_size))
        
        # First layer.
        base_model['dense_1']      = Dense(164, activation='relu')(base_model['input'])
        base_model['batch_norm_1'] = BatchNormalization()(base_model['dense_1'])

        # Second layer.
        base_model['dense_2']      = Dense(150, kernel_initializer='lecun_uniform', activation='relu')(base_model['batch_norm_1'])
        base_model['batch_norm_2'] = BatchNormalization()(base_model['dense_2'])
        #base_model['dropout_2']    = Dropout(rate=0.2)(base_model['activation_2'])
        
        #base_model['dense_3']      = Dense(128, activation='relu')(base_model['batch_norm_2'])
        
        # Output
        output = Dense(output_size, activation='linear')(base_model['batch_norm_2'])
        output = Reshape(target_shape=(output_size,))(output)

        model = Model(inputs=base_model['input'], outputs=output)

        return model
    
    def string_to_filename(self, input):
        output = input.replace('"', '').replace('{', '').replace('}', '').replace(' ', '_').replace(',', '_')
        return output

    def tensorboard_callback(self, locals_, globals_):
        self_ = locals_['self']
        print(locals_, globals_)

        return True    
    
def launch_learn(params):
    print("Starting test with params:", params)
    SimpleNavigation(sys.argv, params)

if __name__ == '__main__':
    def pathstr(v): return os.path.abspath(v)
    
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, layers=[512, 256, 128], layer_norm=False, feature_extraction="mlp", **kwargs)

    if NN_TUNING:
        param_list = []
    
        #nn_architectures = [[256, 256], [128, 128], [64, 64]]
        nn_architectures = [[512, 256, 128], [1024, 512, 256], [512, 256, 128, 64]]
        gammas = [0.99, 0.95, 0.9]
        #decays = [0.0]
        for gamma in gammas:
            for nn_layers in nn_architectures:
                params = {
                          "nn_layers": nn_layers,
                          "gamma": gamma
                          }
                param_list.append(params)

        for param_set in param_list:
            # Custom MLP policy
            class CustomPolicy(FeedForwardPolicy):
                def __init__(self, *args, **kwargs):
                    super(CustomPolicy, self).__init__(*args, layers=params['nn_layers'], layer_norm=False, feature_extraction="mlp", **kwargs)
                       
            launch_learn(param_set)
        
    elif TUNING:
        param_list = []
        
        success_rewards = [1, 10, 100]
        #potential_reward_weights = [0.1, 1]
        potential_reward_weights = [10, 100]
        #collision_penalties = [-0.1, -1]
        collision_penalties = [-10, -100]
        #personal_space_penalties = [-0.1, -1]
        personal_space_penalties = [-10, -100]

        for success_reward in success_rewards:
            for potential_reward_weight in potential_reward_weights:
                for collision_penalty in collision_penalties:
                    for personal_space_penalty in personal_space_penalties:
                        params = {
                                  "success": success_reward,
                                  "potential": potential_reward_weight,
                                  "collision": collision_penalty,
                                  "personal": personal_space_penalty
                                  }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)
    else:        
        SimpleNavigation(sys.argv, dict())


