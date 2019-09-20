#!/usr/bin/env python3

import numpy as np
import random
import os.path
import sys
import datetime
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import DQN

from stable_baselines.gail import ExpertDataset

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

class SimpleNavigation():
    def __init__(self, argv, params):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--test', default=False, action='store_true')
        parser.add_argument('-w', '--weights', type=pathstr, required=False, help='Path to weights file')
        parser.add_argument('-d', '--visualize', default=False, action='store_true')
        parser.add_argument('-s', '--show_sensors', default=False, action='store_true')
        parser.add_argument('--env_config', type=str, default='configs/env.config')
        parser.add_argument('--policy', type=str, default='multi_human_rl')
        parser.add_argument('--policy_config', type=str, default='configs/policy.config')
        parser.add_argument('--train_config', type=str, default='configs/train.config')
        parser.add_argument('-p', '--pre_train', default=False, action='store_true')
        
        args = parser.parse_args()
        #args = vars(parsed_args)
        
        print(args)
        
        if NN_TUNING:
            gamma = 0.9
            params['decay'] = decay = 0
            params['batch_norm'] = 'no'
            success_reward = None
            potential_reward_weight = None
            collision_penalty = None
            time_to_collision_penalty = None
            personal_space_penalty = None          
            slack_reward = None
            energy_cost = None
            slack_reward = None
            learning_rate = 0.001
        elif TUNING:
            success_reward = None
            potential_reward_weight = None
            collision_penalty = None
            discomfort_dist = None            
            discomfort_penalty_factor = params['discomfort']
            safety_penalty_factor = params['safety']
            safe_obstacle_distance = None
            time_to_collision_penalty = None
            personal_space_penalty = None          
            slack_reward = None
            energy_cost = None

            params['learning_trials'] = learning_trials = 1500000
            params['learning_rate'] = learning_rate = 0.0003
            
            #personal_space_cost = 0.0
            #slack_reward = -0.01
            #learning_rate = 0.001
            if not NN_TUNING:
                nn_layers = [256, 128, 64]
                gamma = 0.99
                decay = 0
                batch_norm = 'no'
        else:
            params = dict()
            success_reward = None
            potential_reward_weight = None
            collision_penalty = None
            discomfort_dist = None            
            discomfort_penalty_factor = None
            safety_penalty_factor = None
            safe_obstacle_distance = None
            time_to_collision_penalty = None
            personal_space_penalty = None          
            slack_reward = None
            energy_cost = None
            params['nn_layers'] = nn_layers = [256, 128, 64]
            gamma = 0.99
            decay = 0
            batch_norm = 'no'
            params['learning_trials'] = learning_trials = 1500000
            params['learning_rate'] = learning_rate = 0.0003
            params['arch'] = 'per'

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
        
        visualize = True if args.visualize else None
        show_sensors = True if args.show_sensors else None
        
        env = gym.make('CrowdSim-v0', success_reward=success_reward, collision_penalty=collision_penalty, time_to_collision_penalty=time_to_collision_penalty,
                       discomfort_dist=discomfort_dist, discomfort_penalty_factor=discomfort_penalty_factor, potential_reward_weight=potential_reward_weight,
                       slack_reward=slack_reward, energy_cost=energy_cost, safe_obstacle_distance=safe_obstacle_distance, safety_penalty_factor=safety_penalty_factor,
                       visualize=visualize, show_sensors=show_sensors, testing=args.test, create_walls=False, create_obstacles=True)
        
        print("Gym environment created.")
        
        env.seed()
        np.random.seed()
        
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        
        env.set_robot(robot)
        env.configure(env_config)

        self.human_num = env_config.getint('sim', 'human_num')
        params['n_sonar_sensors'] = env_config.getint('robot', 'n_sonar_sensors')
     
        env = DummyVecEnv([lambda: env])

        if TUNING or NN_TUNING:
            if args.pre_train:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_pretrain_n_peds_' + str(self.human_num) + "_" + self.string_to_filename(json.dumps(params))
            else:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_n_peds_' + str(self.human_num) + "_" + self.string_to_filename(json.dumps(params))

            save_weights_file = tb_log_dir + '/sac_' + ENV_NAME + '_weights_' + self.string_to_filename(json.dumps(params)) +'.h5f'

        else:
            if args.pre_train:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_pretrain_n_peds_' + str(self.human_num) + "_" + self.string_to_filename(json.dumps(params))
            else:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_n_peds_' + str(self.human_num) + "_" + self.string_to_filename(json.dumps(params))

            save_weights_file = tb_log_dir + '/sac' + ENV_NAME + '_weights_final' + '.h5f'

        weights_path = os.path.join(tb_log_dir, "model_weights.{epoch:02d}.h5")
 
        model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=tb_log_dir, learning_rate=learning_rate, buffer_size=100000)
        
        if args.pre_train:
            pretrain_log_dir = os.path.expanduser('~') + '/tensorboard_logs/orca_' + str(self.human_num) + "_" + self.string_to_filename(json.dumps(params))            
            pretrained_weights_file = pretrain_log_dir + '/orca_weights_final.npz'
    
            dataset = ExpertDataset(expert_path=pretrained_weights_file, traj_limitation=1000, batch_size=128)
            #model.pretrain(dataset, n_epochs=1000, learning_rate=1e-3, adam_epsilon=1e-8, val_interval=10)
            model.pretrain(dataset, n_epochs=200)
            
            obs = env.reset()
            n_episodes = 0
            print("Testing pre-trained model...")
            n_test_episodes = 100
            while n_episodes < n_test_episodes:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                if done:
                    n_episodes += 1
                    if n_episodes % 10 == 0:
                    #del info['terminal_observation']
                        print([(key, trunc(info[0][key], 1)) for key in ['success_rate', 'collision_rate', 'timeouts', 'personal_space_violations']])
                    obs = env.reset()
                    
            #env.close()
            #os._exit(0)

        if args.test:
            print("Testing!")
            model = DQN.load(args.weights)
            obs = env.reset()
            n_episodes = 0
            n_test_episodes = 10000
            while n_episodes < n_test_episodes:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                if done:
                    n_episodes += 1
                    #del info['terminal_observation']
                    if n_episodes % 2 == 0:
                        print("episodes:", n_episodes, [(key, trunc(info[0][key], 1)) for key in ['success_rate', 'ped_collision_rate', 'collision_rate', 'timeout_rate', 'personal_space_violations']])
                    obs = env.reset()

            env.close()
            os._exit(0)

        model.learn(total_timesteps=learning_trials, log_interval=10)
        model.save(tb_log_dir + "/stable_baselines")
        print(">>>>> End testing <<<<<", self.string_to_filename(json.dumps(params)))
        print("Final weights saved at: ", tb_log_dir + "/stable_baselines.pkl")

        print("\nTEST COMMAND:\n\npython3 py3_learning.py --test --weights ", tb_log_dir + "/stable_baselines.pkl --visualize")
        
        print("\nTESTING for 100 episodes with params:", params, "\n")

        obs = env.reset()
        n_episodes = 0
        n_test_episodes = 100
        while n_episodes < n_test_episodes:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                n_episodes += 1
                if n_episodes % 100 == 0:
                    print("episodes:", n_episodes, [(key, trunc(info[0][key], 1)) for key in ['success_rate', 'ped_collision_rate', 'collision_rate', 'timeout_rate', 'personal_space_violations']])
                    obs = env.reset()
        env.close()
    
    def create_base_lidar_model(self, input_size, output_size, nn_layers):
        base_model = OrderedDict()
        base_model['input']        = Input(shape=(1, input_size))
        
        # First layer.
        base_model['dense_1']      = Dense(nn_layers[0], activation='relu')(base_model['input'])
        base_model['batch_norm_1'] = BatchNormalization()(base_model['dense_1'])
        #base_model['dropout_1']    = Dropout(rate=0.2)(base_model['batch_norm_1'])

        # Second layer.
        base_model['dense_2']      = Dense(nn_layers[1], kernel_initializer='lecun_uniform', activation='relu')(base_model['batch_norm_1'])
        base_model['batch_norm_2'] = BatchNormalization()(base_model['dense_2'])
        #base_model['dropout_2']    = Dropout(rate=0.2)(base_model['batch_norm_2'])
        
        base_model['dense_3']      = Dense(nn_layers[2], activation='relu')(base_model['batch_norm_2'])
        
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
    print("Starting training with params:", params)
    SimpleNavigation(sys.argv, params)

if __name__ == '__main__':
    def pathstr(v): return os.path.abspath(v)
    
    def trunc(f, n):
        # Truncates/pads a float f to n decimal places without rounding
        slen = len('%.*f' % (n, f))
        return float(str(f)[:slen])
    
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, layers=[256, 128, 64], layer_norm=False, feature_extraction="mlp", **kwargs)

    if NN_TUNING:
        param_list = []
    
        nn_architectures = [[64, 64], [512, 256, 128], [256, 128, 64]]
        #nn_architectures = [[64, 64, 64], [1024, 512, 256], [512, 256, 128, 64]]
        gammas = [0.99, 0.95]
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
        
        discomfort_penalty_factors = [0.05, 0.1, 0.2, 0.5]
        safety_penalty_factors = [0.01, 0.05, 0.1, 0.5]

        for discomfort_penalty_factor in discomfort_penalty_factors:
            for safety_penalty_factor in safety_penalty_factors:
                params = {
                          "discomfort": discomfort_penalty_factor,
                          "safety": safety_penalty_factor
                          }
                param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)
    else:        
        SimpleNavigation(sys.argv, dict())
 
