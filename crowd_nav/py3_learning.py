#!/usr/bin/env python3

import numpy as np
import random
import os.path
import sys
import datetime
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy, FeedForwardPolicy, CnnPolicy
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpLstmPolicy, ActorCriticPolicy, register_policy, mlp_extractor

from stable_baselines import SAC, PPO2
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
        parser.add_argument('-o', '--create_obstacles', type=str2bool, default=False, required=False)
        parser.add_argument('--create_walls', type=str2bool, default=False, required=False)
        parser.add_argument('-n', '--n_sonar_sensors', type=int, required=False)
        parser.add_argument('-p', '--n_peds', type=int, required=False)
        parser.add_argument('--env_config', type=str, default='configs/env.config')
        parser.add_argument('--policy', type=str, default='multi_human_rl')
        parser.add_argument('--policy_config', type=str, default='configs/policy.config')
        parser.add_argument('--train_config', type=str, default='configs/train.config')
        parser.add_argument('--pre_train', default=False, action='store_true')
        parser.add_argument('--display_fps', type=int, required=False, default=1000)

        
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
            params['learning_rate'] = learning_rate = 0.0005
            
            #personal_space_cost = 0.0
            #slack_reward = -0.01
            #learning_rate = 0.001
            if not NN_TUNING:
                nn_layers = [256, 128, 64, 32]
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
            params['nn_layers'] = nn_layers = [256, 128, 64, 32]
            gamma = 0.99
            decay = 0
            batch_norm = 'no'
            params['learning_trials'] = learning_trials = 500000
            params['learning_rate'] = learning_rate = 0.0005
            params['arch'] = 'position_+_velocity_+_discomfort_+_ttc'

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
        
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        
        if args.n_peds is not None:
            env_config.set('sim', 'human_num', args.n_peds)

        self.human_num = env_config.getint('sim', 'human_num')
                
        params['n_peds'] = self.human_num
        
        if args.n_sonar_sensors is not None:
            self.n_sonar_sensors = args.n_sonar_sensors
        else:
            self.n_sonar_sensors = env_config.getint('robot', 'n_sonar_sensors')
        
        params['n_sonar_sensors'] = self.n_sonar_sensors
                
        env = gym.make('CrowdSim-v0', human_num=self.human_num, n_sonar_sensors=self.n_sonar_sensors, success_reward=success_reward, collision_penalty=collision_penalty, time_to_collision_penalty=time_to_collision_penalty,
                       discomfort_dist=discomfort_dist, discomfort_penalty_factor=discomfort_penalty_factor, potential_reward_weight=potential_reward_weight,
                       slack_reward=slack_reward, energy_cost=energy_cost, safe_obstacle_distance=safe_obstacle_distance, safety_penalty_factor=safety_penalty_factor,
                       visualize=visualize, show_sensors=show_sensors, testing=args.test, create_obstacles=args.create_obstacles, create_walls=args.create_walls, display_fps=args.display_fps)
        
        print("Gym environment created.")
                
        env.set_robot(robot)
        env.configure(env_config)
        
        env.seed()
        np.random.seed()
     
        env = DummyVecEnv([lambda: env])

        if TUNING or NN_TUNING:
            if args.pre_train:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_pretrain_' + self.string_to_filename(json.dumps(params))
            else:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_' + self.string_to_filename(json.dumps(params))

            save_weights_file = tb_log_dir + '/sac_' + ENV_NAME + '_weights_' + self.string_to_filename(json.dumps(params)) +'.h5f'

        else:
            if args.pre_train:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_pretrain_' + self.string_to_filename(json.dumps(params))
            else:
                tb_log_dir = os.path.expanduser('~') + '/tensorboard_logs/sac_' + self.string_to_filename(json.dumps(params))

            save_weights_file = tb_log_dir + '/sac' + ENV_NAME + '_weights_final' + '.h5f'

        weights_path = os.path.join(tb_log_dir, "model_weights.{epoch:02d}.h5")
 
        model = SAC(CustomPolicy, env, verbose=1, tensorboard_log=tb_log_dir, learning_rate=learning_rate, buffer_size=50000)
        #model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log=tb_log_dir, learning_rate=learning_rate, buffer_size=100000)

#         policy_kwargs = {
#             "mlp_extractor": self.custom_feature_extractor
#         }

        
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
            model = SAC.load(args.weights)
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
        print("Final weights saved at: ", tb_log_dir + "/stable_baselines.zip")

        print("\nTEST COMMAND:\n\npython3 py3_learning.py --test --weights ", tb_log_dir + "/stable_baselines.zip --visualize")
        
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
        
    def custom_feature_extractor(self, state, **kwargs):
            """
            Copied from stable_baselines policies.py.
            This is nature CNN head where last channel of the image contains
            direct features on the last channel.
    
            :param scaled_images: (TensorFlow Tensor) Image input placeholder
            :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
            :return: (TensorFlow Tensor) The CNN output layer
            """
            activ = tf.nn.relu
            
            num_direct_features = 0
    
            # Take last channel as direct features
            other_features = tf.contrib.slim.flatten(state[..., -1])
            # Take known amount of direct features, rest are padding zeros
            other_features = other_features[:, :num_direct_features]
    
            state = state[..., :-1]
    
#             layer_1 = activ(conv(scaled_images, 'cnn1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
#             layer_2 = activ(conv(layer_1, 'cnn2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
#             layer_3 = activ(conv(layer_2, 'cnn3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#             layer_3 = conv_to_fc(layer_3)

            for i, layer_size in enumerate(layers):
                output = tf.layers.dense(output, layer_size, name='fc' + str(i))
                if layer_norm:
                    output = tf.contrib.layers.layer_norm(output, center=True, scale=True)
                    
            output = activ_fn(output)
    
            #sensor_output = activ(linear(layer_3, 'cnn_fc1', n_hidden=512, init_scale=np.sqrt(2)))
    
            concat = tf.concat((sensor_output, other_features), axis=1)
    
            return other_features

    
def launch_learn(params):
    print("Starting training with params:", params)
    SimpleNavigation(sys.argv, params)

if __name__ == '__main__':
    def pathstr(v): return os.path.abspath(v)
    
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    def trunc(f, n):
        # Truncates/pads a float f to n decimal places without rounding
        slen = len('%.*f' % (n, f))
        return float(str(f)[:slen])
    
    class CustomPolicy2(ActorCriticPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **kwargs):
            super(CustomPolicy2, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)
    
            with tf.variable_scope("model", reuse=reuse):
                activ = tf.nn.tanh
    
                extracted_features = tf.layers.flatten(self.processed_obs)
    
                pi_h = extracted_features
                for i, layer_size in enumerate([64, 64]):
                    pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
                pi_latent = pi_h
    
                vf_h = extracted_features
                for i, layer_size in enumerate([64, 64]):
                    vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
                value_fn = tf.layers.dense(vf_h, 1, name='vf')
                vf_latent = vf_h
    
                self.proba_distribution, self.policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
    
            self.value_fn = value_fn
            self.initial_state = None        
            self._setup_init()
    
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, layers=[256, 128, 64, 32], layer_norm=False, feature_extraction="mlp", **kwargs)

    if NN_TUNING:
        param_list = []
    
        nn_architectures = [[64, 64], [512, 256, 128, 64], [256, 128, 64, 32]]
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
 
