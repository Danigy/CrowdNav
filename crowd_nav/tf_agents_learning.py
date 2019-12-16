#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import os.path
import sys
import datetime

from absl import app
from absl import flags
from absl import logging

import numpy as np

import gin
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from collections import OrderedDict
import argparse
import configparser
import json

import gym
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
            safe_obstacle_distance = None
            safety_penalty_factor = None
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
            freespace_reward = params['freespace']
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
            lookahead_interval = None
            safety_penalty_factor = None
            safe_obstacle_distance = None
            time_to_collision_penalty = None
            personal_space_penalty = None
            freespace_reward = None      
            slack_reward = None
            energy_cost = None
            params['nn_layers'] = nn_layers = [256, 128, 64]
            gamma = 0.99
            decay = 0
            batch_norm = 'no'
            params['learning_trials'] = learning_trials = 1500000
            params['learning_rate'] = learning_rate = 0.0001
            params['test'] = 'tf_agents_in_crowdsim'

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
        params['lookahead_interval'] = env_config.getfloat('reward', 'lookahead_interval')

        
        if args.n_sonar_sensors is not None:
            self.n_sonar_sensors = args.n_sonar_sensors
        else:
            self.n_sonar_sensors = env_config.getint('robot', 'n_sonar_sensors')
        
        params['n_sonar_sensors'] = self.n_sonar_sensors
                
        env = gym.make('CrowdSim-v0', human_num=self.human_num, n_sonar_sensors=self.n_sonar_sensors, success_reward=success_reward, collision_penalty=collision_penalty, time_to_collision_penalty=time_to_collision_penalty,
                       discomfort_dist=discomfort_dist, discomfort_penalty_factor=discomfort_penalty_factor, lookahead_interval=lookahead_interval, potential_reward_weight=potential_reward_weight,
                       slack_reward=slack_reward, energy_cost=energy_cost, safe_obstacle_distance=safe_obstacle_distance, safety_penalty_factor=safety_penalty_factor, freespace_reward=freespace_reward,
                       visualize=visualize, show_sensors=show_sensors, testing=args.test, create_obstacles=args.create_obstacles, create_walls=args.create_walls, display_fps=args.display_fps)
        
        print("Gym environment created.")
                
        env.set_robot(robot)
        env.configure(env_config)
        
        env.seed()
        np.random.seed()
     
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
        
        self.train_eval(env=env)
    
    def string_to_filename(self, input):
        output = input.replace('"', '').replace('{', '').replace('}', '').replace(' ', '_').replace(',', '_')
        return output
    
    def normal_projection_net(self, action_spec,
                              init_action_stddev=0.35,
                              init_means_output_factor=0.1):
        del init_action_stddev
        return normal_projection_network.NormalProjectionNetwork(
          action_spec,
          mean_transform=None,
          state_dependent_std=True,
          init_means_output_factor=init_means_output_factor,
          std_transform=sac_agent.std_clip_transform,
          scale_distribution=True)
    
    def train_eval(self, env,
        root_dir='tensorboard_logs',
        env_name='CrowdSim-v0',
        eval_env_name=None,
        env_load_fn=lambda env_name: env,
        num_iterations=1500000,
        actor_fc_layers=(256, 256),
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(256, 256),
        # Params for collect
        initial_collect_steps=100,
        collect_steps_per_iteration=1,
        replay_buffer_capacity=150000,
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=64,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-4,
        alpha_learning_rate=1e-4,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        # Params for eval
        num_eval_episodes=30,
        eval_interval=1000,
        # Params for summaries and logging
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=5000,
        rb_checkpoint_interval=50000,
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    
        """A simple train and eval for SAC."""
        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'eval')
    
        train_summary_writer = tf.compat.v2.summary.create_file_writer(
          train_dir, flush_millis=summaries_flush_secs * 1000)
        train_summary_writer.set_as_default()
    
        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
          eval_dir, flush_millis=summaries_flush_secs * 1000)
        eval_metrics = [
          py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
          py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
        ]
        eval_summary_flush_op = eval_summary_writer.flush()
    
        global_step = tf.compat.v1.train.get_or_create_global_step()
        with tf.compat.v2.summary.record_if(
          lambda: tf.math.equal(global_step % summary_interval, 0)):
            # Create the environment.
            tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))
            eval_env_name = eval_env_name or env_name
            eval_py_env = env_load_fn(eval_env_name)
    
            # Get the data specs from the environment
            time_step_spec = tf_env.time_step_spec()
            observation_spec = time_step_spec.observation
            action_spec = tf_env.action_spec()
    
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                observation_spec,
                action_spec,
                fc_layer_params=actor_fc_layers,
                continuous_projection_net=normal_projection_net)
            critic_net = critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=critic_obs_fc_layers,
                action_fc_layer_params=critic_action_fc_layers,
                joint_fc_layer_params=critic_joint_fc_layers)
    
            tf_agent = sac_agent.SacAgent(
                time_step_spec,
                action_spec,
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=global_step)
    
        # Make the replay buffer.
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=1,
            max_length=replay_buffer_capacity)
        replay_observer = [replay_buffer.add_batch]
    
        eval_py_policy = py_tf_policy.PyTFPolicy(
            greedy_policy.GreedyPolicy(tf_agent.policy))
    
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_py_metric.TFPyMetric(py_metrics.AverageReturnMetric()),
            tf_py_metric.TFPyMetric(py_metrics.AverageEpisodeLengthMetric()),
        ]
    
        collect_policy = tf_agent.collect_policy
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())
    
        initial_collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=initial_collect_steps).run()
    
        collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration).run()
    
        # Prepare replay buffer as dataset with invalid transitions filtered.
        def _filter_invalid_transition(trajectories, unused_arg1):
          return ~trajectories.is_boundary()[0]
        dataset = replay_buffer.as_dataset(
            sample_batch_size=5 * batch_size,
            num_steps=2).unbatch().filter(
                _filter_invalid_transition).batch(batch_size).prefetch(
                    batch_size * 5)
        dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        trajectories, unused_info = dataset_iterator.get_next()
        train_op = tf_agent.train(trajectories)
    
        summary_ops = []
        for train_metric in train_metrics:
          summary_ops.append(train_metric.tf_summaries(
              train_step=global_step, step_metrics=train_metrics[:2]))
    
        with eval_summary_writer.as_default(), \
             tf.compat.v2.summary.record_if(True):
          for eval_metric in eval_metrics:
            eval_metric.tf_summaries(train_step=global_step)
    
        train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=tf_agent.policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)
    
        with tf.compat.v1.Session() as sess:
          # Initialize graph.
          train_checkpointer.initialize_or_restore(sess)
          rb_checkpointer.initialize_or_restore(sess)
    
          # Initialize training.
          sess.run(dataset_iterator.initializer)
          common.initialize_uninitialized_variables(sess)
          sess.run(train_summary_writer.init())
          sess.run(eval_summary_writer.init())
    
          global_step_val = sess.run(global_step)
    
          if global_step_val == 0:
            # Initial eval of randomly initialized policy
            metric_utils.compute_summaries(
                eval_metrics,
                eval_py_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=global_step_val,
                callback=eval_metrics_callback,
                log=True,
            )
            sess.run(eval_summary_flush_op)
    
            # Run initial collect.
            logging.info('Global step %d: Running initial collect op.',
                         global_step_val)
            sess.run(initial_collect_op)
    
            # Checkpoint the initial replay buffer contents.
            rb_checkpointer.save(global_step=global_step_val)
    
            logging.info('Finished initial collect.')
          else:
            logging.info('Global step %d: Skipping initial collect op.',
                         global_step_val)
    
          collect_call = sess.make_callable(collect_op)
          train_step_call = sess.make_callable([train_op, summary_ops])
          global_step_call = sess.make_callable(global_step)
    
          timed_at_step = global_step_call()
          time_acc = 0
          steps_per_second_ph = tf.compat.v1.placeholder(
              tf.float32, shape=(), name='steps_per_sec_ph')
          steps_per_second_summary = tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_second_ph,
              step=global_step)
    
          for _ in range(num_iterations):
            start_time = time.time()
            collect_call()
            for _ in range(train_steps_per_iteration):
              total_loss, _ = train_step_call()
            time_acc += time.time() - start_time
            global_step_val = global_step_call()
            if global_step_val % log_interval == 0:
              logging.info('step = %d, loss = %f', global_step_val, total_loss.loss)
              steps_per_sec = (global_step_val - timed_at_step) / time_acc
              logging.info('%.3f steps/sec', steps_per_sec)
              sess.run(
                  steps_per_second_summary,
                  feed_dict={steps_per_second_ph: steps_per_sec})
              timed_at_step = global_step_val
              time_acc = 0
    
            if global_step_val % eval_interval == 0:
              metric_utils.compute_summaries(
                  eval_metrics,
                  eval_py_env,
                  eval_py_policy,
                  num_episodes=num_eval_episodes,
                  global_step=global_step_val,
                  callback=eval_metrics_callback,
                  log=True,
              )
              sess.run(eval_summary_flush_op)
    
            if global_step_val % train_checkpoint_interval == 0:
              train_checkpointer.save(global_step=global_step_val)
    
            if global_step_val % policy_checkpoint_interval == 0:
              policy_checkpointer.save(global_step=global_step_val)
    
            if global_step_val % rb_checkpoint_interval == 0:
              rb_checkpointer.save(global_step=global_step_val)
              
if __name__ == '__main__':
    tf.compat.v1.enable_resource_variables()
    logging.set_verbosity(logging.INFO)
    #gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  
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

    if NN_TUNING:
        param_list = []
    
        nn_architectures = [[64, 64], [512, 256, 128], [256, 128, 64]]
        #nn_architectures = [[64, 64, 64], [1024, 512, 256], [512, 256, 128, 64]]
        gammas = [0.99, 0.95]
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
     