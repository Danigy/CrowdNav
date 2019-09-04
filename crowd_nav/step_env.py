import logging
import argparse
import configparser
import os
import torch
import time
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
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

    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # Create the Gym environment
    env = gym.make('CrowdSim-v0')
    
    # configure policy
    policy = policy_factory[args.policy]()
#     if not policy.trainable:
#         parser.error('Policy has to be trainable')
#     if args.policy_config is None:
#         parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    
    env = gym.make('CrowdSim-v0', success_reward=None, collision_penalty=None, time_to_collision_penalty=None, discomfort_dist=None,
                       discomfort_penalty_factor=None, potential_reward_weight=None, slack_reward=None, energy_cost=None, draw_screen=args.visualize)
    print("Gym environment created.")
    
    env.seed(123)
    np.random.seed(123)
    
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    
    env.set_robot(robot)
    env.configure(env_config)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    if args.hallway:
        env.test_sim = 'hallway_crossing'
        
    explorer = Explorer(env, robot, device, gamma=0.9)
     
    policy.set_phase(args.phase)
    policy.set_device(device)

    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)
 
    policy.set_env(env)
    robot.print_info()
    
    if args.visualize:
        while True:
            ob = env.reset(args.phase, args.test_case, debug=True)
            done = False
            #last_pos = np.array(robot.get_position())
            while not done:
                action = robot.act(ob)
                ob, _, done, info = env.step(action, update=True, debug=True, display_fps=1000)
                #time.sleep(0.25)
                #current_pos = np.array(robot.get_position())
                #logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
                #last_pos = current_pos
            #if args.traj:
            #    env.render('traj', args.video_file)
            #else:
            #    env.render('video', args.video_file)
    
            #logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
            #if robot.visible and info == 'reach goal':
            #    human_times = env.get_human_times()
            #    logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
    
#     if args.visualize:
#         episodes = 0
#         ob = env.reset(args.phase, args.test_case, debug=True)
# 
#         while episodes < env.case_size[args.phase]:
#             print("Episode:", episodes)
#             done = False
#             #last_pos = np.array(robot.get_position())
#             #while not done:
#                 #action = robot.act(ob)
#             action = [1.0, 0.0]
#             ob, _, done, info = env.step(action, update=True, debug=True, display_fps=10)
#                 #current_pos = np.array(robot.get_position())
#                 #logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
#                 #last_pos = current_pos
#     #         if args.traj:
#     #             env.render('traj', args.video_file)
#     #         else:
#     #             env.render('video', args.video_file)
#     
#             #logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
#             #if robot.visible and info == 'reach goal':
#              #   human_times = env.get_human_times()
#             #    logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
#             episodes += 1
#     else:
#         explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
