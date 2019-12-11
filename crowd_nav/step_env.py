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
from crowd_sim.envs.utils.action import ActionXY, ActionRot

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', type=str2bool, default=True, required=False)
    parser.add_argument('--show_sensors', default=True, action='store_true')
    parser.add_argument('-o', '--create_obstacles',type=str2bool, default=False, required=False)
    parser.add_argument('-w', '--create_walls',type=str2bool, default=False, required=False)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--hallway', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--fps', type=float, default=1000)

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
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    
    env = gym.make('CrowdSim-v0', success_reward=None, collision_penalty=None, discomfort_dist=None,
                       discomfort_penalty_factor=None, potential_reward_weight=None, lookahead_interval=None, slack_reward=None, energy_cost=None,
                       visualize=args.visualize, show_sensors=args.show_sensors, create_walls=args.create_walls, create_obstacles=args.create_obstacles)

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
    
    n_episodes = 0

    state, ob, obstacles = env.reset(args.phase, args.test_case, debug=True)

    while n_episodes < 100:
        done = False
        
        while not done:
            action = robot.act(ob, create_obstacles=args.create_obstacles, obstacles=obstacles)
            state, ob, _, done, info = env.step(action, update=True, debug=True)
            time.sleep(1.0/args.fps)

        n_episodes += 1
        
        print("episodes:", n_episodes, [(key, trunc(info[key], 2)) for key in ['success_rate', 'ped_collision_rate', 'ped_hits_robot_rate', 'collision_rate', 'timeout_rate', 'personal_space_violations', 'shortest_path_length']])
        #print(info)
            
        obs = env.reset()

    #else:
    #    explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

    env.close()
    os._exit(0)
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
        
    main()
