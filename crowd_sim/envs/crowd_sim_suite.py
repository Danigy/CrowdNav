from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers

def load(argv, params):
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