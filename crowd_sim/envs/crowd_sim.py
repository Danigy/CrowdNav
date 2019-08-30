#!/usr/bin/env python3

import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs/utils'))

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.action import ActionXY, ActionRot

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions
from pymunk.space_debug_draw_options import SpaceDebugDrawOptions, SpaceDebugColor

import random
import math

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.time_to_collision_penalty = None        
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        
        self.n_sensors = 0
        self.sensor_range = 25 # meters
        self.n_episodes = 0
                
        ''' 'OpenAI Gym Requirements '''
        self._seed(123)
        
    def create_robot(self, x, y, r, robot_radius):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.robot_body = pymunk.Body(1, inertia)
        self.robot_body.position = x, y
        self.robot_shape = pymunk.Circle(self.robot_body, robot_radius)
        self.robot_shape.color = THECOLORS["white"]
        self.robot_shape.elasticity = 1.0
        self.robot_shape.collision_type = 3
        collison_handler = self.space.add_wildcard_collision_handler(3)
        collison_handler.begin = self.collision_begin
        collison_handler.separate = self.collision_separate
        self.robot_body.angle = 0
        driving_direction = Vec2d(1, 0).rotated(self.robot_body.angle)
        self.robot_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.robot_body, self.robot_shape)
        
    def create_pedestrian(self, x, y, r):
        ped_body = pymunk.Body(1000, 1000)
        ped_shape = pymunk.Circle(ped_body, r)
        ped_shape.elasticity = 1.0
        ped_body.position = x, y
        ped_body.velocity = random.randint(-self.max_ped_velocity, self.max_ped_velocity) * Vec2d(0, 1)
        ped_shape.color = THECOLORS["orange"]
        self.space.add(ped_body, ped_shape)
        return [ped_body, ped_shape]
    
    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.time_to_collision_penalty = config.getfloat('reward', 'time_to_collision_penalty')        
        self.potential_reward_weight = config.getfloat('reward', 'potential_reward_weight')        
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.draw_screen = config.getboolean('env', 'draw_screen')
        self.show_sensors = config.getboolean('env', 'show_sensors')
        self.display_fps = config.getfloat('env', 'display_fps')

        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.hallway_width = config.getfloat('sim', 'hallway_width')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        
        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))
        
       # ===== Begin Pymunk setup ===== #
        self.scale_factor = 100
        self.angle_offset = np.pi / 2.0
        
        self.width = int(1.2 * self.square_width * self.scale_factor)
        self.height = int(0.9 * self.square_width * self.scale_factor)
        
        #self.width = 1000
        #self.height = 1000
        
        if not self.draw_screen:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.screen_rect = self.screen.get_rect()
            self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.surface2 = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.rect_surface2 = self.surface2.get_rect(center=(0,0))
            self.draw_options = DrawOptions(self.screen)
            self.draw_options.flags = 3

        self.clock = pygame.time.Clock()
        
        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        
        self.max_ped_velocity = 50
        
        # List to hold the pedestrians
        self.pedestrians = []

        if self.draw_screen:
            # Create the Pygame robot for display
            self.create_robot(self.width/2, self.height/2, 0, self.scale_factor * self.robot.radius)
        
        self.action_space = spaces.Box(-1.0, 1.0, shape=[2,])
        self.observation_space = spaces.Box(-1.0, 1.0, shape=[self.n_sensors + 2 + 4 * self.human_num,])
        
    def set_robot(self, robot):
        self.robot = robot

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side
        Rule hallway_crossing: robot must cross path of pedestrians moving at right angles
        
        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'hallway_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_hallway_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, 0, -10, 0, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, 0, px, py, 0, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

        if self.draw_screen:
            if len(self.pedestrians) > 0:
                for i in range(human_num):
                    self.space.remove(self.pedestrians[i][0], self.pedestrians[i][1])                    
                        
            self.pedestrians = []
        
            for i in range(human_num):
                self.pedestrians.append(self.create_pedestrian(self.scale_factor * self.humans[i].px + self.width/2, self.scale_factor * self.humans[i].py + self.height/2, self.scale_factor * self.humans[i].radius))

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, 0, -px, -py, 0, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width

            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, 0, gx, gy, 0, 0, 0, 0)
        return human
    
    def generate_hallway_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.hallway_width

            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.hallway_width
            #gx = px
            #gy = py
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, 0, gx, gy, 0, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times
    
    def _get_state(self):
        state = []

        if self.robot.kinematics == 'holonomic':
            gx = self.robot.gx - self.robot.px
            gy = self.robot.gy - self.robot.py
        else:
            gx = self.robot.gx - self.robot.px
            gy = self.robot.gy - self.robot.py
            
            cos_theta = np.cos(-self.robot.theta)
            sin_theta = np.sin(-self.robot.theta)
                
            gx_rotated = gx * cos_theta - gy * sin_theta
            gy_rotated = gx * sin_theta + gy * cos_theta
            
            gx = gx_rotated
            gy = gy_rotated

        state.append(gx / self.sensor_range)
        state.append(gy / self.sensor_range)
        
        for i, human in enumerate(self.humans):
            if self.robot.kinematics == 'holonomic':
                px = human.px - self.robot.px
                py = human.py - self.robot.py
                vx = human.vx - self.robot.vx
                vy = human.vy - self.robot.vy
            else:
                px = human.px - self.robot.px
                py = human.py - self.robot.py
                
                px_rotated = px * cos_theta - py * sin_theta
                py_rotated = px * sin_theta + py * cos_theta
                
                px = px_rotated
                py = py_rotated

                vx_rel = human.vx - self.robot.vx
                vy_rel = human.vy - self.robot.vy
                
                vx_rotated = vx_rel * cos_theta - vy_rel * sin_theta
                vy_rotated = vx_rel * sin_theta + vy_rel * cos_theta
                
                vx = vx_rotated
                vy = vy_rotated
                
            #print(self.robot.theta, np.sqrt(vx**2 + vy**2), px, py, gx, gy, vx, vy)
                
            state.append(px / self.sensor_range)
            state.append(py / self.sensor_range)
            state.append(vx / self.robot.max_linear_velocity)
            state.append(vy / self.robot.max_linear_velocity)
            
#             if self.draw_screen:
#                 rel_vel = Vec2d(vx_rel, vy_rel)                
#                 rel_pos = self.robot_body.position + 2 * self.scale_factor * rel_vel
#     
#                 pygame.draw.lines(self.surface, (0, 255, 0), False, [Vec2d(self.robot_body.position[0], self.screen_y(self.robot_body.position[1])), Vec2d(rel_pos[0], self.screen_y(rel_pos[1]))], 3)

        return np.array(state)
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, phase='test', test_case=None, debug=False):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}

            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
                
            px = np.random.random() * self.square_width * 0.2 * sign
            
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1           
            
            py = -self.square_width * 0.35
            
            gx = np.random.random() * self.square_width * 0.2 * sign
            gy = self.square_width * 0.35
            
            self.robot.set(px, py, 0, gx, gy, 0, 0, 0, np.pi/2)

            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        state = self._get_state()
        
        if debug:
            return ob
        else:
            return state

    def onestep_lookahead(self, action, debug=False, draw_screen=None, display_fps=None):
        return self.step(action, update=False, debug=debug, draw_screen=None, display_fps=None)

    def step(self, action, update=True, debug=False, draw_screen=None, display_fps=None):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        self.draw_screen = draw_screen if draw_screen is not None else self.draw_screen
        self.display_fps = display_fps if display_fps is not None else self.display_fps

        if self.robot.kinematics == 'holonomic':
            action = ActionXY(action[0], action[1])                        
        else:
            action = ActionRot((action[0] + 1.0) / 2.0, action[1])
            
        self.n_episodes += 1
        
        human_actions = []

        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
          
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        reward, done, info = self._get_reward(action)

        if update:
            #self.test_update_without_robot()
            
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
            
        # Update Pymunk
        self.space.step(self.time_step)

        # Update Pygame screen
        if self.draw_screen:
            pygame_gx = int(self.scale_factor * self.robot.gx + self.width / 2.0)
            pygame_gy = int(self.scale_factor * self.robot.gy + self.height / 2.0)                
            pygame.draw.circle(self.surface, (0, 255, 0, 200), (pygame_gx, self.screen_y(pygame_gy)), 30)                           
            pygame_px = int(self.scale_factor * self.robot.px + self.width/2)
            pygame_py = int(self.scale_factor * self.robot.py + self.height/2)
            
            self.robot_body.position = Vec2d(pygame_px, pygame_py)
            self.robot_body.angle = self.robot.theta
            
            pygame.draw.circle(self.surface, (255, 255, 255, 40), (pygame_px, self.screen_y(pygame_py)), int(self.scale_factor * self.robot.personal_space))
            
            index = 0
            
            for human in self.humans:
                human_state = human.get_full_state()
                
                pygame_px = int(self.scale_factor * human_state.px + self.width/2)
                pygame_py = int(self.scale_factor * human_state.py + self.height/2)
                
                self.pedestrians[index][0].position = Vec2d(pygame_px, pygame_py)
                
                pygame.draw.circle(self.surface, (255, 255, 255, 40), (pygame_px, self.screen_y(pygame_py)), int(self.scale_factor * human.personal_space))
                
#                 ellipse_left = pygame_px - int(self.scale_factor * human.personal_space/2)
#                 ellipse_top = self.screen_y(pygame_py + int(self.scale_factor * human.personal_space/2))
#                 
#                 ellipse_rect = pygame.Rect(ellipse_left, ellipse_top, int(1.2 * self.scale_factor * human.personal_space), int(self.scale_factor * human.personal_space))
# 
#                 pygame.draw.ellipse(self.surface2, (255, 255, 255, 40), ellipse_rect)
#                 
#                 rotated_surface = pygame.transform.rotate(self.surface2, 45)
#                 rotated_rect = rotated_surface.get_rect()
#                 rotated_rect.clamp_ip(self.screen_rect)
#                 self.screen.blit(rotated_surface, self.rect_surface2.center)
                
                pygame_gx = int(self.scale_factor * human_state.gx + self.width/2)
                pygame_gy = int(self.scale_factor * human_state.gy + self.height/2)
                pygame.draw.circle(self.surface, (255, 0, 0, 200), (pygame_gx, self.screen_y(pygame_gy)), 10)                

                index += 1
            
            self.space.debug_draw(self.draw_options)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.display_fps)
            self.screen.fill(THECOLORS["black"])
            self.surface.fill(THECOLORS["black"])
            #self.surface2.fill(THECOLORS["black"])
            
        # Convert Observable state to numpy state
        state = self._get_state()

        #print(state)

        if debug:
            return ob, reward, done, info
        else:
            return state, reward, done, info

    def test_update_without_robot(self):
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # store current positions and velocities of all non-robot agents
            self.human_positions = []
            self.human_velocities = []            

            for i in range(len(self.humans)):
                self.human_positions[i] = [self.humans[i].px, self.humans[i].py, self.humans[i].theta]
                self.human_velocities[i] = [self.humans[i].vx, self.humans[i].vy, self.humans[i].vr]                
            
            # update all non-robot agents
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)        

    def _get_reward(self, action):
        if self.robot.kinematics == 'holonomic':
            action = ActionXY(action[0], action[1])
        else:
            action = ActionRot(action[0], action[1])
            
        # collision detection
        dmin = float('inf')
        collision = False
        
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py

            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
                
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step

            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')
                    
        # velocity projection to detect "cutting off"
        velocity_dmin = float('inf')
        cutting_off = False
        
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py

            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            
            lookahead_time = 1.0
            
            ex = px + vx * lookahead_time
            ey = py + vy * lookahead_time

            # project motion ahead lookahead_time seconds
            velocity_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius

            if velocity_dist < velocity_dmin:
                velocity_dmin = velocity_dist

#         # check if the robot is cutting off a pedestrian
#         time_to_collision_danger = False
#         for i, human in enumerate(self.humans):
#             dx = self.humans[i].px - self.robot.px
#             dy = self.humans[i].py - self.robot.py
#             
#             dist = np.sqrt(dx**2 + dy**2) - self.humans[i].radius - self.robot.radius
#             
#             if self.robot.kinematics == 'holonomic':
#                 vx = human.vx - self.robot.vx
#                 vy = human.vy - self.robot.vy
#             else:
#                 raise NotImplementedError
# 
#             v = np.sqrt(vx**2 + vy**2)
# 
#             v_projection = (vx * px + vy * py) / dist # v dot p over ||p||
# 
#             time_to_collision = dist / (v_projection + 0.0001)
# 
#             if time_to_collision < 0:
#                 time_to_collision = float('inf')
# 
#             if time_to_collision < 2.0:
#                 time_to_collision_danger = True
#                 break

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < 2.0 * self.robot.radius

        done = False
        info = dict()
        
        reward = 0
        
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info['status'] = 'timeout'
        elif collision:
            reward = self.collision_penalty
            done = True
            info['status'] = 'collision'            
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info['status'] = 'success'
        elif velocity_dmin < self.discomfort_dist:
            # penalize agent for getting too close
            # adjust the reward based on FPS
            reward = (velocity_dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info['status'] = 'unsafe'
#         elif time_to_collision_danger:
#             reward = self.time_to_collision_penalty
#             done = False
#             info['status'] = 'time_to_collision'            
        else:
            done = False
            info['status'] = 'nothing'
            
        if not done:
            reward += -0.001 # slack reward

            
        # Get initial goal potential and collision potential
        if not done:
            if self.n_episodes == 1:
                self.initial_potential = self.get_potential()
                self.normalized_potential = 1.0
            
            # Get delta potential and compute reward
            current_potential = self.get_potential()
            new_normalized_potential = current_potential / self.initial_potential
            potential_reward = self.normalized_potential - new_normalized_potential
            reward += potential_reward * self.potential_reward_weight
            self.normalized_potential = new_normalized_potential
            info['status'] = 'nothing'

        return reward, done, info
    
    def get_potential(self):
        return np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])) 

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                global global_step
                global arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
        
    def distance_to_ellipse(self, semi_major, semi_minor, p):  
        px = abs(p[0])
        py = abs(p[1])
    
        tx = 0.707
        ty = 0.707
    
        a = semi_major
        b = semi_minor
    
        for x in range(0, 3):
            x = a * tx
            y = b * ty
    
            ex = (a*a - b*b) * tx**3 / a
            ey = (b*b - a*a) * ty**3 / b
    
            rx = x - ex
            ry = y - ey
    
            qx = px - ex
            qy = py - ey
    
            r = math.hypot(ry, rx)
            q = math.hypot(qy, qx)
    
            tx = min(1, max(0, (qx * r / q + ex) / a))
            ty = min(1, max(0, (qy * r / q + ey) / b))
            t = math.hypot(ty, tx)
            tx /= t 
            ty /= t 

        return (math.copysign(a * tx, p[0]), math.copysign(b * ty, p[1]))

    def collision_begin(self, space, arbiter, data):
        self.crashed = True
        return True
    
    def collision_separate(self, space, arbiter, data):
        self.crashed = False
        return False
    
    def screen_y(self, y):
        return self.height - y
    
    def rot_center(self, image, angle):
    
        center = image.get_rect().center
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center = center)
    
        return rotated_image, new_rect
        
if __name__ == '__main__':
    cs = CrowdSim()
    while True:
        cs.step(1.0, 0.1)
