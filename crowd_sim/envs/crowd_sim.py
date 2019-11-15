#!/usr/bin/env python3

import logging
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import logging

import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
import sys, os
import atexit

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs/utils'))

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import Timeout, Danger, ReachGoal, Collision, Nothing
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
import time

from collections import OrderedDict

from copy import deepcopy

atexit.register(pygame.quit)
atexit.register(pygame.display.quit)

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, human_num=None, n_sonar_sensors=None, success_reward=None, collision_penalty=None, time_to_collision_penalty=None, discomfort_dist=None,
                       discomfort_penalty_factor=None, potential_reward_weight=None, slack_reward=None,
                       energy_cost=None, safe_obstacle_distance=None, safety_penalty_factor=None, freespace_reward=None, visualize=None,
                       show_sensors=None, expert_policy=False, testing=False, create_walls=False, create_obstacles=False, display_fps=1000):
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
        self.success_reward = success_reward or None
        self.collision_penalty = collision_penalty or None
        self.time_to_collision_penalty = time_to_collision_penalty or None
        self.discomfort_dist = discomfort_dist or None
        self.discomfort_penalty_factor = discomfort_penalty_factor or None
        self.slack_reward = slack_reward or None
        self.energy_cost = energy_cost or None
        self.safe_obstacle_distance = safe_obstacle_distance or None
        self.safety_penalty_factor = safety_penalty_factor or None  
        self.freespace_reward = freespace_reward or None        

        self.lookahead_interval = 3.0
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
        self.human_num = human_num or None
        self.n_sonar_sensors = n_sonar_sensors or None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

        self.n_episodes = 0
        self.n_steps = 0
        self.n_successes = 0
        self.n_collisions = 0
        self.n_ped_collisions = 0
        self.n_ped_hits_robot = 0
        self.n_timeouts = 0
        self.n_personal_space_violations = 0
        self.n_cutting_off = 0
        self.crashed = False
        self.ped_collision = False
        self.sensor_readings = None
        self.sqrt_2 = np.sqrt(2.0)
        self.max_ttc = -1.0

        self.visualize = visualize
        self.show_sensors = show_sensors
        self.expert_policy = expert_policy
        self.testing = testing
        self.create_walls = create_walls
        self.create_obstacles = create_obstacles
        self.display_fps = display_fps

        self.static = list()
                
        ''' 'OpenAI Gym Requirements '''
        self._seed(123)
        
    def create_robot(self, x, y, r, robot_radius):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.robot_body = pymunk.Body(1, inertia)
        self.robot_body.position = x, y
        self.robot_body.angle = r
        self.robot_shape = pymunk.Circle(self.robot_body, robot_radius)
        self.robot_shape.color = THECOLORS["white"]
        self.robot_shape.elasticity = 1.0
        self.robot_shape.collision_type = 1
        self.robot_shape.group = 1
        robot_filter = pymunk.ShapeFilter(categories=1)
        self.robot_shape.filter = robot_filter
        collison_pedestrian = self.space.add_collision_handler(1, 2)
        collison_object = self.space.add_collision_handler(1, 3)
        collison_pedestrian.begin = self.collision_pedestrian_begin
        collison_pedestrian.separate = self.collision_pedestrian_separate
        collison_object.begin = self.collision_object_begin
        collison_object.separate = self.collision_object_separate
        driving_direction = Vec2d(1, 0).rotated(self.robot_body.angle)
        self.robot_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.robot_body, self.robot_shape)
        
    def create_pedestrian(self, x, y, r):
        ped_body = pymunk.Body(1000, 1000)
        ped_shape = pymunk.Circle(ped_body, r)
        ped_shape.elasticity = 1.0
        ped_shape.collision_type = 2
        ped_shape.group = 2
        ped_filter = pymunk.ShapeFilter(categories=2)
        ped_shape.filter = ped_filter
        ped_body.position = x, y
        ped_body.velocity = Vec2d(0, 0)
        ped_shape.color = THECOLORS["orange"]
        self.space.add(ped_body, ped_shape)
        return [ped_body, ped_shape]
    
    def crowd_obstacle_to_pygame(self, obstacle):
        #pygame_segments = tuple(pygame_segments * self.scale_factor + self.width
        vertices = np.array(obstacle)
        
        for i in range(len(obstacle)):
            vertices[i][0] = vertices[i][0] * self.scale_factor + self.width/2.0
            vertices[i][1] = vertices[i][1] * self.scale_factor + self.height/2.0

        pymunk_obstacle = pymunk.Poly(self.space.static_body, vertices)
        
        return pymunk_obstacle

#         for segment in pygame_segments:
#             self.static.append(pymunk.Segment(
#                     self.space.static_body,
#                     (segment[0][0] * self.scale_factor + self.width/2, segment[0][1] * self.scale_factor + self.height/2),
#                     (segment[1][0] * self.scale_factor + self.width/2, segment[1][1] * self.scale_factor + self.height/2), 5))

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        
        self.success_reward = self.success_reward or config.getfloat('reward', 'success_reward')
        self.collision_penalty = self.collision_penalty or config.getfloat('reward', 'collision_penalty')
        self.time_to_collision_penalty = config.getfloat('reward', 'time_to_collision_penalty')        
        self.potential_reward_weight = config.getfloat('reward', 'potential_reward_weight')        
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.safety_penalty_factor = config.getfloat('reward', 'safety_penalty_factor')
        self.safe_obstacle_distance = config.getfloat('reward', 'safe_obstacle_distance')
        self.freespace_reward = config.getfloat('reward', 'freespace_reward')
        self.slack_reward = config.getfloat('reward', 'slack_reward')
        self.energy_cost = config.getfloat('reward', 'energy_cost')
        self.position_noise = config.getfloat('humans', 'position_noise')
        self.velocity_noise = config.getfloat('humans', 'velocity_noise')        
                
        self.lookahead_interval = config.getfloat('reward', 'lookahead_interval')        
        self.visualize = self.visualize or config.getboolean('env', 'visualize')
        self.show_sensors = self.show_sensors or config.getboolean('env', 'show_sensors')
        self.display_fps = self.display_fps or config.getfloat('env', 'display_fps')
        self.n_sonar_sensors =self.n_sonar_sensors or config.getint('robot', 'n_sonar_sensors')

        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.hallway_width = config.getfloat('sim', 'hallway_width')
            self.human_num = self.human_num or config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        
        # ===== Crowdsim to Pygame conversion ===== #
        self.scale_factor = 100
        self.angle_offset = 0.0

        self.sensor_range = self.square_width # meters
        self.n_sensor_samples = 40
        self.sensor_gap = 30 # pixels
        self.sensor_spread = 10      # pixels
        self.max_pygame_sensor_range = self.sensor_range * self.scale_factor

        
        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        self.width = int(1.0 * self.square_width * self.scale_factor)
        self.height = int(1.0 * self.square_width * self.scale_factor)
        
        #self.width = 1000
        #self.height = 1000
        
        if not self.visualize:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.screen = pygame.display.set_mode((self.width, self.height))
        #self.screen_rect = self.screen.get_rect()

        self.screen.set_alpha(None)
        self.draw_options = DrawOptions(self.screen)
        self.draw_options.flags = 3
        if self.visualize:
            self.surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.surface2 = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        #self.rect_surface2 = self.surface2.get_rect(center=(0,0))

        self.clock = pygame.time.Clock()
        
        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
                
        # List to hold the pedestrians
        self.pedestrians = []
        
        self.obstacles = []
        self.perturbed_obstacles = []
        self.pymunk_obstacles = []
        
        # Create walls.
        wall_width_start = 0.01 * self.width
        wall_width_end = 0.99 * self.width
        wall_height_start = 0.01 * self.height
        wall_height_end = 0.99 * self.height
        
        if self.create_walls:
            self.static = [
                pymunk.Segment(
                    self.space.static_body,
                    (wall_width_start, wall_height_start), (wall_width_end, wall_height_start), 3),
                pymunk.Segment(
                    self.space.static_body,
                    (wall_width_start, wall_height_start), (wall_width_start, wall_height_end), 3),
                pymunk.Segment(
                    self.space.static_body,
                    (wall_width_start, wall_height_end), (wall_width_end, wall_height_end), 3),
                pymunk.Segment(
                    self.space.static_body,
                    (wall_width_end, wall_height_start), (wall_width_end, wall_height_end), 3)
                #             pymunk.Segment(
                #                 self.space.static_body,
                #                 (self.width/2, self.height/2), (self.width/2, 1), 2)
            ]
        
        # Create obstacles
        if self.create_obstacles:
            self.obstacles = [] 
            self.obstacles.append([(-0.5, 1.5), (0.5, 1.5), (0.5, 0.5), (-0.5, 0.5)])
            self.obstacles.append([(-3.0, 0.5), (-2.5, 0.5), (-2.5, -0.5), (-3.0, -0.5)])
            self.obstacles.append([(4.0, -0.5), (3.5, -0.5), (3.5, -1.5), (4.0, -1.5)])
            
            self.make_pymunk_obstacles(self.obstacles)
             
        self.action_space = spaces.Box(-1.0, 1.0, shape=[2,])
        #self.observation_space = spaces.Box(-1.0, 1.0, shape=[self.n_sonar_sensors + 2 + 2 + 5 * self.human_num,])
        self.observation_space = spaces.Box(-1.0, 1.0, shape=[self.n_sonar_sensors + 2 + 5 * self.human_num,])
        
    def make_pymunk_obstacles(self, obstacles=None):
        self.space.remove(self.static)
        self.pymunk_obstacles = []
        self.perturbed_obstacles = []
        
        for obstacle in self.obstacles:
            vertices = list()
            for vertex in obstacle:
                perturbed_vertex_x = vertex[0] * (1 + np.random.uniform(-0.1, 0.1))
                perturbed_vertex_y = vertex[1] * (1 + np.random.uniform(-0.1, 0.1))
                vertices.append((perturbed_vertex_x, perturbed_vertex_y))
            self.perturbed_obstacles.append(vertices)

        self.static = list()
        
        for obstacle in self.perturbed_obstacles:
            pymunk_obstacle = self.crowd_obstacle_to_pygame(obstacle)
            vertices = pymunk_obstacle.get_vertices()
            self.static.append(pymunk_obstacle)
            
        for s in self.static:
            s.friction = 1.
            obstacle_filter = pymunk.ShapeFilter(categories=3)
            s.filter = obstacle_filter
            s.collision_type = 3
            s.group = 3
            s.color = THECOLORS["red"]
            self.pymunk_obstacles.append(s)
            
        self.space.add(self.static)

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
#                         for agent in [self.robot] + self.humans:
#                             #if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
#                             if norm((px - agent.px, py - agent.py)) < human.radius:
#                                 collide = True
#                                 break
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
            px = self.square_width * 0.4 * sign
            py = (np.random.random() - 0.5) * self.hallway_width

#             px = (np.random.random() - 0.5) * self.square_width
#             py = (np.random.random() - 0.5) * self.square_width

            collide = False
#             for agent in [self.robot] + self.humans:
#                 if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
#                     collide = True
#                     break
            if not collide:
                break
        while True:
            gx = self.square_width * 0.4 * -sign
            gy = (np.random.random() - 0.5) * self.hallway_width

#            gx = (np.random.random() - 0.5) * self.square_width
#            gy = (np.random.random() - 0.5) * self.square_width

            collide = False
#             for agent in [self.robot] + self.humans:
#                 #if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
#                 if norm((gx - agent.gx, gy - agent.gy)) < human.radius:
#                     collide = True
#                     break
            if not collide:
                break
        human.set(px, py, 0, gx, gy, 0, 0, 0, 0)
        return human

#     def get_human_times(self):
#         """
#         Run the whole simulation to the end and compute the average time for human to reach goal.
#         Once an agent reaches the goal, it stops moving and becomes an obstacle
#         (doesn't need to take half responsibility to avoid collision).
# 
#         :return:
#         """
#         # centralized orca simulator for all humans
#         if not self.robot.reached_destination():
#             raise ValueError('Episode is not done yet')
#         params = (10, 10, 5, 5)
#         sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
#         sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
#                      self.robot.get_velocity())
#         for human in self.humans:
#             sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())
# 
#         max_time = 1000
#         while not all(self.human_times):
#             for i, agent in enumerate([self.robot] + self.humans):
#                 vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
#                 if norm(vel_pref) > 1:
#                     vel_pref /= norm(vel_pref)
#                 sim.setAgentPrefVelocity(i, tuple(vel_pref))
#             sim.doStep()
#             self.global_time += self.time_step
#             if self.global_time > max_time:
#                 logging.warning('Simulation cannot terminate!')
#             for i, human in enumerate(self.humans):
#                 if self.human_times[i] == 0 and human.reached_destination():
#                     self.human_times[i] = self.global_time
# 
#             # for visualization
#             self.robot.set_position(sim.getAgentPosition(0))
#             for i, human in enumerate(self.humans):
#                 human.set_position(sim.getAgentPosition(i + 1))
#             self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
# 
#         del sim
#         return self.human_times

    def _get_state(self):
        # Get the current location of the robot and the sensor readings there
        x, y = self.robot_body.position
                
        state = []
        self.sensor_readings = []

        if self.n_sonar_sensors > 0:
            state = list(self.get_sonar_readings(x, y, self.robot_body.angle))
            
            #print(state)

            #state = list(self.get_sonar_readings_xy(x, y, self.robot_body.angle))
            self.sensor_readings = deepcopy(state)
            
#             state_xy = list(self.get_sonar_readings_xy(x, y, self.robot_body.angle))
#             
#             for [px, py] in state_xy:
#                 if self.robot.vx * px + self.robot.vy * py == 0:
#                     time_to_collision = -1.0
#                     
#                 time_to_collision = (px**2 + py**2) / (self.robot.vx * px + self.robot.vy * py)
#                 
#                 if time_to_collision < 0:
#                     time_to_collision = -1.0
#                 else:
#                     time_to_collision = 1.0 - np.tanh(time_to_collision / 10.0)
#                     
#                 state.append(px)
#                 state.append(py)
#                 state.append(time_to_collision)
#                 self.sensor_readings.append(np.sqrt(px**2 + py**2))
        
#         if self.robot.kinematics == 'holonomic':
#             state.append(self.robot.vx / self.robot.max_linear_velocity / self.sqrt_2)
#             state.append(self.robot.vy / self.robot.max_linear_velocity / self.sqrt_2)
#         else:
#             self.robot.v = np.sqrt(self.robot.vx**2 + self.robot.vy**2)
#             state.append(self.robot.v / self.robot.max_linear_velocity)
#             state.append(self.robot.vr / self.robot.max_angular_velocity)
                   
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

        state.append(gx / self.square_width / self.sqrt_2)
        state.append(gy / self.square_width / self.sqrt_2)
        
        for i, human in enumerate(self.humans):
            px_future = (human.px + human.vx * self.lookahead_interval) - self.robot.px
            py_future = (human.py + human.vy * self.lookahead_interval) - self.robot.py
                
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
                
                px_future_rotated = px_future * cos_theta - py_future * sin_theta
                py_future_rotated = px_future * sin_theta + py_future * cos_theta
                
                px_future = px_future_rotated
                py_future = py_future_rotated

                vx_rel = human.vx - self.robot.vx
                vy_rel = human.vy - self.robot.vy
                
                vx_rotated = vx_rel * cos_theta - vy_rel * sin_theta
                vy_rotated = vx_rel * sin_theta + vy_rel * cos_theta
                
                vx = vx_rotated
                vy = vy_rotated

            # Add some noise
            px = (1.0 + random.uniform(-1, 1) * self.position_noise) * px
            py = (1.0 + random.uniform(-1, 1) * self.position_noise) * py
            vx = (1.0 + random.uniform(-1, 1) * self.velocity_noise) * vx
            vy = (1.0 + random.uniform(-1, 1) * self.velocity_noise) * vy
                
            if (vx * px + vy * py) == 0:
                time_to_collision = -1.0
            else:
                time_to_collision = -1.0 * (px**2 + py**2) / (vx * px + vy * py)
            
            if time_to_collision < 0:
                time_to_collision = -1.0
            else:
                time_to_collision = 1.0 - np.tanh(time_to_collision / 10.0)
#                 if time_to_collision > self.max_ttc:
#                     self.max_ttc = time_to_collision
#                     print(self.max_ttc)
                
            #print(self.robot.theta, np.sqrt(vx**2 + vy**2), px, py, gx, gy, vx, vy)
            
            #state.append(px_future / self.square_width)
            #state.append(py_future / self.square_width)
            #state.append(time_to_collision)
            #state.append(px / self.square_width)
            #state.append(py / self.square_width)
            #state.append((px + vx * self.lookahead_interval) / self.square_width)
            #state.append((py + vy * self.lookahead_interval) / self.square_width)
            #state.append(vx / self.robot.max_linear_velocity / self.sqrt_2)
            #state.append(vy / self.robot.max_linear_velocity / self.sqrt_2)

            state.append(px / self.square_width / self.sqrt_2)
            state.append(py / self.square_width / self.sqrt_2)
            state.append(vx / self.robot.max_linear_velocity / self.sqrt_2)
            state.append(vy / self.robot.max_linear_velocity / self.sqrt_2)
            state.append(time_to_collision)
            
        for i in range(len(state)):
            #if abs(state[i]) > 1:
            #    print(state[i])
            state[i] = max(-1.0, min(1.0, state[i]))
                
        assert all([abs(x) <= 1.0 for x in state])
            
        #print(state)
        
        return np.array(state)
    
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, phase='test', test_case=None, debug=False):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.n_steps = 0
        
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
            
            if self.create_obstacles:
                self.make_pymunk_obstacles(self.obstacles)

            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
                
            px = np.random.random() * self.square_width * 0.2 * sign
             
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
             
            py = self.square_width * 0.4 * sign# * np.random.uniform(0.8, 1.2)
            
            theta = np.random.random() * 2 * np.pi
             
            gx = np.random.random() * self.square_width * 0.2 * sign
            gy = self.square_width * 0.4 * -sign  #* np.random.uniform(0.8, 1.2)

#             px = np.random.random() * self.square_width * 0.2
#             py = np.random.random() * self.square_width * 0.2
# 
#             gx = np.random.random() * self.square_width * 0.2
#             gy = np.random.random() * self.square_width * 0.2
            
            self.robot.set(px, py, theta, gx, gy, 0, 0, 0, 0)

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
        
        try:
            self.robot_body.position
        except:
            pygame_px = int(self.scale_factor * self.robot.px + self.width/2)
            pygame_py = int(self.scale_factor * self.robot.py + self.height/2)
            self.create_robot(pygame_px, pygame_py, np.pi/2, self.scale_factor * self.robot.radius)
            
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
        
        # Update Pymunk
        self.space.step(self.time_step)

        state = self._get_state()
        
        if debug or self.expert_policy:
            return state, ob, self.perturbed_obstacles
        else:
            return state

    def onestep_lookahead(self, action, debug=True, visualize=None, display_fps=None):
        return self.step(action, update=False, debug=debug, visualize=None, display_fps=None)

    def step(self, action, update=True, debug=False, visualize=None, display_fps=None):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        self.visualize = visualize if visualize is not None else self.visualize
        self.display_fps = display_fps if display_fps is not None else self.display_fps

        # Convert the action variables to robot actions
        if self.robot.kinematics == 'holonomic':
            scaled_action = ActionXY(action[0] * self.robot.v_pref, action[1] * self.robot.v_pref)
            #action = ActionXY(action[0], action[1])
        else:
            # Only allow forward motion and rotations
            #action = ActionRot(((action[0] + 1.0) / 2.0) * self.robot.v_pref, action[1])
            scaled_action = ActionRot(((action[0] + 1.0) / 2.0) * self.robot.v_pref, action[1])
            
        self.n_steps += 1

        # Compute the next human actions from the current observations
        human_actions = []

        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
          
            #robot_visible = np.random.uniform()
            #if robot_visible < 0.5:
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob, create_obstacles=self.create_obstacles, obstacles=self.perturbed_obstacles))

        # Move humans and robot according to current observation and action
        if update:            
#             # store state, action value and attention weights
#             self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
#             
#             if hasattr(self.robot.policy, 'action_values'):
#                 self.action_values.append(self.robot.policy.action_values)
#                 
#             if hasattr(self.robot.policy, 'get_attention_weights'):
#                 self.attention_weights.append(self.robot.policy.get_attention_weights())

            # Move the robot
            if debug:
                if self.human_num == 0:
                    ob = [self.robot.get_observable_state()]
                
                robot_action = self.robot.act(ob, create_obstacles=self.create_obstacles, obstacles=self.perturbed_obstacles)
                self.robot.step(robot_action)
            else:
                self.robot.step(scaled_action)

            # Move each human
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
                
            self.global_time += self.time_step
            
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # Get the new observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
                
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
                
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
            
        # Convert observable state to numpy state for Tensorflow stable-baselines
        state = self._get_state()
            
        # Compute the reward
        reward, done, info = self._get_reward(scaled_action)

        # Update Pygame objects                 
        pygame_robot_px = int(self.scale_factor * self.robot.px + self.width/2)
        pygame_robot_py = int(self.scale_factor * self.robot.py + self.height/2)
            
        self.robot_body.position = Vec2d(pygame_robot_px, pygame_robot_py)
        self.robot_body.angle = self.robot.theta
        
        if self.visualize:
            pygame_gx = int(self.scale_factor * self.robot.gx + self.width / 2.0)
            pygame_gy = int(self.scale_factor * self.robot.gy + self.height / 2.0)                
            pygame.draw.circle(self.surface, (0, 255, 0, 200), (pygame_gx, self.screen_y(pygame_gy)), 30)

            pygame.draw.circle(self.surface, (255, 255, 255, 40), (pygame_robot_px, self.screen_y(pygame_robot_py)), int(self.scale_factor * self.robot.personal_space))
        
        index = 0
        
        for human in self.humans:
            human_state = human.get_full_state()
            
            pygame_px = int(self.scale_factor * human_state.px + self.width/2)
            pygame_py = int(self.scale_factor * human_state.py + self.height/2)
            
            self.pedestrians[index][0].position = Vec2d(pygame_px, pygame_py)

            # Show the elongation of the personal space in the direction of motion
            px = human_state.px - self.robot.px
            py = human_state.py - self.robot.py

            #if self.robot.kinematics == 'holonomic':
            vx = human_state.vx
            vy = human_state.vy
            #else:
            #    raise NotImplementedError
        
            ex = human_state.px + vx * self.lookahead_interval
            ey = human_state.py + vy * self.lookahead_interval

            pygame_ex = int(self.scale_factor * ex + self.width/2)
            pygame_ey = int(self.scale_factor * ey + self.height/2)

            if self.visualize:
                pygame.draw.circle(self.surface, (255, 255, 255, 40), (pygame_px, self.screen_y(pygame_py)), int(self.scale_factor * human.personal_space))

                pygame.draw.lines(self.surface, (0, 255, 0), False, [Vec2d(pygame_px, self.screen_y(pygame_py)), Vec2d(pygame_ex, self.screen_y(pygame_ey))], 3)                

                pygame_gx = int(self.scale_factor * human_state.gx + self.width/2)
                pygame_gy = int(self.scale_factor * human_state.gy + self.height/2)
                pygame.draw.circle(self.surface, (255, 0, 0, 200), (pygame_gx, self.screen_y(pygame_gy)), 10)                

            index += 1
            
        # Update Pymunk
        self.space.step(self.time_step)
            
        if self.visualize:
            for obstacle in self.pymunk_obstacles:
                if type(obstacle) == pymunk.shapes.Poly:
                    vertices = obstacle.get_vertices()
                    screen_vertices = list()
                    for vertex in vertices:
                        vertex[1] = self.screen_y(vertex[1])
                        screen_vertices.append(vertex)
                    pygame.draw.polygon(self.surface, (255, 0, 0, 100), screen_vertices)
                elif type(obstacle) == pymunk.shapes.Segment:
                    pygame.draw.line(self.surface, (255, 0, 0, 100), obstacle.a, obstacle.b, 3)
            
            self.space.debug_draw(self.draw_options)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.surface.fill(THECOLORS["black"])
            self.screen.fill(THECOLORS["black"])

            self.clock.tick(self.display_fps)
        if debug or self.expert_policy:
            return state, ob, reward, done, info
        else:
            return state, reward, done, info

    def _get_reward(self, action, debug=False):
        # collision detection
        dmin = float('inf')
        collision = False
    
        # minimum distance detection between robot and humans
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py

            # closest distance between boundaries of two agents
            closest_dist = np.linalg.norm([px, py]) - human.radius - self.robot.radius

            if closest_dist < dmin:
                dmin = closest_dist

        # velocity projection to elongate the personal space in the direction of motion
        velocity_dmin = float('inf')
        cutting_off = False
        
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py

            ex = px + human.vx * self.lookahead_interval
            ey = py + human.vy * self.lookahead_interval
            
            # get the nearest distance to segment connecting the current position and future position
            velocity_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius

            if velocity_dist < velocity_dmin:
                velocity_dmin = velocity_dist

        distance_from_goal = np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy]))

        if self.testing:
            reaching_goal = distance_from_goal < 2.0 * self.robot.radius
        else:
            reaching_goal = distance_from_goal < 1.0 * self.robot.radius

        done = False
        
        reward = 0
        
        info_dict = OrderedDict()
                
        if self.global_time >= self.time_limit - 1:
            self.n_timeouts += 1
            reward = 0
            done = True
            info = Timeout()
        elif self.crashed or self.ped_collision:
            # For pedestrians, only blame the robot if it is moving faster than a lower threshold
            if self.ped_collision:
                if np.linalg.norm([self.robot.vx, self.robot.vy]) < 0.05:
                    self.n_ped_hits_robot += 1
                else:
                    self.n_ped_collisions += 1
                    reward = self.collision_penalty
            else:
                self.n_collisions += 1
                reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            self.n_successes += 1
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # penalize agent for getting too close
            # adjust the reward based on FPS
            self.n_personal_space_violations += 1
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(dmin)
        elif velocity_dmin < self.discomfort_dist:
            # penalize agent for getting too close
            # adjust the reward based on FPS
            self.n_personal_space_violations += 1
            reward = (velocity_dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(velocity_dmin)
        else:
            done = False
            info = Nothing()

        if not done:
            # time cost (slack reward)
            reward += self.slack_reward * self.time_step
            
            # energy cost
            reward += self.energy_cost * np.linalg.norm(np.array([self.robot.vx, self.robot.vy])) * self.time_step
# 
#             # X-Y safety factor
#             if self.n_sonar_sensors > 0:
#                 sensor_readings = np.array(self.sensor_readings)
#                 sensor_readings = np.reshape(sensor_readings, (int(len(sensor_readings)/2), 2))                                             
#    
#                 min_obstacle_distance = np.min(np.linalg.norm(sensor_readings, axis=1))
#                 min_obstacle_distance *= self.max_pygame_sensor_range / self.scale_factor
#                    
#                 if min_obstacle_distance < self.safe_obstacle_distance:
#                     obstacle_cost = (min_obstacle_distance - self.safe_obstacle_distance) * self.safety_penalty_factor * self.time_step
#                 else:
#                     obstacle_cost = 0.0
#                    
#                 reward += obstacle_cost
            
            if self.n_sonar_sensors > 0:
                # Encourage free space motion
                sensor_readings = np.array(self.sensor_readings)
                mean_distance_to_obstacles = np.mean(sensor_readings)
                freespace_reward = mean_distance_to_obstacles * self.freespace_reward * self.time_step
                reward += freespace_reward

                # Distance safety factor
                min_obstacle_distance = min(sensor_readings[np.where(sensor_readings >= 0)])
                min_obstacle_distance *= self.max_pygame_sensor_range / self.scale_factor

                #if min_obstacle_distance < self.safe_obstacle_distance:
                #obstacle_cost = (min_obstacle_distance - self.safe_obstacle_distance) * self.safety_penalty_factor * self.time_step
                obstacle_cost = 1.0 / (max(0.0, min_obstacle_distance - self.safe_obstacle_distance - self.robot.radius) + 0.001)
                obstacle_cost *= self.safety_penalty_factor * self.time_step
                #else:
                #    obstacle_cost = 0.0
                
                #print(min_obstacle_distance, obstacle_cost)
                   
                reward += obstacle_cost

            # Get initial goal potential and collision potential
            if self.n_steps == 1:
                self.initial_potential = self.get_potential()
                self.normalized_potential = 1.0
            
            # Get delta potential and compute reward
            current_potential = self.get_potential()
            new_normalized_potential = current_potential / self.initial_potential
            potential_reward = self.normalized_potential - new_normalized_potential
            
            potent = potential_reward * self.potential_reward_weight
            #reward += potential_reward * self.potential_reward_weight / self.robot.v_pref
            reward += potential_reward * self.potential_reward_weight
            
            #print(obstacle_cost, potential_reward * self.potential_reward_weight)
            self.normalized_potential = new_normalized_potential
            info = Nothing()
        else:
            self.n_episodes += 1
                        
        #print("Collision?", collision, " Success?", reaching_goal, " Discomfort: ", discomfort, " Potential", potent, " Reward", reward)

        if not debug:
            info_dict = {'episodes': self.n_episodes,
                         'successes': self.n_successes,
                         'collisions': self.n_collisions,
                         'ped_collisions': self.n_ped_collisions,
                         'ped_hits_robot': self.n_ped_hits_robot,
                         'timeouts': self.n_timeouts,
                         'personal_space_violations': self.n_personal_space_violations,
                         'cutting_off': self.n_cutting_off,
                         'success_rate': 0 if self.n_episodes == 0 else 100 * self.n_successes / self.n_episodes,
                         'collision_rate': 0 if self.n_episodes == 0 else 100 * self.n_collisions / self.n_episodes,
                         'ped_collision_rate': 0 if self.n_episodes == 0 else 100 * self.n_ped_collisions / self.n_episodes,
                         'ped_hits_robot_rate': 0 if self.n_episodes == 0 else 100 * self.n_ped_hits_robot / self.n_episodes,
                         'timeout_rate': 0 if self.n_episodes == 0 else 100 * self.n_timeouts / self.n_episodes
                         }
                         
            info = info_dict

        return reward, done, info
    
    def get_potential(self):
        return np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])) 

    def get_sonar_readings_closest_point(self, x, y, angle):
        readings = []
        
        angle += self.angle_offset
                
        if self.n_sonar_sensors > 1:
            delta_theta = np.pi / self.n_sonar_sensors
        else:
            delta_theta = 0
        
        for i in range(self.n_sonar_sensors):
            j = int(self.n_sonar_sensors / 2) - i
            sensor_angle = angle + j * delta_theta
            readings.append(self.detect_sensor_ping_xy(x, y, sensor_angle, i))
                                
        min_distance = np.Inf
        closest_point = None
        
        for reading in readings:
            distance = np.linalg.norm(reading)
            if distance < min_distance:
                min_distance = distance
                closest_point = reading
                                        
        return closest_point

    def detect_sensor_ping_xy(self, x, y, angle, index):        
        x1 = x + self.max_pygame_sensor_range * np.sqrt(2) * np.cos(angle)
        y1 = y + self.max_pygame_sensor_range * np.sqrt(2) * np.sin(angle)
                
        if self.visualize and self.show_sensors:
            if index == int(self.n_sonar_sensors / 2):
                pygame.draw.lines(self.surface, (255, 0, 0), False, [Vec2d(x,self.screen_y(y)), Vec2d(x1, self.screen_y(y1))], 1)
            else:
                pygame.draw.lines(self.surface, (255, 255, 0), False, [Vec2d(x,self.screen_y(y)), Vec2d(x1, self.screen_y(y1))], 1)

        robot_filter = pymunk.ShapeFilter(mask=(pymunk.ShapeFilter.ALL_MASKS ^ 1))
        seqment_query_info = self.space.segment_query_first(Vec2d(x, y), Vec2d(x1, y1), 0, shape_filter=robot_filter)

        if seqment_query_info is not None:
#             # circles are pedestrians
#             if type(seqment_query_info.shape) == pymunk.shapes.Circle:
#                 rel_x = x1 - x
#                 rel_y = y1 - y
#                 ping_x = x
#                 ping_y = y
#             else:
            ping_x = seqment_query_info.point.x
            ping_y = seqment_query_info.point.y

            rel_x = ping_x - x
            rel_y = ping_y - y
        else:
            rel_x = x1 - x
            rel_y = y1 - y
            ping_x = rel_x
            ping_y = rel_y
            
        if self.visualize and self.show_sensors:
            pygame.draw.circle(self.surface, (0, 255, 255, 200), (int(ping_x), self.screen_y(int(ping_y))), 10)
        
        # add some noise
        rel_x = (1.0 + random.uniform(-1, 1) * self.position_noise) * rel_x
        rel_y = (1.0 + random.uniform(-1, 1) * self.position_noise) * rel_y
        
        return np.array([rel_x, rel_y])
    
#     def get_sonar_readings_xy(self, x, y, angle):
#         readings = []
#         
#         angle += self.angle_offset
#                 
#         if self.n_sonar_sensors > 1:
#             delta_theta = np.pi / self.n_sonar_sensors
#         else:
#             delta_theta = 0
#         
#         for i in range(self.n_sonar_sensors):
#             j = int(self.n_sonar_sensors / 2) - i
#             sensor_angle = angle + j * delta_theta
#             sensor_point = self.detect_sensor_ping_xy(x, y, sensor_angle, i)
#             readings.append(sensor_point)
# 
#         np_readings = np.array(readings, dtype=np.float32)
# 
#         np_readings /= (self.square_width * self.scale_factor)
#         
#         #print(list(np_readings))
#                         
#         return np_readings
    
    def get_sonar_readings_xy(self, x, y, angle):
        readings = []
        
        angle += self.angle_offset
                        
        if self.n_sonar_sensors > 1:
            delta_theta = np.pi / self.n_sonar_sensors
        else:
            delta_theta = 0
        
        for i in range(self.n_sonar_sensors):
            j = int(self.n_sonar_sensors / 2) - i
            sensor_angle = angle + j * delta_theta
            sensor_point = self.detect_sensor_ping_xy(x, y, sensor_angle, i)
            readings.append(sensor_point[0])
            readings.append(sensor_point[1])
            
        np_readings = np.array(readings, dtype=np.float32)

        np_readings /= (self.square_width * self.scale_factor)
        
        #print(list(np_readings))
                        
        return np_readings
    
    def get_sonar_readings(self, x, y, angle):
        readings = []
        
        angle += self.angle_offset
                
        if self.n_sonar_sensors > 1:
            delta_theta = np.pi / self.n_sonar_sensors
        else:
            delta_theta = 0
        
        for i in range(self.n_sonar_sensors):
            j = int(self.n_sonar_sensors / 2) - i
            sensor_angle = angle + j * delta_theta
            readings.append(self.detect_sensor_ping(x, y, sensor_angle, i))
            
        np_readings = np.array(readings, dtype=np.float32)

        np_readings /= (self.square_width * self.scale_factor)
        
        #print(list(np_readings))
                        
        return np_readings

    def detect_sensor_ping(self, x, y, angle, index):        
        x1 = x + self.max_pygame_sensor_range * np.cos(angle)
        y1 = y + self.max_pygame_sensor_range * np.sin(angle)
        
        if self.visualize and self.show_sensors:
            if index == int(self.n_sonar_sensors / 2):
                pygame.draw.lines(self.surface, (255, 0, 0), False, [Vec2d(x,self.screen_y(y)), Vec2d(x1, self.screen_y(y1))], 1)
            else:
                pygame.draw.lines(self.surface, (255, 255, 0), False, [Vec2d(x,self.screen_y(y)), Vec2d(x1, self.screen_y(y1))], 1)

        robot_filter = pymunk.ShapeFilter(mask=(pymunk.ShapeFilter.ALL_MASKS ^ 1))
        seqment_query_info = self.space.segment_query_first(Vec2d(x,y), Vec2d(x1, y1), 0, shape_filter=robot_filter)

        if seqment_query_info is not None:
            ping_x = seqment_query_info.point.x
            ping_y = seqment_query_info.point.y
            distance = np.linalg.norm((ping_x - x, ping_y - y))
            if self.visualize and self.show_sensors:
                pygame.draw.circle(self.surface, (0, 255, 255, 200), (int(ping_x), self.screen_y(int(ping_y))), 10)
        else:
            distance = self.square_width * self.scale_factor
        
        # add some noise
        distance = (1.0 + random.uniform(-1, 1) * self.position_noise) * distance

        return distance

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

    def collision_object_begin(self, space, arbiter, data):
        self.crashed = True
        return True
    
    def collision_object_separate(self, space, arbiter, data):
        self.crashed = False
        return False
    
    def collision_pedestrian_begin(self, space, arbiter, data):
        self.ped_collision = True
        return True
    
    def collision_pedestrian_separate(self, space, arbiter, data):
        self.ped_collision = False
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
