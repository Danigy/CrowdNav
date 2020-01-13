import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

env = make_vec_env('LunarLanderContinuous-v2', n_envs=1)

model_name = "sac_lunar_lander"

model = SAC(MlpPolicy, env, verbose=1, learning_rate=0.0005, buffer_size=100000, tensorboard_log="./tensorboard_logs/stable_baselines_test_1500000")

model.learn(total_timesteps=1500000, log_interval=10)
model.save(model_name)
