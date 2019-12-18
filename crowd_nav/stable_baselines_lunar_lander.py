import gym
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy as PPO2_MlpPolicy
from stable_baselines import SAC, PPO2

#env = gym.make('LunarLanderContinuous-v2')
#env = DummyVecEnv([lambda: env])

env = make_vec_env('LunarLanderContinuous-v2', n_envs=1)

rl_algorithm = "sac"
model_name = rl_algorithm + "_lunar_lander"

if rl_algorithm == "sac":
    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard_logs/stable_baselines")
elif rl_algorithm == "ppo2":
    model = PPO2(PPO2_MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard_logs/stable_baselines")
    
#model.learn(total_timesteps=500000, log_interval=10)
#model.save(model_name)

#del model # remove to demonstrate saving and loading

if rl_algorithm == "sac":
    model = SAC.load(model_name)
elif rl_algorithm == "ppo2":
    model = PPO2.load(model_name)

obs = env.reset()

ave_episode_reward = 0
total_episode_reward = 0
n_episodes = 0
episode_reward = 0

while n_episodes < 100:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    if done:
        total_episode_reward += episode_reward
        episode_reward = 0
        n_episodes += 1
        ave_episode_reward = total_episode_reward / n_episodes
        print("Episode: " , n_episodes, " Ave Reward: ", ave_episode_reward)
    env.render()
    
