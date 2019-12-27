import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('CartPole-v0')

model = DQN(MlpPolicy, env, verbose=1, learning_rate=1e-3, buffer_size=50000, tensorboard_log="./tensorboard_logs/stable_baselines")
#model.learn(total_timesteps=100000, log_interval=100)
#model.save("deepq_cartpole")

#del model # remove to demonstrate saving and loading

model = DQN.load("deepq_cartpole")

obs = env.reset()

n_episodes = 0

reward = 0.0

while n_episodes < 100:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    reward += rewards
    env.render()
    if dones:
        n_episodes += 1
        print("Episode:", n_episodes, "Reward:", reward)
        reward = 0
        obs = env.reset()
    