#!/usr/bin/env python3

import tensorflow as tf

from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy, SACPolicy

class CustomPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

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
            #vf_latent = vf_h

            #self.proba_distribution, self.policy, self.q_value = \
            #    self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        #self.initial_state = None        
        #self._setup_init()
        
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

# class KerasPolicy(MlpPolicy):
#     def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **kwargs):
#         super(KerasPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)
# 
#         with tf.variable_scope("model", reuse=reuse):
#             flat = tf.keras.layers.Flatten()(self.processed_obs)
# 
#             x = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc_0')(flat)
#             pi_latent = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc_1')(x)
# 
#             x1 = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc_0')(flat)
#             vf_latent = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc_1')(x1)
# 
#             value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)
# 
#             self.proba_distribution, self.policy, self.q_value = \
#                 self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
# 
#         self.value_fn = value_fn
#         self.initial_state = None
#         self._setup_init()
# 
#     def step(self, obs, state=None, mask=None, deterministic=False):
#         print(self.action)
#         if deterministic:
#             action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
#                                                    {self.obs_ph: obs})
#         else:
#             action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
#                                                    {self.obs_ph: obs})
#         return action, value, self.initial_state, neglogp
# 
#     def proba_step(self, obs, state=None, mask=None):
#         return self.sess.run(self.policy_proba, {self.obs_ph: obs})
# 
#     def value(self, obs, state=None, mask=None):
#         return self.sess.run(self._value, {self.obs_ph: obs})

model = SAC(CustomPolicy, "Pendulum-v0", verbose=1)
model.learn(25000)

env = model.get_env()
obs = env.reset()

reward_sum = 0.0
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    env.render()
    if done:
        print("Reward: ", reward_sum)
        reward_sum = 0.0
        obs = env.reset()

env.close()
