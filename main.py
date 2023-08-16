import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.utils import sample_item
register(
    id='KnapsackEnv-v0',
    entry_point='envs.knapsack_env:KnapsackEnv', #your_env_folder.envs:NameOfYourEnv
)

env = DummyVecEnv([lambda: gym.make('KnapsackEnv-v0',N0=50, T=100, item_distributions=sample_item, m=5)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)

# To evaluate or use the model after training
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print('obs', obs)




# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[128, 128], vf=[128, 128]))
#
# model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, batch_size=128, learning_rate=0.0004, policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=110, log_interval=10, progress_bar=True) #
#
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")































##EVALUATION


# vec_env = model.get_env()
#
# vec_env = model.get_env()
# obs = vec_env.reset()
# total_reward = np.array([0.,0.,0.,0.])
# dones = False
# for i in range(1000):
#     action_masks = get_action_masks(env)
#     action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
#     obs, rewards, dones, info = env.step(action)
#     r = np.array(rewards)
#     total_reward += r
#     if (dones[0] == True):
#         print("FIRST")
#         print(rewards)
#         print(total_reward)
#     if (dones[1] == True):
#         print("SECOND")
#         print(rewards)
#         print(total_reward)
#     if (dones[2] == True):
#         print("THIRD")
#         print(rewards)
#         print(total_reward)
#     if (dones[3] == True):
#         print("FOURTH")
#         print(rewards)
#         print(total_reward)

    # env.render()
#









# def mask_fn(env: gym.Env) -> np.ndarray:
#     # Do whatever you'd like in this function to return the action mask
#     # for the current env. In this example, we assume the env has a
#     # helpful method we can rely on.
#     return env.valid_action_mask()
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100, log_interval=1)
# model.save("dqn_cartpole")
#
# del model
# model = PPO.load("dqn_cartpole")


# for i in range(10):
#     action, _states = model.predict(obs)
#     print('action', action)
#     obs, rewards, dones, info = env.step(action)
#     print('obs', obs)

# del model
#
# model = DQN.load("dqn_cartpole")
# obs = env.reset()
# for i in range(100):
#     action, _state = model.predict(obs, deterministic=True, action_masks=env.valid_action_mask())
#     print(action)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     # VecEnv resets automatically
#     if done:
#       obs = env.reset()