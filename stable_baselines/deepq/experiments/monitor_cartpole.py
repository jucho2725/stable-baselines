
'''
This is slightly edited copy of train_cartpole.py
It also includes callback function for monitoring
'''

import argparse

import gym
import numpy as np


from stable_baselines.deepq import DQN, MlpPolicy

'''for monitoring'''
import os
import matplotlib.pyplot as plt

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy


# def callback(lcl, _glb):
#     """
#     The callback function for logging and saving
#
#     :param lcl: (dict) the local variables
#     :param _glb: (dict) the global variables
#     :return: (bool) is solved
#     """
#     # stop training if reward exceeds 199
#     if len(lcl['episode_rewards'][-101:-1]) == 0:
#         mean_100ep_reward = -np.inf
#     else:
#         mean_100ep_reward = round(float(np.mean(lcl['episode_rewards'][-101:-1])), 1)
#     is_solved = lcl['step'] > 100 and mean_100ep_reward >= 199
#     return not is_solved


def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True

def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    # y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

def main(args):
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    print("Making a new model")
    env = gym.make("CartPole-v0")
    env = Monitor(env, log_dir, allow_early_resets=True)
    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
    # model.learn(total_timesteps=args.max_timesteps, callback=callback)
    print("Learning started. It takes some time...")
    model.learn(total_timesteps=args.max_timesteps)

    print("Saving model to Cartpole_model3.pkl")
    model.save("Cartpole_model3.pkl")
    print("Plotting Learning Curve")
    plot_results(log_dir)

if __name__ == '__main__':
    # Create log dir
    log_dir = "/tmp/190207/"
    os.makedirs(log_dir, exist_ok=True)
    best_mean_reward, n_steps = -np.inf, 0

    parser = argparse.ArgumentParser(description="Train DQN on Cartpole")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
