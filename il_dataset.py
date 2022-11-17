import os
import sys
sys.path.insert(0, '.')

import argparse
import gym
import gym_minigrid
import torch
from PIL import Image

from expert.grid_world_expert import GridWorldExpert
from gym_minigrid.wrappers import ILObsWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--num_eps', type=int, required=True,
                help='number of trajectories to save')
parser.add_argument('--dir', type=str, required=True,
                help='path to save trajectories')
parser.add_argument('--render', action="store_true", default=False)

args = parser.parse_args()
num_episodes = args.num_eps
data_dir = args.dir
render_image = args.render

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

env_name = 'MiniGrid-FourRooms-v0'
env = ILObsWrapper(gym.make(env_name))

def reset_data():
    return {
        "obs": [],
        "direction": [],
        "next_obs": [],
        "actions": [],
        "done": [],
        "reward": [],
    }

def append_data(data, s, d, ns, a, done, reward):
    data["obs"].append(s)
    data["direction"].append(d)
    data["next_obs"].append(ns)
    data["actions"].append(a)
    data["done"].append(done)
    data["reward"].append(reward)

def extend_data(data, episode):
    data["obs"].extend(episode["obs"])
    data["direction"].extend(episode["direction"])
    data["next_obs"].extend(episode["next_obs"])
    data["actions"].extend(episode["actions"])
    data["done"].extend(episode["done"])
    data["reward"].extend(episode["reward"])
    # print(len(data["obs"]), len(data["next_obs"]), len(data["actions"]), len(data["done"]), len(data["reward"]))

def render(env, fn):
    img = Image.fromarray(env.render('rgb_array'))
    img.save(os.path.join(data_dir, fn))

def extract_data(env, obs, direction, actions, render_image=False):
    episode = reset_data()
    done = False
    i = 0
    if render_image:
        render(env, '%05d_initial.png' % i)
    while not done and i < len(actions):
        next_obs, reward, done, info = env.step(actions[i])
        i += 1
        if render_image:
            render(env, '%05d_action_%d.png' % (i, actions[i-1]))
        append_data(episode, obs, next_obs['direction'], next_obs['image'], actions[i-1], done, reward)
        # if info['ep_found_goal']:
        #     print('found goal at step', i, len(actions))
        obs = next_obs['image']
        direction = next_obs['direction']

    if i == len(actions):
        return episode, reward > 0 # info['ep_found_goal'] == 1
    else:
        return None, False

def post_process(path, direction):
    new_path = []
    for p in path:
        if direction == p:
            new_path.append(2)
        elif (direction + 1) % 4 == p:
            # turn right, move forward
            new_path.extend([1, 2])
        elif (direction + 3) % 4 == p:
            # turn left, move forward
            new_path.extend([0, 2])
        else:
            # turn back, move forward
            new_path.extend([1, 1, 2])
        direction = p
    return new_path

expert = GridWorldExpert()
cur_count = 0
success_count = 0
data = reset_data()
while cur_count < num_episodes:
    obs = env.reset()
    obs, planner_obs, direction = obs['image'], obs['planner_image'], obs['direction']
    actions = post_process(expert._solve_env(planner_obs), direction)
    episode, success = extract_data(env, obs, direction, actions)
    if episode is None:
        continue
    else:
        extend_data(data, episode)
        success_count += success
        cur_count += 1

if render_image:
    obs = env.reset()
    obs, planner_obs, direction = obs['image'], obs['planner_image'], obs['direction']
    actions = post_process(expert._solve_env(planner_obs), direction)
    episode, success = extract_data(env, obs, direction, actions, render_image=True)

dones = data["done"]
obs = data["obs"]
next_obs = data["next_obs"]
actions = data["actions"]
rewards = data["reward"]
torch.save(
    {
        "done": torch.FloatTensor(dones),
        "obs": torch.tensor(obs),
        "next_obs": torch.tensor(next_obs),
        "actions": torch.tensor(actions),
        "rewards": torch.tensor(rewards),
    },
    os.path.join(data_dir, 'data.pt'),
)
print("Saved to ", data_dir)
print("Num episodes:", num_episodes, success_count)
print("Num steps:", len(data["obs"]))
