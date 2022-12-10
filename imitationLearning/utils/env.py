import numpy as np
import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


def make_env(env_key, seed=None, render_mode=None,agent_view_size_param=7):
    #env = gym.make(env_key, render_mode=render_mode)
    env = gym.make(env_key, render_mode=render_mode, agent_view_size=agent_view_size_param)
    env.reset(seed=seed)
    return env


class ILDatasetWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, il_view_size=7):
        super().__init__(env)

        assert il_view_size % 2 == 1
        assert il_view_size >= 3

        self.il_view_size = il_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(il_view_size, il_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, "il_image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.il_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "il_image": image}


NODE_TO_ONE_HOT = {
    # Empty square
    (1, 0, 0): [1, 0, 0, 0],
    # Wall
    (2, 5, 0): [0, 1, 0, 0],
    # Goal
    (8, 1, 0): [0, 0, 1, 0],
    # Agent
    (10, 0, 0): [0, 0, 0, 1],
    (10, 0, 1): [0, 0, 0, 1],
    (10, 0, 2): [0, 0, 0, 1],
    (10, 0, 3): [0, 0, 0, 1],
}


class ExpertKnowledgeWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_sizes=[9, 11, 15]):
        super().__init__(env)

        assert all(agent_view_size % 2 == 1 for agent_view_size in agent_view_sizes)
        self.agent_view_sizes = agent_view_sizes

        # Compute observation space with specified view size
        obs_space_dict = {**self.observation_space.spaces}
        for agent_view_size in agent_view_sizes:
            new_image_space = gym.spaces.Box(
                low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
            )
            obs_space_dict['expert%d_image' % agent_view_size] = new_image_space

        # Override the environment's observation spaceexit
        self.observation_space = gym.spaces.Dict(obs_space_dict)

    def observation(self, obs):
        new_obs = {**obs}
        env = self.unwrapped

        for agent_view_size in self.agent_view_sizes:
            grid, vis_mask = env.gen_obs_grid(agent_view_size)

            # Encode the partially observable view into a numpy array
            image = grid.encode(vis_mask)
            new_obs['expert%d_image' % agent_view_size] = image

        return new_obs

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        grid_shape = full_grid.shape
        full_grid = full_grid.reshape(-1, 3)
        full_grid = np.array(list(map(lambda x: NODE_TO_ONE_HOT[tuple(x)], full_grid)))
        grid_shape = grid_shape[:-1] + (4,) # last dim of NODE_TO_ONE_HOT
        full_grid = full_grid.reshape(grid_shape)

        obs['planner_image'] = full_grid
        obs['direction'] = self.agent_dir
        return obs, {}
