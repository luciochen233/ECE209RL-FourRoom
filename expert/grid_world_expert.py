import numpy as np

from .solve_policy import SolvePolicy
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def pretty_print_graph(graph, agent_pos, goal_pos):
    for j in range(graph.shape[1]):
        row_str = ''
        for i in range(graph.shape[0]):
            if (i,j) == agent_pos:
                row_str += 'A'
            elif (i,j) == goal_pos:
                row_str += 'G'
            else:
                row_str += str(int(graph[i,j]))
        print(row_str)

def convert_to_graph(grid_state):
    width, height, _ = grid_state.shape
    graph = np.zeros((width, height))
    agent_pos = None
    goal_pos = None
    for i in range(width):
        for j in range(height):
            grid_val = grid_state[i, j, :]
            node_type = np.argmax(grid_val)
            if node_type == 2:
                # Goal
                goal_pos = (i, j)
                node_val = 1
            elif node_type == 3:
                # Agent start
                agent_pos = (i, j)
                node_val = 1
            elif node_type == 1:
                # Wall
                node_val = 0
            elif node_type == 0:
                # Empty
                node_val = 1
            else:
                print(grid_val)
                raise ValueError('Unrecognized grid val')
            graph[j, i] = node_val
    return graph, agent_pos, goal_pos


class GridWorldExpert(SolvePolicy):
    def _solve_env(self, state):
        graph, agent_pos, goal_pos = convert_to_graph(state)

        grid = Grid(matrix=graph.tolist())
        start_node = grid.node(*agent_pos)
        end_node = grid.node(*goal_pos)
        finder = AStarFinder()
        path, _ = finder.find_path(start_node, end_node, grid)
        # For debugging print this out.
        # print(grid.grid_str(path=path, start=start_node, end=end_node))
        path_diffs = [(b[0]-a[0], b[1]-a[1]) for a,b in zip(path, path[1:])]

        # Found through brute force.
        diff_translate = {
                # left
                (-1,0): 2,
                # up
                (0,-1): 3,
                # right
                (1,0): 0,
                # down
                (0,1): 1,
                }

        path_diffs = [diff_translate[x] for x in path_diffs]
        return path_diffs
