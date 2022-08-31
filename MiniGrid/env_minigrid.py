from queue import Full
import gym
import gym_minigrid
import random
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import *
from gym_minigrid.register import register
from matplotlib import pyplot as plt

class RandomGoalEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
        goal_pos = None
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # goal_pos = np.random.randint(width-2), np.random.randint(1, height-2)
        goal_pos = self.goal_pos
        # self.place_obj(Goal())
        self.put_obj(Goal(), goal_pos[0], goal_pos[1])


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class FourRoomsEnvExtended(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=7, max_steps=200)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info

register(
    id='MiniGrid-FourRoomsExt-v0',
    entry_point='env_minigrid:FourRoomsEnvExtended'
)


class RandomGoalEnv6x6(RandomGoalEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, goal_pos=(random.randint(1, 4),random.randint(2,4)), **kwargs)

register(
    id='MiniGrid-RandomGoal-6x6-v0',
    entry_point='env_minigrid:RandomGoalEnv6x6'
)


class GymMiniGridEnv:
    def __init__(self, game, seed, full_state):
        self.game = game
        self.env = gym.make(game)
        self.env.seed(seed)
        random.seed(seed)
        self.full_state = full_state
        # if self.full_state:
        #     self.env = RGBImgObsWrapper(self.env) # Get pixel observations
        # self.env = FullyObsWrapper(self.env)
        self.env = ImgObsWrapper(self.env) # Get rid of the 'mission' field
        self.num_actions = self.env.action_space.n
        if not self.full_state:
            sample_state = np.atleast_3d(self.reset())
            sample_state = sample_state.flatten()
            self.num_states = len(sample_state)
        # print(sample_state)
        # print(state_space)
        # # self.env.render()
        # input()
        # raise
    
    def reset(self):
        if self.full_state:
            return self.env.reset()
        else:
            return self.env.reset().flatten()

    def full_reset(self):
        self.env = gym.make(self.game)
        self.env = ImgObsWrapper(self.env)
        if self.full_state:
            return self.env.reset()
        else:
            return self.env.reset().flatten()

    def step(self, action):
        next_obs, reward, done, _ = self.env.step(action)
        if self.full_state:
            return next_obs, reward, done
        else:
            return next_obs.flatten(), reward, done
    
    def close(self):
        return self.env.close()