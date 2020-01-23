import collections
import gym
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np
import cv2
import inspect
import itertools
import os


# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


_FILENAMES = [
    "./common/materials.xml",
    "./common/skybox.xml",
    "./common/visual.xml",
]

def read_model(xml_path):
    return resources.GetResource(xml_path)


def get_assets(dir_path):
    assets = {filename: resources.GetResource(os.path.join(dir_path, filename))
                for filename in _FILENAMES}
    return assets

def get_model_and_assets(xml_path, dir_path):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return read_model(xml_path), get_assets(dir_path)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def speed(self):
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]


class HalfCheetah(base.Task):
    """A `Task` to train a running Cheetah."""

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        for _ in range(200):
            physics.step()

        physics.data.time = 0
        self._timeout_progress = 0
        super(HalfCheetah, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(physics.speed(),
                        bounds=(_RUN_SPEED, float('inf')),
                        margin=_RUN_SPEED,
                        value_at_margin=0,
                        sigmoid='linear')


class HalfCheetahEnv(object):
    def __init__(self, xml_path, dir_path, use_pixel=False):
        physics = Physics.from_xml_string(*get_model_and_assets(xml_path, dir_path))
        self.env = control.Environment(physics, HalfCheetah(random=None), time_limit=_DEFAULT_TIME_LIMIT)
        self.n_actions = self.env.action_spec().shape[0]
        self.action_space = gym.spaces.Box(low=np.ones(self.n_actions)*-1, high=np.ones(self.n_actions), dtype=np.float32)
        if use_pixel:
            self.observation_space = gym.spaces.Box(low=np.zeros((64, 64, 3)), high=np.ones((64, 64, 3)), dtype=np.float32)
        else:
            n_states = 1
            for key in self.env.observation_spec().keys():
                n_states += self.env.observation_spec()[key].shape[0]
            self.observation_space = gym.spaces.Box(low=np.ones(n_states)*-200, high=np.ones(n_states)*200, dtype=np.float32)
        self.use_pixel = use_pixel
        self.num_steps = 0
        self.previous_time = 0

 
    def reset(self):
        self.num_steps = 0
        self.previous_time = 0
        time_step = self.env.reset()
        if not self.use_pixel:
            state = self.env._physics.get_state()
        else:
            state = self.env._physics.render(camera_id=0)
            state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_LINEAR)
        return state


    def step(self, action):
        assert action.shape[0] == self.n_actions
        time_step = self.env.step(action)
        if not self.use_pixel:
            state = self.env._physics.get_state()
        else:
            state = self.env._physics.render(camera_id=0)
            state = cv2.resize(state, (64, 64), interpolation=cv2.INTER_LINEAR)
        reward = time_step.reward
        self.num_steps += 1
        done = self.num_steps >= 1000
        return state, reward, done, None

if __name__ == '__main__':
    xml_path = 'common/cheetah.xml'
    dir_path = os.path.dirname(os.path.abspath(__file__))
    env = HalfCheetahEnv(xml_path, dir_path, True)
    obs = env.reset()
    state = env.env._physics.get_state()
    obs = env.env._physics.render(camera_id=0)
    action = env.action_space.sample()
    _ = env.step(action)
    next_state = env.env._physics.get_state()
    # set to the previous state and get obs
    env.env._physics.set_state(state)
    new_obs = env.env._physics.render(camera_id=0)
    diff = (obs - new_obs)**2.0
    print(np.sum(diff))
    cv2.imwrite('diff.png', obs-new_obs)
