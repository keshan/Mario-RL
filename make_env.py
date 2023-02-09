import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from loguru import logger
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

TEST_EPISODES = 500


def make_env(show_plot=False, testing=False):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    if testing:
        return env

    logger.info(f"Shape of the state {env.reset().shape}")
    env = GrayScaleObservation(env, keep_dim=True)

    if show_plot:
        plt.imshow(env.reset())
        plt.show()

    env = DummyVecEnv([lambda: env])
    # Stacking 4 frames together so, the model can get an idea of the 'movements'.
    # Channels are at the last position of the matrix.
    env = VecFrameStack(env, 4, channels_order="last")

    logger.info(f"Mario preprocessed env created! New shape {env.reset().shape}")

    return env


def test_run():

    env = make_env(testing=True)
    env.reset()

    for i in range(TEST_EPISODES):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        logger.info(f"Reward at episode {i} = {reward}")
    env.close()


if __name__ == "__main__":
    test_run()
