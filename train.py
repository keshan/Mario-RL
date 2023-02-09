import argparse
import os
import time

from loguru import logger
from make_env import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def setup(save, log):
    current_time = str(time.time_ns())

    MODEL_PATH = os.path.join("./models", current_time)
    LOG_PATH = os.path.join("./logs", current_time)

    if save:
        os.makedirs(MODEL_PATH, exist_ok=True)
        logger.debug(f"Model path created: {MODEL_PATH}")
    if log:
        os.makedirs(LOG_PATH, exist_ok=True)
        logger.debug(f"Log path created: {LOG_PATH}")

    return MODEL_PATH, LOG_PATH


def train(env, model_path, log_path):
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log=log_path,
        learning_rate=1e-6,
        n_steps=512,
    )
    logger.info("====== Training Starting ======")
    model.learn(1000, progress_bar=True, callback=[eval_callback])
    logger.info("====== Finished training ======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mario RL")

    parser.add_argument("--show", default=False)
    parser.add_argument("--save_best", default=True)
    parser.add_argument("--log", default=True)

    args = parser.parse_args()

    MODEL_PATH, LOG_PATH = setup(args.save_best, args.log)
    env = make_env(show_plot=args.show)
    train(env, MODEL_PATH, LOG_PATH)
