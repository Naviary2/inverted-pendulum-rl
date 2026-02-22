# pendulum/train.py

"""Train a PPO agent to balance the inverted pendulum.

Uses stable-baselines3 with SubprocVecEnv for parallel training across
all available CPU cores.

Usage:
    python -m pendulum.train                # defaults
    python -m pendulum.train --timesteps 1000000 --n-envs 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from .config import PendulumConfig, TrainingConfig
from .environment import CartPendulumEnv


def _make_env(pendulum_cfg: PendulumConfig, max_episode_steps: int):
    """Return a callable that creates a fresh environment instance."""

    def _init():
        return Monitor(
            CartPendulumEnv(
                pendulum_config=pendulum_cfg,
                max_episode_steps=max_episode_steps,
            )
        ) 

    return _init


def train(
    pendulum_cfg: PendulumConfig | None = None,
    training_cfg: TrainingConfig | None = None,
) -> PPO:
    """Run PPO training and return the trained model."""
    p_cfg = pendulum_cfg or PendulumConfig()
    t_cfg = training_cfg or TrainingConfig()

    # Parallel environments
    env_fns = [_make_env(p_cfg, t_cfg.max_episode_steps) for _ in range(t_cfg.n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Eval environment (single process)
    eval_env = Monitor(
        CartPendulumEnv(
            pendulum_config=p_cfg,
            max_episode_steps=t_cfg.max_episode_steps,
        )
    )

    # Ensure save directory exists
    save_dir = Path(t_cfg.model_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "logs"),
        eval_freq=max(t_cfg.n_steps // t_cfg.n_envs, 1) * 5,
        n_eval_episodes=10,
        deterministic=True,
    )

    if t_cfg.model_load_path:
        print(f"Loading existing model from {t_cfg.model_load_path} …")
        print("Note: hyperparameters (learning_rate, n_steps, etc.) are loaded "
              "from the saved model; TrainingConfig hyperparameters are ignored.")
        model = PPO.load(
            t_cfg.model_load_path,
            env=vec_env,
            verbose=1,
            tensorboard_log=t_cfg.tensorboard_log or None,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=t_cfg.learning_rate,
            n_steps=t_cfg.n_steps,
            batch_size=t_cfg.batch_size,
            n_epochs=t_cfg.n_epochs,
            gamma=t_cfg.gamma,
            gae_lambda=t_cfg.gae_lambda,
            clip_range=t_cfg.clip_range,
            ent_coef=t_cfg.ent_coef,
            verbose=1,
            tensorboard_log=t_cfg.tensorboard_log or None,
        )

    # print(model.policy) # Print the policy architecture (number of inputs, hidden layers, neurons, outputs)

    print(f"Training with {t_cfg.n_envs} parallel environments "
          f"for {t_cfg.total_timesteps:,} timesteps …")
    if t_cfg.tensorboard_log:
        print(f"TensorBoard logs: {t_cfg.tensorboard_log}  "
              f"(run: tensorboard --logdir {t_cfg.tensorboard_log})")
    model.learn(total_timesteps=t_cfg.total_timesteps, callback=eval_callback)
    model.save(str(save_dir / "final"))
    print(f"Model saved to {str(save_dir / 'final')}")

    vec_env.close()
    eval_env.close()
    return model


# ---- CLI entry point -------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on inverted pendulum")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel envs (default: all cores)")
    parser.add_argument("--save-path", type=str, default="ppo_pendulum",
                        help="Directory inside models/ to save the trained model")
    parser.add_argument("--load-model", type=str, default="",
                        help="Path inside models/ to an existing model to continue training")
    parser.add_argument("--tensorboard-log", type=str, default="logs/tensorboard",
                        help="Directory for TensorBoard logs (empty string to disable)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_path = str(Path("models") / args.save_path)
    load_path = str(Path("models") / args.load_model) if args.load_model else ""
    t_cfg = TrainingConfig(
        total_timesteps=args.timesteps,
        model_save_path=save_path,
        model_load_path=load_path,
        tensorboard_log=args.tensorboard_log,
    )
    if args.n_envs is not None:
        t_cfg.n_envs = args.n_envs
    train(training_cfg=t_cfg)
