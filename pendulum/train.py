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
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from .config import PendulumConfig, TrainingConfig
from .environment import CartPendulumEnv

# The filename (without extension) used for the fully-trained model inside its model directory.
FINAL_MODEL_FILENAME = "final"
LIVE_DASHBOARD_DATA_FILENAME = "live"

# ==============================================================================
# Custom Callback for Live Visualization and Checkpointing
# ==============================================================================

class LiveDashboardCallback(BaseCallback):
    """
    A custom callback that saves data for a live-updating visualizer,
    correctly handling parallel environments by batching updates.
    """

    def __init__(self, save_dir: Path, save_freq: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq

        # Define file paths
        self.latest_model_path = self.save_dir / "latest_model.zip"
        self.stats_path = self.save_dir / "live_stats.json"
        
        # Initialize tracking variables
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """This method is called after each step in the training."""
        
        # --- Batch the updates for parallel environments ---
        newly_finished_episodes = 0
        last_checkpoint_episode = 0

        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    newly_finished_episodes += 1

                    print(f"Episode {self.episode_count + newly_finished_episodes} finished with reward {info['episode']['r']} and length {info['episode']['l']} steps.")
                    
                    # --- 1. Collect all stats from this step first ---
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    
                    # --- 2. Check if a checkpoint should be saved ---
                    # We check if the *previous* count was below the threshold
                    # and the *new* count is at or above it.
                    if (self.episode_count // self.save_freq) < ((self.episode_count + newly_finished_episodes) // self.save_freq):
                        # Store the episode number for the checkpoint, but don't save yet.
                        # This ensures only one checkpoint is saved per batch.
                        last_checkpoint_episode = self.episode_count + newly_finished_episodes
                        print(f"Checkpoint triggered at episode {last_checkpoint_episode} (after processing batch)")

            # Update the total episode count after processing the batch
            self.episode_count += newly_finished_episodes

        # --- Perform a single save operation after the loop ---
        if newly_finished_episodes > 0:
            # --- Save Stats ---
            stats = {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "total_steps": self.num_timesteps,
                "total_episodes": self.episode_count
            }
            with open(self.stats_path, "w") as f:
                json.dump(stats, f)
                
            # --- Save Latest Model ---
            self.model.save(self.latest_model_path)

            print(f"Saved latest model to {self.latest_model_path} and stats to {self.stats_path} after processing {newly_finished_episodes} new episodes.")
            
            # --- Save Checkpoint if Triggered ---
            if last_checkpoint_episode > 0:
                checkpoint_path = self.save_dir / f"model_checkpoint_{last_checkpoint_episode}.zip"
                self.model.save(checkpoint_path)
                # if self.verbose > 0:
                print(f"Saved checkpoint for episode {last_checkpoint_episode} to {checkpoint_path}")

        return True

# ==============================================================================
# Main Training Functions
# ==============================================================================

def _make_env(pendulum_cfg: PendulumConfig, max_episode_steps: int):
    """Return a callable that creates a fresh environment instance."""

    def _init():
        # The Monitor wrapper is what captures episode rewards and lengths
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
) -> PPO | None:
    """Run PPO training and return the trained model."""
    p_cfg = pendulum_cfg or PendulumConfig()
    t_cfg = training_cfg or TrainingConfig()

    # Parallel environments
    env_fns = [_make_env(p_cfg, t_cfg.max_episode_steps) for _ in range(t_cfg.n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Eval environment (single process) for the EvalCallback
    eval_env = Monitor(
        CartPendulumEnv(
            pendulum_config=p_cfg,
            max_episode_steps=t_cfg.max_episode_steps,
        )
    )

    # Ensure save directories exist.
    # All files for one model live under save_dir (e.g. models/ppo_pendulum/).
    # Live dashboard data goes into save_dir/live/ and the final trained model
    # is saved as save_dir/final.zip.
    save_dir = Path(t_cfg.model_save_path).parent / Path(t_cfg.model_save_path).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    live_dir = save_dir / LIVE_DASHBOARD_DATA_FILENAME
    live_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup Callbacks ---
    # 1. EvalCallback saves the "best" model based on performance on a separate test env
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "logs"),
        eval_freq=max(t_cfg.n_steps // t_cfg.n_envs, 1) * 5,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    # 2. Our custom LiveDashboardCallback for the UI
    live_dashboard_callback = LiveDashboardCallback(save_dir=live_dir, save_freq=50) # Save a checkpoint every 50 episodes
    
    # Combine callbacks into a list
    callbacks = [eval_callback, live_dashboard_callback]

    if t_cfg.model_load_path:
        load_dir = Path(t_cfg.model_load_path)
        if not load_dir.is_dir():
            print(f"Error: model directory not found: {load_dir}")
            return None
        load_zip = load_dir / f"{FINAL_MODEL_FILENAME}.zip"
        if not load_zip.is_file():
            print(f"Error: trained model not found: {load_zip}")
            return None
        resolved_load_path = str(load_dir / FINAL_MODEL_FILENAME)
        print(f"Loading existing model from {load_zip} …")
        print("Note: hyperparameters (learning_rate, n_steps, etc.) are loaded "
              "from the saved model; TrainingConfig hyperparameters are ignored.")
        model = PPO.load(
            resolved_load_path,
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
        
    # Pass the list of callbacks to the learn method
    model.learn(total_timesteps=t_cfg.total_timesteps, callback=callbacks)
    
    # Save the final model inside the model directory as final.zip
    final_model_path = save_dir / FINAL_MODEL_FILENAME
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}.zip")

    vec_env.close()
    eval_env.close()
    return model


# ---- CLI entry point -------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on inverted pendulum")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel envs (default: all cores)")
    parser.add_argument("--save-path", type=str, default="models/ppo_pendulum",
                        help=f"Model directory; {FINAL_MODEL_FILENAME}.zip is written inside it (e.g. models/ppo_pendulum/{FINAL_MODEL_FILENAME}.zip)")
    parser.add_argument("--load-model", type=str, default="",
                        help=f"Model directory to resume training from (must contain {FINAL_MODEL_FILENAME}.zip)")
    parser.add_argument("--tensorboard-log", type=str, default="logs/tensorboard",
                        help="Directory for TensorBoard logs (empty string to disable)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t_cfg = TrainingConfig(
        total_timesteps=args.timesteps,
        model_save_path=args.save_path,
        model_load_path=args.load_model,
        tensorboard_log=args.tensorboard_log,
    )
    if args.n_envs is not None:
        t_cfg.n_envs = args.n_envs
    train(training_cfg=t_cfg)