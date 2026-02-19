"""Smoke-test: PPO training runs for a few steps without error."""

import pytest
from pendulum.config import PendulumConfig, TrainingConfig
from pendulum.train import train


def test_training_smoke(tmp_path):
    """Very short training run to verify the pipeline works end-to-end."""
    t_cfg = TrainingConfig(
        total_timesteps=512,
        n_envs=2,
        n_steps=64,
        batch_size=32,
        model_save_path=str(tmp_path / "test_model"),
    )
    model = train(training_cfg=t_cfg)
    assert model is not None
    # Model file should exist
    assert (tmp_path / "test_model.zip").exists()
