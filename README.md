# Inverted Pendulum RL

A reinforcement learning project that trains a PPO agent to balance an inverted pendulum on a cart. The agent is trained using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and the physics are simulated with [MuJoCo](https://mujoco.org/) via [Gymnasium](https://gymnasium.farama.org/). A real-time [Pygame](https://www.pygame.org/) visualizer lets you watch the trained agent — or interact with the cart yourself by clicking and dragging it.

## Features

- **PPO training** across all available CPU cores using parallel environments
- **MuJoCo physics** for accurate rigid-body simulation
- **Pygame visualizer** with real-time rendering and mouse interaction
- **Configurable** physics, training hyperparameters, and display settings via `pendulum/config.py`

## Requirements

- Python 3.10+
- A system capable of running MuJoCo (Linux, macOS, or Windows)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Naviary2/inverted-pendulum-rl.git
   cd inverted-pendulum-rl
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the agent

```bash
python -m pendulum.train
```

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | `500000` | Total training timesteps |
| `--n-envs` | all CPU cores | Number of parallel environments |
| `--save-path` | `models/ppo_pendulum` | Where to save the trained model |

Example:

```bash
python -m pendulum.train --timesteps 1000000 --n-envs 8
```

### Visualize (random / no model)

```bash
python -m pendulum.visualize
```

### Visualize a trained model

```bash
python -m pendulum.visualize --model models/ppo_pendulum
```

While the visualizer is running:

- **R** — reset the environment
- **Click + drag** the cart to interact with the simulation manually

## Project Structure

```
inverted-pendulum-rl/
├── assets/
│   └── inverted_pendulum.xml   # MuJoCo scene definition
├── pendulum/
│   ├── config.py               # Physics, training & display configuration
│   ├── environment.py          # Gymnasium environment wrapper
│   ├── interaction.py          # Mouse drag controller
│   ├── train.py                # PPO training script
│   └── visualize.py            # Pygame visualizer
└── requirements.txt
```

## Configuration

All tunable parameters live in `pendulum/config.py`:

- **`PendulumConfig`** — cart/pendulum physics (track length, link lengths, gravity, FPS)
- **`TrainingConfig`** — PPO hyperparameters (timesteps, learning rate, batch size, etc.)
- **`VisualizationConfig`** — display settings (resolution, colours, scale)
