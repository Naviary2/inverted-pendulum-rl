# Inverted Pendulum RL

A reinforcement learning project that trains a PPO agent to balance an inverted pendulum on a cart. The agent is trained using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and the physics are simulated with [MuJoCo](https://mujoco.org/) via [Gymnasium](https://gymnasium.farama.org/). A real-time [Pygame](https://www.pygame.org/) visualizer lets you watch the trained agent — or interact with the cart yourself by clicking and dragging it.

## Requirements

- Python 3.11+

## Installation

2. **Create and activate a virtual environment**

   ```bash
   python3.11 -m venv .venv
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

| Flag           | Default               | Description                                           |
| -------------- | --------------------- | ----------------------------------------------------- |
| `--timesteps`  | `500000`              | Total training timesteps                              |
| `--n-envs`     | all CPU cores         | Number of parallel environments                       |
| `--save-path`  | `models/ppo_pendulum` | Where to save the trained model                       |
| `--load-model` | None                  | Path to a pre-trained model to continue training from |

### Visualize (no model)

```bash
python -m pendulum.visualize
```

### Visualize a trained model

```bash
python -m pendulum.visualize --model models/ppo_pendulum.zip
```

While the visualizer is running:

- **R** — reset the environment
- **G** — toggle cart lock (free fall pendulums)
- **Click + drag** the cart to interact with the simulation manually

## Configuration

All tunable parameters live in `pendulum/config.py`:

- **`PendulumConfig`** — cart/pendulum physics (track length, link lengths, gravity, FPS)
- **`TrainingConfig`** — PPO hyperparameters (timesteps, learning rate, batch size, etc.)
- **`VisualizationConfig`** — display settings (resolution, colours, scale)
