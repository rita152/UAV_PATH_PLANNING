# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a multi-agent reinforcement learning (MARL) project for UAV path planning in aerial combat scenarios. The system trains UAVs to navigate from start to goal while avoiding obstacles and maintaining formation using deep RL algorithms.

## Project Structure

- **rl_env/path_env.py**: Core Gym environment (`RlGame`) that simulates the aerial combat scenario with leader-follower formation
- **main_DDPG.py**: Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation
- **main_SAC.py**: Multi-Agent Soft Actor-Critic (MASAC) implementation
- **main.py**: Random policy baseline for comparison
- **assignment/**: Pygame-based visualization and simulation components
  - **components/player.py**: `Hero` (leader UAV), `Enemy` (follower UAV), `Obstacle`, and `Goal` sprite classes
  - **constants.py**: Environment constants (screen size, FPS, colors, etc.)
- **plot.py**: Visualization utilities for training curves
- **OU_test.py**: Ornstein-Uhlenbeck noise testing

## Key Architecture Details

### State Space (7-dimensional per agent)
- Position (x, y) normalized by 1000
- Speed normalized by 30
- Heading angle (theta) in radians normalized by 360/57.3
- Target position (x, y) - for leaders: goal position; for followers: leader position
- Obstacle collision flag (0 or 1)

### Action Space (2-dimensional continuous)
- Acceleration: [-1, 1]
- Turning rate (phi): [-1, 1]

### Agent Dynamics
- **Leader (Hero)**: Speed range [10, 20], acceleration factor 0.3, turn rate factor 0.6, dt=1
- **Follower (Enemy)**: Speed range [10, 40], acceleration factor 0.6, turn rate factor 1.2, dt=1
- Position updates: `x += speed * cos(theta) * dt`, `y -= speed * sin(theta) * dt`

### Reward Structure
- **Leader**: Goal proximity (-0.001*distance), goal reaching (+1000), obstacle collision (-500 if hit, -2 if within 40), boundary violation (-1), formation keeping (0 if follower within 50 units, -0.001*distance otherwise), speed matching (+1)
- **Follower**: Formation keeping (-0.001*distance to leader), speed matching (+1)

### Network Architecture
- **DDPG**: Actor [state_dim → 50 → 20 → action_dim], Critic [state+action → 40 → 20 → 1]
- **SAC**: Actor [state_dim → 256 → 256 → mean/std], Critic (twin Q) [state+action → 256 → 256 → 1]

### Multi-Agent Setup
The environment uses centralized training with decentralized execution:
- Each agent has its own Actor-Critic networks
- Global state includes all agents' states flattened for critic input
- Agents are trained with separate replay buffers but shared experiences

## Running the Code

### Training
```bash
# MADDPG training
python main_DDPG.py  # Set Switch=0 at line 29

# MASAC training
python main_SAC.py   # Set Switch=0 at line 32
```

### Testing
```bash
# MADDPG testing
python main_DDPG.py  # Set Switch=1 at line 29, RENDER=True at line 14

# MASAC testing
python main_SAC.py   # Set Switch=1 at line 32, RENDER=True at line 15

# Random baseline
python main.py       # Set RENDER=True at line 13
```

### Key Configuration Parameters
- `N_Agent`: Number of leader UAVs (typically 1)
- `M_Enemy`: Number of follower UAVs (must be 1 for SAC, see warning at main_SAC.py:200-204)
- `RENDER`: Enable Pygame visualization
- `EP_MAX`/`EPIOSDE_ALL`: Training episodes (500)
- `EP_LEN`: Max steps per episode (1000)
- `TEST_EPIOSDE`: Number of test episodes (100)
- `MemoryCapacity`: Replay buffer size (20000)
- `TRAIN_NUM`: Number of independent training runs for averaging

### Model Checkpoints
Models are saved every 20-50 episodes after episode 200:
- DDPG: `Path_DDPG_actor_new.pth`, `Path_DDPG_actor_1_new.pth`
- SAC: `Path_SAC_actor_L1.pth`, `Path_SAC_actor_F1.pth`

### Output Files
Training statistics are saved as pickled dictionaries:
- `shoplistfile`: Mean/std of episode rewards during training
- `shoplistfile_test`: Test episode metrics (V, U, T, FKR)

## Important Notes

- **Hardcoded paths**: All file paths are absolute (e.g., `/home/zp/vscode_projects/path planning/`). Update these when working in different environments.
- **Multi-follower limitation**: SAC implementation currently only supports `M_Enemy=1`. To use multiple followers, modifications are needed in `path_env.py` (see warning at main_SAC.py:200-204).
- **Noise schedule**: OU noise is applied for first 20-50 episodes, then disabled.
- **dt parameter**: Set to 1 in player.py:9. Each step represents 1 second of simulation time.
