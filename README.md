# Deep Q-Network (DQN) with Video Frame Processing

This project implements a Deep Q-Network (DQN) agent to learn actions based on video frames. The network utilizes Convolutional Neural Networks (CNNs) to extract features from video frames and predict actions in a reinforcement learning setting.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Structure](#model-structure)
- [Agent Details](#agent-details)
- [Training Process](#training-process)
- [Results](#results)
- [License](#license)

## Introduction

In this project, a DQN agent is trained to predict actions (e.g., moving left, right, or staying still) based on consecutive video frames. The video frames are processed and stacked to provide temporal context for action prediction. The agent uses two neural networks: one for estimating Q-values and another for stabilizing training (target network).

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install tensorflow opencv-python matplotlib
```

## Usage
### Step 1: Prepare Video Input
- Place the video file you want to use for training in the visual/ directory and name it video.mp4. You can modify the video path in the script if needed.

### Step 2: Run the Training
- You can run the training script by executing:

```bash
python dqn_video.py
```

This will start the DQN agent training on the given video file. The agent will learn to predict actions based on the frames extracted from the video.

### Step 3: Visualize Results
- Once training is completed, a plot will be displayed showing the total reward over episodes.

## Model Structure
The neural network used in this DQN implementation consists of:

- Conv2D layers: Extract features from video frames using convolutional filters.
- Flatten layer: Flatten the output of the convolutional layers.
- Dense layer: Learn patterns from the flattened feature map.
- Output layer: Predict Q-values for each possible action.
- The architecture is designed to process a stack of 4 consecutive frames to capture temporal dependencies.

### Agent Details
The DQNAgent class defines the agent behavior:

- Memory: Stores past experiences (state, action, reward, next_state, done) for experience replay.
- Epsilon-Greedy Policy: Explores random actions initially, then gradually shifts to exploiting learned knowledge by decaying epsilon over time.
- Target Model: Provides stability by being updated less frequently than the main model.
- Key hyperparameters include:

- gamma (discount factor): 0.99
- epsilon (exploration rate): Starts at 1.0 and decays to 0.1
- epsilon_decay: 0.995
- learning_rate: 0.00025

### Training Process
- Episodes: The training process consists of several episodes, during which the agent interacts with the environment (video frames) and learns through reward maximization.
- Experience Replay: The agent replays batches of stored experiences to train the model, which helps break correlations and stabilize training.
- Target Network Updates: The weights of the target network are periodically updated to those of the main model to stabilize learning.

### Preprocessing
- Video frames are resized to 84x84 and normalized to values between 0 and 1. Four frames are stacked along the depth axis to create a state representation.

### Results
- After training, the program generates a plot showing the total reward per episode. The training process can be further tuned by adjusting hyperparameters such as the number of episodes, batch size, or learning rate.


## License
This project is licensed under the MIT License. See the LICENSE file for more details.


This `README.md` includes instructions for installation, usage, and an overview of the project components. Let me know if you'd like any modifications!
