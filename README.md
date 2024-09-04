# Deep Q-Learning for Video Game Agent

This project implements a Deep Q-Learning (DQN) agent that plays a video game using computer vision techniques to process video frames. The agent learns through reinforcement learning by interacting with the game environment.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Algorithm](#algorithm)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction

This project demonstrates how a Deep Q-Learning (DQN) agent can be trained to play a video game. The DQN agent utilizes a Convolutional Neural Network (CNN) to predict the best action based on the current state of the game (represented as a sequence of stacked video frames). The game is processed frame by frame, and the agent decides whether to move left, right, or center based on the observed game state.

## Features

- Deep Q-Learning with experience replay.
- Convolutional Neural Network (CNN) for state representation.
- Real-time decision-making based on video frame analysis.
- Reward and penalty system for agent training.
- Visual representation of agent performance over episodes.

## Requirements

- Python 3.x
- NumPy
- TensorFlow
- OpenCV
- Matplotlib

## Usage

1. **Prepare the Environment:** Make sure you have all the required libraries installed.
2. **Run the Code:** Use the main function provided in the script to start training the DQN agent.
3. **Monitor the Training:** The total rewards per episode will be printed to the console and plotted at the end.

## Code Structure

- `build_model()`: Constructs the Convolutional Neural Network model for the agent.
- `DQNAgent`: Defines the Deep Q-Learning agent, including methods for action selection, memory replay, and updating the target model.
- `preprocess_frame()`: Preprocesses video frames to be fed into the model.
- `stack_frames()`: Stacks frames to create a state representation for the agent.
- `check_player_death()`: Checks if the agent 'dies' or collides with obstacles.
- `main()`: The main loop where the training of the agent occurs.

## Algorithm

1. **Initialize the Agent:** The DQN agent is initialized with a neural network and a target network.
2. **Preprocess Frames:** Convert game video frames to a format suitable for the model.
3. **Stack Frames:** Create a state by stacking multiple consecutive frames.
4. **Choose Action:** Use an epsilon-greedy strategy to balance exploration and exploitation.
5. **Perform Action and Observe Result:** Execute the action and observe the reward and the next state.
6. **Store Experience:** Save the experience in memory for experience replay.
7. **Experience Replay:** Train the model using randomly sampled experiences from memory.
8. **Update Target Network:** Periodically update the target network to stabilize training.

## Results

The agent's performance is measured by the total rewards it accumulates over a series of episodes. A graph will be generated at the end of the training showing the agent's learning progress.

## Future Improvements

- Implement a prioritized experience replay to improve training efficiency.
- Add more actions and states to make the agent more versatile.
- Use advanced techniques like Double DQN or Dueling DQN to enhance performance.

## License

This project is licensed under the MIT License.
