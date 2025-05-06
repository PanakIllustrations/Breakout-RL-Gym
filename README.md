# Breakout DQN Agent

An implementation of a Deep Q-Network (DQN) agent that learns to play Atari Breakout using reinforcement learning.

This project uses PyTorch and the Gymnasium environment to train an agent with experience replay, target network stabilization, and an epsilon-greedy exploration strategy.

## Training Results

![training_results](training_results.png?raw=true)

## Demo

| Episode 1 | Episode 2 | Episode 3 | 
|-----------|-----------|-----------|
| ![Episode 1](videos/breakout/breakout-episode-0.gif) | ![Episode 2](videos/breakout/breakout-episode-1.gif) | ![Episode 3](videos/breakout/breakout-episode-2.gif) | 

| Episode 4 | Episode 5 |
|-----------|-----------|
| ![Episode 4](videos/breakout/breakout-episode-3.gif) | ![Episode 5](videos/breakout/breakout-episode-4.gif) |

## Highlights

- **Replay Buffer**: Helps break the correlation between sequential states.
- **Target Network**: Updated every few episodes to stabilize Q-value updates.
- **CNN-Based Q-Network**: Processes stacked game frames to estimate action values.
- **Epsilon-Greedy Strategy**: Starts with full exploration and gradually shifts toward exploitation.

## Conclusion

This project implimented a Deep Q-Network agent to learn and play Breakouto on the Atari using pixel input. The goal was to evaluate the effctiveness of DQN in a classic reinforcement learning setting and understand the impact of various components like frame stacking, replay buffers, and epsilon-greedy exploration.

Throughout training, the agent showed a clear learning curve—with noticeable improvements around 450 and 750 episodes. While the resulting model likely performs worse than any manual strategy, the main takeaway is how quickly reinforcement learning can develop adaptive behavior with relatively minimal domain knowledge.
