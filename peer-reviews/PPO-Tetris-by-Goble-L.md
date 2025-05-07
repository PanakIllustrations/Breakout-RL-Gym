##  PPO Tetris
 by Leonora Goble


### 1. Does it include the clear overview on what the project is about? (4/4)
The main goal of the project is to test whether PPO with a simple reward function could be successfully applied to a challenging problem like Tetris.

### 2. Does it explain how the environment works and what the game rules are? (4/4)
Tetris complexity is explained, noting its NP-hard nature and delayed rewards. A Gymnasium environment is used with explanations of default actions and why this control space creates problems for RL algorithms.
### 3. Does it explain clearly the model(s) of choices, the methods and purpose of tests and experiments? (7/7)
PPO is explained as based on trust regions to prevent drastic policy updates. Actor-critic was chosen over CNN due to hardware limits. Actions are grouped into piece placements with feature vector observations. The reward function remains simple to test learning with minimal guidance.
### 4. Does it show problem solving procedure- e.g. how the author solved and improved when an algorithm doesn't work well... (7/7)
The documentation shows improvement steps: default PPO settings, increased training steps, reward clipping, grouped actions, feature vector observations, and hyperparameter tuning. Each change includes reasoning and metrics.
### 5. Does it include the results summary, interpretation of experiments and visualization (e.g. performance comparison table, graphs etc)? (5/7)
Results include metrics and videos showing agent behavior. No graphs show learning curves over time, which was noted as a limitation.
### Does it include discussion and suggestions for improvements or future work? (5)
Models didn't achieve good performance but lessons about refining algorithms are discussed. Future work includes hyperparameter experimentation, CNN policies, transfer learning strategies, and seed dependence testing.
### Does it include all deliverables (3/3)
Includes GitHub repository, README writeup, videos, and citations.
### Is the writeup well organized overall? Overall writeup quality. (3)
The writeup has a clear structure from problem overview to approach, results, conclusion, and future work, with a process log providing development insights.

## Total Score 38/40
