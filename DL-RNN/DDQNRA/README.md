# Double Deep Q-Networks with Ring Attractor (DDQNRA)

This implementation integrates a ring attractor with the Double Deep Q-Networks (DDQN) algorithm proposed by Hado van Hasselt, Arthur Guez, and David Silver in their paper "Deep Reinforcement Learning with Double Q-learning" (AAAI 2016) [1].

The ring attractor is used to enhance action selection in the reinforcement learning agent. The integration is demonstrated using both the Super Mario Bros environment from OpenAI Gym [2] and the highway driving environment [3].

## Requirements

- Python 3.x
- PyTorch
- OpenAI Gym
- gym-super-mario-bros
- NumPy
- Matplotlib

## Running the Code

To run the DDQNRA implementation, simply execute the following command:

```bash
python3 DDQNRA.py
```

The script will train the agent using the DDQN algorithm with the ring attractor and periodically save model checkpoints and performance metrics.

## Ring Attractor Implementation

The ring attractor implementation for the DDQN agent can be found in the `RA.py` file. This standalone script provides the necessary code and explanations for integrating the ring attractor component into other research projects or reinforcement learning implementations.

## References

[1] [van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. In AAAI Conference on Artificial Intelligence.] (https://arxiv.org/abs/1509.06461)

[2] [Kauten, C. (2018). Super Mario Bros for OpenAI Gym. GitHub repository.] (https://github.com/Kautenja/gym-super-mario-bros)

[3] [Leurent, E. (2018). An environment for autonomous driving decision-making. GitHub repository.](https://github.com/eleurent/highway-env)
