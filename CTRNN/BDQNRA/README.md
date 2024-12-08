# Bayesian Deep Q-Networks with Ring Attractor (BDQNRA)

This implementation integrates an exogenous ring attractor with the Bayesian Deep Q-Networks (BDQN) algorithm proposed by Kamyar Azizzadenesheli and Animashree Anandkumar in their paper "Efficient Exploration through Bayesian Deep Q-Networks" (ICLR 2018) [1].

The ring attractor is used to enhance action selection in the reinforcement learning agent. The integration is demonstrated using the Super Mario Bros environment from OpenAI Gym [2].

## Requirements

- Python 3.x
- OpenAI Gym
- gym-super-mario-bros
- MXNet
- NumPy
- Matplotlib

## Running the Code

To run the BDQNRA implementation, simply execute the following command:

```bash
python3 BDQNRA.py
```

The script will train the agent using the BDQN algorithm with the exogenous ring attractor and periodically save model checkpoints and performance metrics.

## References

[1] [Azizzadenesheli, K., & Anandkumar, A. (2018). Efficient Exploration through Bayesian Deep Q-Networks. In International Conference on Learning Representations (ICLR).](https://arxiv.org/abs/1802.04412)

[2] [Kauten, C. (2018). Super Mario Bros for OpenAI Gym. GitHub repository.](https://github.com/Kautenja/gym-super-mario-bros)
