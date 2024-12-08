# Efficient Zero with Ring Attractor (EffZeroRA)

**Important Note:** The majority of the files in this repository, except for `RA.py` and `RA_Double.py`, are owned by the original authors of the Efficient Zero algorithm. You can find their offical implementation [here](https://github.com/YeWR/EfficientZero).

This implementation integrates a ring attractor with the Efficient Zero algorithm proposed by Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, and Yang Gao in their paper "Mastering Atari Games with Limited Data" (NeurIPS 2021) [1].

The ring attractor is used to enhance action selection in the reinforcement learning agent. The integration is demonstrated on a subset of Atari games from the Arcade Learning Environment [2].

## Requirements
- Python 3.x
- PyTorch
- Atari Py
- NumPy
- Ray
- Gym
- Cython
- TensorBoard
- Kornia

You can install the required dependencies by running:
```bash
pip install -r requirements.txt
```

## Running the Code
To run the EffZeroRA implementation, use the following command:
```bash
python3 main.py --env <env_name> --seed <seed_value> --case atari --opr <operation> --amp_type torch_amp --use_max_priority --use_priority
```

Replace `<env_name>` with the desired Atari environment (e.g., `MsPacmanNoFrameskip-v4`), `<seed_value>` with the random seed, and `<operation>` with either `train` or `test`.

For example, to train the agent on the Ms. Pac-Man environment with seed 0, run:
```bash
python3 main.py --env MsPacmanNoFrameskip-v4 --seed 0 --case atari --opr train --amp_type torch_amp --use_max_priority --use_priority
```

The script will train the agent using the Efficient Zero algorithm with the ring attractor and save the trained model checkpoints.

## Ring Attractor Implementation

The ring attractor implementations for the Efficient Zero agent can be found in `RA.py` and `RA_Double.py`. These modules provide two variants:

### Single Ring Attractor (`RA.py`)
A neural circuit that creates a continuous action space representation through a circular topology of neurons. It includes an inhibitory mechanism for action selection and temporal dynamics for stable decision-making.

### Double Ring Attractor (`RA_Double.py`)
An enhanced version featuring two coupled ring attractors that work in parallel. The dual ring structure enables more complex action patterns and provides increased robustness in decision-making through redundancy and cross-communication between rings.

## References
[1] [Ye, W., Liu, S., Kurutach, T., Abbeel, P., & Gao, Y. (2021). Mastering Atari Games with Limited Data. In Advances in Neural Information Processing Systems (NeurIPS).](https://arxiv.org/abs/2111.00210)

[2] [Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade Learning Environment: An Evaluation Platform for General Agents. Journal of Artificial Intelligence Research, 47, 253-279.](https://arxiv.org/abs/1207.4708)