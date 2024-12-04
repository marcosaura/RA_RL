# Spatial-aware decision-making with Ring Attractors in Reinforcement Learning systems

This repository contains the implementation of a novel approach to reinforcement learning (RL) that integrates ring attractors into the action selection process. The project aims to improve learning speed and performance, especially in spatially structured tasks.

## Project Overview

Our approach incorporates ring attractors, a mathematical model inspired by neural circuit dynamics, into RL algorithms. We provide two main implementations:

1. Exogenous Ring Attractor Model
2. Deep Learning Ring Attractor Model

Both models are designed to enhance RL performance in various environments.

## Key Features

- Integration of ring attractors into RL action selection
- Uncertainty-aware decision making
- Adaptable to various RL tasks and environments
- Improved spatial awareness in action selection

## Main Components

### 1. Exogenous Ring Attractor Model

- Implemented using Continuous-Time Neural Networks (CTRNN)
- Integrated with Bayesian Deep Q-Networks (BDQN)
- Includes uncertainty quantification through Bayesian Linear Regression

### 2. Deep Learning Ring Attractor Model

- Implemented using Recurrent Neural Networks (RNNs)
- Easily integrable with existing Deep RL architectures
- Provides end-to-end training capabilities

## Project Structure

```
ring-attractor-rl/
│
├── exogenous_model/
│   ├── snn_implementation.py
│   ├── bdqn_integration.py
│   └── uncertainty_quantification.py
│
├── deep_learning_model/
│   ├── rnn_implementation.py
│   └── rl_integration.py
│
├── utils/
│   ├── data_processing.py
│   └── visualization.py
│
├── experiments/
│   ├── atari_benchmark.py
│   └── custom_environments.py
│
├── tests/
│   ├── test_exogenous_model.py
│   └── test_deep_learning_model.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

(Placeholder for usage instructions and examples)

## Contributing

We welcome contributions to this project. Please feel free to submit pull requests or open issues to discuss potential improvements or report bugs.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite our paper:

(Placeholder for citation information)

## Contact

For any questions or concerns, please open an issue in this repository.
