# [Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](https://openreview.net/pdf?id=E5ulvtj86q)

This repository contains the implementation of a novel approach to reinforcement learning (RL) that integrates ring attractors into the action selection process. The project aims to improve learning speed and performance, especially in spatially structured tasks.

## Project Overview

Our approach incorporates ring attractors, a mathematical model inspired by neural circuit dynamics, into RL algorithms. We provide two main implementations:

1. Exogenous Ring Attractor Model (CTRNN-based)
2. Deep Learning Ring Attractor Model (DL-RNN-based)

Both models are designed to enhance RL performance in various environments.

Each implementation folder contains a file named`RA.py` that provides the implementation details for the ring attractor component in isolation. These files can be easily reused and integrated into other research projects or implementations.
Researchers and developers interested in leveraging the ring attractor component for their own work can find the necessary code and explanations in the respective RA.py files. They can also follow the implementation examples to understand how to integrate the ring attractor module into an RL agent.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
Alternatively you can run setup.py by:

```bash
pip install .
```

## Usage

Instructions on how to run each implementation can be found in the respective implementations:

- For the Exogenous Ring Attractor Model, refer to the README in the `CTRNN` folder.
- For the Deep Learning Ring Attractor Model, refer to the README in the `DL-RNN` folder.

Each folder contains specific instructions, examples, and any additional setup required for running the models.

## Project Structure

```
RA_RL/
│
├── CTRNN
│   └── BDQNRA
│       ├── BDQNRA.py
│       └── RA.py
        └── README.py
├── DL-RNN
│   ├── DDQNRA
│   │   ├── DDQN.py
│   │   └── RA.py
|   |   └── README.py
│   └── EfficientZeroRA
│       ├── RA.py
|       ├── RA_Double.py
│       ├── main.py
|       ├── README.py
|
├── requirements.txt
├── setup.py
└── README.md
```

## Contributing

We welcome contributions to this project. Please feel free to submit pull requests or open issues to discuss potential improvements or report bugs.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ring-attractor-rl-2025,
  title={Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems},
  author={Anonymous},
  journal={Under Review},
  year={2024}
}
```

## Contact

For any questions or concerns, please open an issue in this repository.
