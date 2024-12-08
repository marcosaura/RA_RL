import torch
import torch.nn as nn
import numpy as np

class RingAttractor(nn.Module):
    def __init__(self, input_dim, num_excitatory, tau=10.0, beta=10, lambda_decay=0.9):
        """
        Implementation of the ring attractor model as described in Section 3.2.
        The model uses a RNN to maintain stable spatial representations through 
        circular connectivity patterns.
        
        Args:
            input_dim: Dimension of input features Φθ(s)
            num_excitatory: Number of excitatory neurons arranged in ring topology
            tau: Initial time integration constant controlling temporal evolution 
            beta: Initial scaling factor for preventing tanh saturation
            lambda_decay: Decay parameter for distance-dependent weights
        """
        super(RingAttractor, self).__init__()
        self.lambda_decay = lambda_decay  # Controls decay of potential over distance in ring
        self.scale = 0.000025  # Scaling factor for weight initialization
        
        # Learnable parameters from Section 3.2.1
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))  # Time constant τ for signal integration
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))  # Scaling factor β for action-values
        
        # RNN implementation of ring attractor dynamics
        self.rnn = nn.RNN(input_dim, num_excitatory, bias=False)
        
        # V(s): Fixed input-to-hidden connections preserving ring topology (Eq. 13)
        fixed_weights = torch.Tensor(self._create_distance_weights(num_excitatory))
        self.rnn.weight_ih_l0 = nn.Parameter(fixed_weights, requires_grad=False)
        
        # U(v): Learnable hidden-to-hidden connections for action relationships (Eq. 13)
        self.rnn.weight_hh_l0 = nn.Parameter(torch.Tensor(self._create_distance_weights(num_excitatory)))
        
        print("Input weights V(s) (fixed):")
        print(self.rnn.weight_ih_l0.data)
        print("\nHidden weights U(v) (learnable):")
        print(self.rnn.weight_hh_l0.data)

    def _create_distance_weights(self, N):
        """
        Implements the distance-dependent weight function from Eq. 13:
        w_{m,n} = e^{-d(m,n)/λ}
        where d(m,n) is the circular distance between neurons m and n
        """
        weights = np.zeros((N, N))
        for m in range(N):
            for n in range(N):
                # Calculate circular distance d(m,n)
                d_mn = min(abs(m - n), N - abs(m - n))
                # Apply exponential decay based on distance
                weights[m,n] = self.scale * np.exp(-d_mn / self.lambda_decay)
        return weights

    def forward(self, x):
        """
        Implements Eq. 19 from Section 3.2.1:
        Q(s,a) = β tanh((1/τ)Φθ(st)^T V + h(v)^T U)
        
        Args:
            x: Input features Φθ(st) representing state
        Returns:
            action_value_pairs: Q(s,a) for all actions in ring topology
        """
        # Scale input by learnable temporal integration constant τ
        x = (1.0/self.tau) * x
        output, _ = self.rnn(x)
        action_value_pairs = self.beta * output
        return action_value_pairs