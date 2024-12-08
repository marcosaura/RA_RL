import math
import torch
import torch.nn as nn
import numpy as np
from core.model import BaseNet, renormalize

class RingAttractorLayer(nn.Module):
    def __init__(self, input_size, num_neurons, connectivity_strength=0.1, tau=10.0, beta=10.0):
        """
        Implementation of the ring attractor model using RNN architecture.
        Includes central inhibitory "no action" neuron as described in Section A.4.1
        of the paper.
        
        Args:
            input_size: Size of input feature vector
            num_neurons: Number of neurons in ring attractor, equal to the size of the action space
            connectivity_strength: Strength of synaptic connections
            tau: Initial time integration constant controlling temporal evolution
            beta: Initial scaling factor for preventing output saturation
        """
        super(RingAttractorLayer, self).__init__()

        self.scale = connectivity_strength
        self.lambda_decay = 0.9  # Decay parameter for distance-dependent weights
        
        # Learnable parameters from Section 3.2.1
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))  # Time constant τ
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))  # Scaling factor β
        
        # Initialize RNN layer
        self.rnn = nn.RNN(input_size, num_neurons, bias=False, batch_first=True)
        
        # Initialize tracking for model steps
        self.step_log = 0
        
        # Create connectivity matrices for ring topology
        input_to_hidden = self._create_connectivity_matrix(
            input_size-(num_neurons-1), 
            num_neurons-1, 
            connectivity_strength
        )
        hidden_to_hidden = self._create_connectivity_matrix_hidden(
            num_neurons-1, 
            connectivity_strength
        )
        
        # Add inhibitory "no action" neuron connections
        input_to_hidden = self._add_inhibitory_neuron(input_to_hidden, connectivity_strength)
        hidden_to_hidden = self._add_inhibitory_neuron(hidden_to_hidden, connectivity_strength)
        
        # Set RNN weights to implement ring topology with inhibitory neuron
        self.rnn.weight_ih_l0 = nn.Parameter(torch.Tensor(input_to_hidden), requires_grad=False) #Not trainable keeps RA structure
        self.rnn.weight_hh_l0 = nn.Parameter(torch.Tensor(hidden_to_hidden))

    def _add_inhibitory_neuron(self, arr, strength, inhib_strength=1.0):
        """
        Add inhibitory "no action" neuron connections as described in Section A.4.1.
        This neuron is positioned centrally with equal connections to all other neurons.
        
        Args:
            arr: Base connectivity matrix
            strength: Base connection strength
            inhib_strength: Strength of inhibitory connections (default=1.0 as per paper)
            
        Returns:
            Modified matrix with inhibitory neuron connections
        """
        # Create row for inhibitory neuron (equal connections to all neurons)
        inhib_row = np.full((1, arr.shape[1]), inhib_strength)
        
        # Add the inhibitory neuron row at the top of matrix
        arr = np.vstack([inhib_row, arr])
        
        # Create column for connections to inhibitory neuron
        inhib_col = np.full((arr.shape[0], 1), inhib_strength)
        arr = np.hstack([inhib_col, arr])
        
        return arr

    def _create_connectivity_matrix_hidden(self, size, strength):
        """
        Create connectivity matrix for hidden-to-hidden connections in the ring.
        Distance function follows paper's definition in Eq. 13.
        """
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                # Calculate circular distance as per paper
                distance = min(abs(i - j), size - abs(i - j))
                strength = self.scale * np.exp(-distance / self.lambda_decay)
                matrix[i, j] = self.scale * strength

        return matrix
    
    def _create_connectivity_matrix(self, input_size, output_size, strength):
        """
        Create connectivity matrix for input-to-hidden connections.
        Maintains circular topology and spatial relationships between actions.
        """
        matrix = np.zeros((output_size, input_size))
        ratio = int(input_size/output_size)
        
        for i in range(output_size):
            for j in range(input_size):
                # Calculate circular distance considering input/output size ratio
                strength = min(abs(i - int(j/ratio)), output_size - abs(i - int(j/ratio)))
                matrix[i, j] = self.scale * strength

        return matrix

    def forward(self, x):
        """
        Forward pass implementing ring attractor dynamics with inhibitory neuron.
        Implements Eq. 19 from Section 3.2.1:
        Q(s,a) = β tanh((1/τ)Φθ(st)^T V + h(v)^T U)
        
        Args:
            x: Input tensor Φθ(st) representing state
        Returns:
            ring_attractor_output: Action-values Q(s,a) after ring attractor dynamics
        """
        
        # Scale input by learnable temporal integration constant τ
        x = (1.0/self.tau) * x
        
        # Apply ring attractor dynamics
        ring_attractor_output, _ = self.rnn(x)
        
        # Scale output by learnable factor β to prevent saturation
        ring_attractor_output = self.beta * ring_attractor_output
            
        return ring_attractor_output
