import torch
import torch.nn as nn
import numpy as np

class DoubleRingAttractorLayer(nn.Module):
    def __init__(self, input_size, output_size, connectivity_strength=0.1, tau=10.0, beta=10.0):
        """
        Implementation of a double ring attractor model using RNN architecture.
        Features two coupled ring attractors with cross-connections and inhibitory neurons.
        
        Args:
            input_size: Size of input feature vector
            output_size: Number of neurons in each ring attractor reflecting the number of actions in the action space
            connectivity_strength: Strength of synaptic connections
            tau: Forward pass input integration constant controlling the addition of new information to the RNN layer
            beta: Scaling factor, preventing output saturation for action values
        """
        
        self.connectivity_strength = connectivity_strength
        self.lambda_decay = 0.9  # Decay parameter for distance-dependent weights
        self.cross_coupling_factor_K = 0.1  # Strength of coupling between rings
        
        # Learnable parameters
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        
        # Initialize RNN for double ring
        self.rnn = nn.RNN(input_size*2, output_size*2, bias=False, batch_first=True)
        
        # Create base connectivity matrices for each ring
        input_to_hidden = self._create_input_connectivity(
            input_size-(output_size-1), 
            output_size-1, 
            connectivity_strength
        )
        hidden_to_hidden = self._create_hidden_connectivity(
            output_size-1, 
            connectivity_strength
        )
        
        # Add inhibitory neurons to base matrices
        input_to_hidden_single = self._add_inhibitory_neuron(
            input_to_hidden, 
            int(input_size/output_size), 
            connectivity_strength
        )
        hidden_to_hidden_single = self._add_inhibitory_neuron(
            hidden_to_hidden, 
            1, 
            connectivity_strength
        )
        
        # Create double ring matrices with cross-connections
        input_to_hidden_double = self._create_double_ring_matrix(
            input_to_hidden_single, 
            input_size, 
            output_size
        )
        hidden_to_hidden_double = self._create_double_ring_matrix(
            hidden_to_hidden_single, 
            output_size, 
            output_size
        )
        
        # Set RNN weights
        self.rnn.weight_ih_l0 = nn.Parameter(torch.Tensor(input_to_hidden_double), requires_grad=False) #Not trainable keeps RA structure
        self.rnn.weight_hh_l0 = nn.Parameter(torch.Tensor(hidden_to_hidden_double))
        
    def _create_double_ring_matrix(self, single_matrix, dim1, dim2):
        """
        Creates a double ring matrix with cross-connections from a single ring matrix.
        
        Args:
            single_matrix: Base connectivity matrix for a single ring
            dim1, dim2: Dimensions for the double matrix
            
        Returns:
            Double ring matrix with cross-connections
        """
        double_matrix = np.zeros((dim2*2, dim1*2))
        
        # Main diagonal blocks (primary connections within each ring)
        double_matrix[0:dim2, 0:dim1] = single_matrix
        double_matrix[dim2:dim2*2, dim1:dim1*2] = single_matrix
        
        # Off-diagonal blocks (cross-connections between rings)
        double_matrix[0:dim2, dim1:dim1*2] = single_matrix * self.cross_coupling_factor_K
        double_matrix[dim2:dim2*2, 0:dim1] = single_matrix * self.cross_coupling_factor_K
        
        return double_matrix
    
    def _create_input_connectivity(self, input_size, output_size, strength):
        """
        Creates connectivity matrix for input-to-hidden connections maintaining
        circular topology.
        """
        matrix = np.zeros((output_size, input_size))
        ratio = int(input_size/output_size)
        
        for i in range(output_size):
            for j in range(input_size):
                # Calculate circular distance considering input/output size ratio
                distance = min(abs(i - int(j/ratio)), output_size - abs(i - int(j/ratio)))
                matrix[i, j] = self.connectivity_strength * np.exp(-distance / self.lambda_decay)

        return matrix
    
    def _create_hidden_connectivity(self, size, strength):
        """
        Creates connectivity matrix for hidden-to-hidden connections within each ring.
        """
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                # Calculate circular distance as per paper
                distance = min(abs(i - j), size - abs(i - j))
                matrix[i, j] = self.connectivity_strength * np.exp(-distance / self.lambda_decay)

        return matrix
    
    def _add_inhibitory_neuron(self, arr, n, value):
        """
        Adds inhibitory "no action" neuron connections to the connectivity matrix.
        
        Args:
            arr: Base connectivity matrix
            n: Number of columns to add
            value: Connection strength for inhibitory neuron
        """
        inhibitory_row = np.full((1, arr.shape[1]), value)
        arr = np.vstack([inhibitory_row, arr])
        
        inhibitory_cols = np.full((arr.shape[0], n), value)
        arr = np.hstack([inhibitory_cols, arr])
        
        return arr
    
    def forward(self, x):
        """
        Forward pass implementing double ring attractor dynamics.
        
        Args:
            x: Input tensor representing state
        Returns:
            Output after ring attractor dynamics
        """
        x = (1.0/self.tau) * x
        ring_output, _ = self.rnn(x)
        return self.beta * ring_output