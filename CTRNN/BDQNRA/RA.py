import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class NetworkParams:
    """Network parameters for the ring attractor.
    All parameter values are taken from:
    Sun, X., Mangan, M., & Yue, S. (2018). An Analysis of a Ring Attractor Model 
    for Cue Integration. In Biomimetic and Biohybrid Systems (Living Machines 2018), 
    pp 459-470."""
    wEEk: float = 45.0  # Excitatory-to-excitatory connection strength constant
    wIEk: float = 60.0   # Inhibitory-to-excitatory connection strength constant
    wEIk: float = -6.0   # Excitatory-to-inhibitory connection strength constant
    wIIk: float = -1.0   # Inhibitory-to-inhibitory connection strength constant
    
    exc_threshold: float = -1.5  # Base activation threshold for excitatory neurons
    inh_threshold: float = -7.5  # Base activation threshold for inhibitory neurons
    
    exc_decay: float = 0.005   # Time decay constant for excitatory neuron activity
    inh_decay: float = 0.00025 # Time decay constant for inhibitory neuron activity
    
    sigma: float = 120.0   # Connection width parameter

@dataclass
class SimulationParams:
    """Simulation parameters."""
    T: float = 0.05      # Total simulation time
    Ti: float = 0.001    # Initial stabilisation time
    tau: float = 1e-4    # Integration time step
    n_neurons: int = 20  # Number of excitatory neurons

class RingAttractor:
    def __init__(self, sim_params: Optional[SimulationParams] = None, 
                net_params: Optional[NetworkParams] = None):
        """
        Initialize Ring Attractor network.
        
        Args:
            sim_params: Optional simulation parameters. If None, uses defaults
            net_params: Optional network parameters. If None, uses defaults
        """
        # Initialize parameters with defaults if not provided
        self.sim_params = sim_params or SimulationParams()
        self.net_params = net_params or NetworkParams()
        
        # Calculate number of timesteps for total simulation and initial stabilization
        self.Nt = int(np.floor(self.sim_params.T / self.sim_params.tau))
        self.Nti = int(np.floor(self.sim_params.Ti / self.sim_params.tau))
        
        # Initialize neuron activation arrays
        self.v = np.zeros((self.sim_params.n_neurons, self.Nt))  # Excitatory neurons
        self.u = np.zeros((1, self.Nt))  # Inhibitory neuron (single central neuron)
        
        # Set initial activation state for excitatory neurons
        self.v[:, 0] = 0.05 * np.ones(self.sim_params.n_neurons)
        
        # Calculate preferred orientation angles for each excitatory neuron
        # Evenly spaced around 360 degrees
        self.alpha_n = np.linspace(0, 360 - 360/self.sim_params.n_neurons, 
                                self.sim_params.n_neurons).reshape(-1, 1)
        
        # Calculate connection weights:
        # For wEE: Full distance-dependent matrix computed using angular differences
        self.wEE = self._compute_weight_matrix()
        self.wIE = self.net_params.wIEk * np.exp(0)  # Inhibitory to excitatory weight, simplified as Inhibitory is placed in the middle of the ring.
        self.wEI = self.net_params.wEIk * np.exp(0)  # Excitatory to inhibitory weight, simplified as Inhibitory is placed in the middle of the ring.
        self.wII = self.net_params.wIIk * np.exp(0)  # Inhibitory self-connection weight, simplified as Inhibitory is placed in the middle of the ring.

    def _compute_weight_matrix(self) -> np.ndarray:
        """
        Compute wEE matrix based on neural distances.
        
        Returns:
            2D array of connection weights between excitatory neurons
        """
        # Calculate minimum angular differences between all neuron pairs
        # accounting for circular wrapping at 360 degrees
        diff_matrix = np.minimum(
            np.abs(self.alpha_n - self.alpha_n.T),
            360 - np.abs(self.alpha_n - self.alpha_n.T)
        )
        wEE = np.exp(-diff_matrix**2 / (2 * self.net_params.sigma**2))
        
        # Scale weights by kernel strength and normalize by number of neurons
        return wEE * (self.net_params.wEEk / self.sim_params.n_neurons)
    
    def generate_action_signal(self, Q: float, alpha_a: float, sigma_a: float) -> np.ndarray:
        """
        Generate action signal for ring attractor input based on action value and direction.
        
        Args:
            Q: Action value Q(s,a) - determines height of the Gaussian
            alpha_a: Action direction angle in degrees - determines center of the Gaussian
            sigma_a: Action value variance - determines width of the Gaussian
            
        Returns:
            Array of shape (n_neurons, Nt) containing the action signal input for each neuron over time
        """
        # Calculate minimum angular difference between each neuron's preferred direction
        # and the action direction, accounting for circular wrapping
        diff = np.min([np.abs(self.alpha_n - alpha_a),
                    360 - np.abs(self.alpha_n - alpha_a)], axis=0)
        
        # Generate Gaussian signal based on action parameters
        signal = (Q * np.exp(-diff**2 / (2 * sigma_a**2)) / 
                (np.sqrt(2 * np.pi) * sigma_a))
        
        # Create time-varying signal matrix
        # Signal is zero during initial stabilization period (0 to Ti)
        # then constant for the remainder of the simulation
        x = np.zeros((self.sim_params.n_neurons, self.Nt))
        x[:, self.Nti:] = np.repeat(signal.reshape(-1, 1), 
                                self.Nt - self.Nti, axis=1)
        return x
    
    def action_space_integration(self, action_values: List[Tuple[float, float, float]]) -> int:
        """
        Perform action selection following Eqs. 8-9 in paper.
        
        Args:
            action_values: List of (Q(s,a), α_a(a), σ_a) tuples for each action
            
        Returns:
            Selected action index based on neural activity
        """
        # Generate all action signals
        input_signals = [self.generate_action_signal(Q, alpha_a, sigma_a) 
                        for Q, alpha_a, sigma_a in action_values]
        
        # Run integration
        for t in range(1, self.Nt):
            # Sum all action signals
            total_input = sum(signal[:, t-1] for signal in input_signals)
            
            # Update excitatory neurons
            network_input = (self.net_params.exc_threshold + 
                           np.dot(self.wEE, self.v[:, t-1]) + 
                           self.wEI * self.u[:, t-1] + 
                           total_input)
            
            self.v[:, t] = self.v[:, t-1] + (-self.v[:, t-1] + 
                np.maximum(0, network_input)) * self.sim_params.tau / self.net_params.exc_decay
            
            # Update inhibitory neuron
            inhibitory_input = (self.net_params.inh_threshold + 
                              self.wIE * np.sum(self.v[:, t-1]) / self.sim_params.n_neurons + 
                              self.wII * self.u[:, t-1])
            
            self.u[:, t] = self.u[:, t-1] + (-self.u[:, t-1] + 
                np.maximum(0, inhibitory_input)) * self.sim_params.tau / self.net_params.inh_decay

        # Convert neural activity to action selection
        max_neuron = np.argmax(self.v[:, -1])
        action_idx = int(max_neuron * len(action_values) / self.sim_params.n_neurons)
        return action_idx

def main():
    """Example usage with action selection."""
    # Example action values: (Q(s,a), α_a(a), σ_a)
    action_values = [
        (40, 0, 40),    # Action 0 
        (30, 45, 40),   # Action 1
        (20, 90, 40),   # Action 2
        (10, 135, 40),  # Action 3
        (15, 180, 40),  # Action 4
        (25, 225, 40),  # Action 5
        (35, 270, 40),  # Action 6
        (45, 315, 40)   # Action 7
    ]
    
    ra = RingAttractor()
    selected_action = ra.action_space_integration(action_values)
    print(f"Selected action: {selected_action}")

if __name__ == '__main__':
    main()