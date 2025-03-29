from collections import deque
import numpy as np
from sum_tree import SumTree

class ReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        Initialize a prioritized replay buffer.

        Args:
            buffer_size: Maximum size of the buffer
            alpha: Controls how much prioritization is used (0 = no prioritization, 1 = full prioritization)
            beta: Controls importance sampling correction (0 = no correction, 1 = full correction)
            beta_increment: Amount to increase beta over time
            epsilon: Small positive constant to ensure non-zero priority
        """
        self.tree = SumTree(buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Get a random value within the segment
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            
            # Get the corresponding experience from the tree
            idx, priority, experience = self.tree.get_leaf(v)
            
            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)
        
        # Extract and reshape experiences
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        weights = (self.buffer_size * sampling_probabilities) ** -self.beta
        weights /= weights.max()  # Normalize weights
        
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones),
            indices,
            np.array(weights)
        )
    
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def size(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
