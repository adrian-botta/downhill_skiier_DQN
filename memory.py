import numpy as np
class Memory():
    """
    This class implements a replay memory that keeps certain number of observations in the data.
    When it exceeds the capacity, it removes old memory in FIFO manner.
    """
    def __init__(self, capacity):
        """
        Initialize the Memory
        
        Args:
            capacity (int): the size of the memory. e.g. buffer or max capacity to hold memory
        """
        self.capacity = capacity
        self.__memory = []
        
    def add(self, observation):
        """
        Add a single observation to the memory
        
        Args:
            observation (list): a single set of (old_state, action, reward, new_state, is_terminal)
        """
        self.__memory.append(observation)
        self.__memory = self.__memory[-self.capacity:] # Keep the latest memory within the given capacity
        self.mem_size = len(self.__memory)
        
    def sample(self, sample_size):
        """
        Randomly sample from the memory
        
        Args:
            sample_size (int): a batch size to retrieve from the memory
        
        Returns:
            nparray: a numpy array of randomly selected memory
        """
        self.mem_size = len(self.__memory)
        mem_inds = np.asarray(range(self.mem_size))
        sample_inds = np.random.choice(mem_inds, sample_size)
        return np.asarray(self.__memory)[sample_inds]
    
    def is_full(self):
        """
        Check whether the memory is full of not
        
        Returns:
            bool: whether the memory is full according to the capacity or not
        """
        return len(self.__memory) >= self.capacity