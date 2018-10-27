import numpy as np
from collections import deque
class Env_wrapper:
    """
    This class is a wrapper around the AI Gym environment variable that reduces the dimension of the state to
    reduce memory usage
    """
    def __init__(self, env, repeat_freq):
        """
        Initialize the wrapper
        
        Args:
            env (environment variable): the AI Gym environment variable to be wrapped
        """
        self.env = env
        self.repeat_freq = repeat_freq
        self._state_buffer = deque(maxlen=self.repeat_freq)

    def reset(self):
        """
        Resets the game and returns a processed version of the state
        """
        state = self.env.reset()
        for s in range(self.repeat_freq):
            self._state_buffer.append(np.squeeze(self.process_state(state)))
        return np.stack(self._state_buffer,axis=2)
    
    def step(self, action):
        """
        Applies an action given the current state of the environment and returns the processed version of the
        state, the reward for that action, whether the game is finished or not, and supplementary information
        on the game.
        
        Args:
            action (int): value between 0 and 2 for the 3 actions in the environment action space
        """
        total_reward = 0.0
        for s in range(self.repeat_freq):
            state, reward, done, info = self.env.step(action)
            self._state_buffer.append(np.squeeze(self.process_state(state)))
            total_reward += reward#self.process_reward(reward)
            if done:
                break
        #in the case that the game is done before the full repeat freq, fill the matrix with the last state
        if s+1 < self.repeat_freq:
            for filler in range(self.repeat_freq-(s+1)):
                self._state_buffer.append(np.squeeze(self.process_state(state)))

        return np.stack(self._state_buffer,axis=2), total_reward, done, info
        
    def process_state(self, state):
        """
        Reduces the dimensions of the original atari image state by turning the 
        image into greyscale and downsampling
        
        Args:
            state (np.matrix): the original state containing pixels
        """
        grey_state = np.mean(state, axis=2).astype(np.uint8) #Greyscale
        down_state = grey_state[::2, ::2] #Downsample
        scale_state = (down_state - np.max(down_state))/(np.max(down_state)-np.min(down_state))
        out_state = np.expand_dims(scale_state, axis=2)
        out_state = np.expand_dims(down_state, axis=2)
        return out_state
    
    def process_reward(self, reward):
        return np.sign(reward)