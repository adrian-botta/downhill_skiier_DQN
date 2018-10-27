from brain import *
from memory import *
import math

class Skiier:
    def __init__(self, action_space, LEARNING_RATE, GAMMA, LAMBDA, MEMORY_CAPACITY, BATCH_SIZE, max_explore_game, repeat_freq):
        #Hyperparameters
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MAX_EPSILON = 1
        self.MIN_EPSILON = 0.01
        self.LAMBDA = LAMBDA
        self.repeat_freq = repeat_freq
        #Initialization
        self.action_space = action_space
        self.state_size = (125, 80, repeat_freq)
        self.brain = Brain(LEARNING_RATE, input_shape = self.state_size, model_print = False)
        self.memory = Memory(self.MEMORY_CAPACITY)
        self.epsilon = 1
        self.max_explore_game = max_explore_game
        self.episode = 0
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space, size=1)[0]
        else:
            action_values = self.brain.predict(np.expand_dims(state, axis=0))
            action = np.argmax(action_values)
            return action
            
    def observe(self, state, action, reward, next_state, done):
        observation = (state, action, reward, next_state, done)
        self.memory.add(observation)
         
    def replay(self):
        """
        Replay helps the model map the current state to the discounted 
        reward for the action taken at that state
        """
        batch = self.memory.sample(self.BATCH_SIZE)
        batch_len = len(batch)
        
        #Setting up batch to train model
        x = np.zeros((batch_len, self.state_size[0], self.state_size[1], self.state_size[2]))
        y = np.zeros((batch_len, self.action_space))
        
        #Setting up states batch
        states = np.asarray([ obs[0] for obs in batch ])
        #states = states.transpose(0,2,3,1)
        
        #Setting up future states batch
        no_state = np.zeros((1,self.state_size[0], self.state_size[1], self.state_size[2]))
        states_f = np.array([ (no_state if obs[4] is None else obs[3]) for obs in batch ])
        #states_f = states_f.transpose(0,2,3,1)
        
        #print("states shape:", states.shape)
        #old predicted Q values
        action_predictions = self.brain.predict(states)
        #new predicted Q values
        action_predictions_f = self.brain.predict(states_f)
        print("future_states",states_f.shape)
        print("action_predictions_f",action_predictions_f.shape)
        
        #Future Q values
        targets_f = []
        
        for batch_index in range(batch_len):
            state, action, reward, next_state, done = batch[batch_index]
            
            if done:
                target = reward
            else:
                action_prediction_f = action_predictions_f[batch_index]
                target = reward + self.GAMMA * np.amax(action_prediction_f) #[0]
                
            target_f = action_predictions[batch_index]
            print("target_f",target_f.shape)
            
            target_f[action] = target #[0]
            targets_f.append(target_f)
        
        #self.epsilon = (self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) 
        #                * math.exp(-self.LAMBDA * self.episode))
        if self.episode <= self.max_explore_game:
            self.epsilon = (((self.MIN_EPSILON-self.MAX_EPSILON)/self.max_explore_game)*self.episode
                            + self.MAX_EPSILON)
        else:
            self.epsilon = self.MIN_EPSILON
        self.brain.train(states, np.asarray(targets_f), epochs=1)