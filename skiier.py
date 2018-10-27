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
        current_states = np.asarray([ obs[0] for obs in batch ])
        #states = states.transpose(0,2,3,1)
        
        #Setting up future states batch
        no_state = np.zeros((1,self.state_size[0], self.state_size[1], self.state_size[2]))
        next_states = np.array([ (no_state if obs[4] is None else obs[3]) for obs in batch ])
        #states_f = states_f.transpose(0,2,3,1)
        
        #print("states shape:", states.shape)
        #old predicted Q values: action_predictions
        predicted_Q_values_of_current_states = self.brain.predict(current_states)
        predicted_Q_values_of_next_states = self.brain.predict(next_states)

        updated_Q_values_of_current_states = []
        
        for batch_index in range(batch_len):
            state, action_at_current, reward_at_current, next_state, done = batch[batch_index]
            if done:
                updated_Q_value_of_current_state = reward_at_current
            else:
                predicted_Q_values_of_next_state = predicted_Q_values_of_next_states[batch_index]
                updated_Q_value_of_current_state = (reward_at_current + self.GAMMA *
                                                   np.amax(predicted_Q_values_of_next_state)) #[0]
                
            predicted_Q_values_of_current_state = predicted_Q_values_of_current_states[batch_index] #target_f
            predicted_Q_values_of_current_state[action_at_current] = updated_Q_value_of_current_state #[0]
            updated_Q_values_of_current_states.append(predicted_Q_values_of_current_state)
        
        #EXPONENTIAL EPSILON DECAY
        #self.epsilon = (self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) 
        #                * math.exp(-self.LAMBDA * self.episode))
        
        #LINEAR EPSILON DECAY
        if self.episode <= self.max_explore_game:
            self.epsilon = (((self.MIN_EPSILON-self.MAX_EPSILON)/self.max_explore_game)*self.episode
                            + self.MAX_EPSILON)
        else:
            self.epsilon = self.MIN_EPSILON
        self.brain.train(current_states, np.asarray(updated_Q_values_of_current_states), epochs=1)