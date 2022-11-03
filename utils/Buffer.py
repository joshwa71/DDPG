import numpy as np

class ReplayBuffer():
    def __init__(self, memory_size, input_shape, n_actions):
        self.mem_size = memory_size
        self.mem_count = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, _state, done):
        index = self.mem_count % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = _state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.mem_count +=1

    def sample_memory(self, batch_size):
        width = min(self.mem_size, self.mem_count)
        rand_vector = np.random.choice(width, batch_size)
        states = self.state_memory[rand_vector]
        _states = self.new_state_memory[rand_vector]
        actions = self.action_memory[rand_vector]
        rewards = self.reward_memory[rand_vector]
        dones = self.done_memory[rand_vector]
        return states, actions, rewards, _states, dones
        
