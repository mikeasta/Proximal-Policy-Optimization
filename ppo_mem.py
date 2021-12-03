import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.states  = []
        self.probs   = []
        self.values  = []
        self.actions = []
        self.rewards = []
        self.dones   = []

        self.batch_size = batch_size

    
    def generate_batches(self):
        n_states    = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices     = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches     = [indices[i : i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
                    np.array(self.actions),\
                    np.array(self.probs),\
                    np.array(self.values),\
                    np.array(self.rewards),\
                    np.array(self.dones),\
                    batches

    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states  = []
        self.probs   = []
        self.values  = []
        self.actions = []
        self.rewards = []
        self.dones   = []
