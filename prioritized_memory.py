import random
import numpy as np
from SumTree import SumTree

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

if __name__=='__main__':
    memory = Memory(10)
    memory.add(3,(1,1,1,1))
    memory.add(2,(0,4,1,1))
    memory.add(10,(1,3,4,2))

    batch, idxs, is_weight = memory.sample(3)
    print(batch)
    print(idxs)
    print(is_weight)

    memory.update(idxs[0], 30)
    
    batch, idxs, is_weight = memory.sample(3)
    print(batch)
    print(idxs)
    print(is_weight)