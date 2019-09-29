import numpy as np


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.shuffle = False

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle
