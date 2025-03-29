import numpy as np

class SumTree:
    """
    A binary sum tree data structure for efficient sampling based on priorities.
    The leaves contain the priorities, and internal nodes hold the sum of their children.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum number of elements to store
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # Binary tree structure
        self.data_pointer = 0
        self.data = np.zeros(capacity, dtype=object)  # Experience storage

    def add(self, priority, data):
        # Add priority to the tree
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        # Update a specific node and propagate the change
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, v):
        """
        Get a leaf node based on a value 'v' in range [0, total_priority].
        Returns (tree_idx, priority, experience)
        """
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    v -= self.tree[left_child]
                    parent_idx = right_child
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    def total_priority(self):
        return self.tree[0]