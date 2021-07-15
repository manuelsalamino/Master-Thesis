# Import libraries
from concurrent.futures import (ThreadPoolExecutor,
                                wait)
import numpy as np


# INode class
class INode:
    leaf: bool
    size: int
    boundaries: np.ndarray
    splitAtt: int
    splitValue: int

    # Fit the node and eventually create its children
    def fit(self, X: np.ndarray, e: int, l: int, fw: np.ndarray):
        self.leaf = True
        self.size = X.shape[0]
        self.boundaries = np.apply_along_axis(lambda x: (x.min(), x.max()), axis=0, arr=X).T
        # Create node children if the conditions are satisfied
        if e < l and X.shape[0] > 1 and not np.isclose(X, X[0]).all():
            self.leaf = False
            # Keep only indices of columns that not have all values identical
            indices = np.asarray(range(X.shape[1]))[np.apply_along_axis(lambda x: not np.isclose(x, x[0]).all(),
                                                                        axis=1, arr=self.boundaries)]
            # Pick up randomly a split attribute and value among the valid ones
            self.splitAtt = np.random.choice(indices, p=fw[indices]/sum(fw[indices]))
            self.splitValue = np.random.uniform(self.boundaries[self.splitAtt][0],
                                                self.boundaries[self.splitAtt][1])
            # Build child nodes using multithreading
            futures = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures.append(executor.submit(INode().fit,
                                               X[X[:, self.splitAtt] <= self.splitValue], e + 1, l, fw))
                futures.append(executor.submit(INode().fit,
                                               X[X[:, self.splitAtt] > self.splitValue], e + 1, l, fw))
            wait(futures)
            self.left = futures[0].result()
            self.right = futures[1].result()
        return self

    # Profile the passed sample, returning the depth of the leaf it falls into
    def profile(self, x: np.ndarray, e: int):
        if self.leaf:
            return e + self.c(self.size)
        if x[self.splitAtt] <= self.splitValue:
            return self.left.profile(x, e + 1)
        else:  # x[self.splitAtt] > self.splitValue
            return self.right.profile(x, e + 1)

    @staticmethod
    def c(n: int):
        if n <= 1:
            return 0.
        elif n == 2:
            return 1.
        else:
            return 2.0 * (np.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n
