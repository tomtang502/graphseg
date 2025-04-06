class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)

        if rootA != rootB:
            # Union by rank
            if self.rank[rootA] < self.rank[rootB]:
                self.parent[rootA] = rootB
            elif self.rank[rootA] > self.rank[rootB]:
                self.parent[rootB] = rootA
            else:
                self.parent[rootB] = rootA
                self.rank[rootA] += 1
    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def count_sets(self):
        return sum(1 for i in range(len(self.parent)) if i == self.parent[i])

    def get_set_elements(self, i):
        root = self.find(i)
        return [x for x in range(len(self.parent)) if self.find(x) == root]