class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # 路径压缩
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def find_connected_components(vertices, hyperedges):
    uf = UnionFind(len(vertices))
    
    vertex_index = {vertex: idx for idx, vertex in enumerate(vertices)}
    
    for edge in hyperedges:
        base = edge[0]
        for vertex in edge[1:]:
            uf.union(vertex_index[base], vertex_index[vertex])
    
    # 构建连通分量字典
    components = {}
    for vertex in vertices:
        root = uf.find(vertex_index[vertex])
        if root not in components:
            components[root] = []
        components[root].append(vertex)
    
    return list(components.values())

if __name__ == "__main__":
    # 示例用法
    vertices = ["a","b","c","d","e"]
    hyperedges = [["a","b","c"], ["b","c"], ["d","e"]]
    connected_components = find_connected_components(vertices, hyperedges)

    # 输出每个连通子图
    for i, component in enumerate(connected_components):
        print(f"连通子图 {i + 1}: {component}")

