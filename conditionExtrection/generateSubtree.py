"""
A Python program to demonstrate the adjacency
list representation of the graph
"""

# A class to represent the adjacency list of the node
class AdjNode:
    def __init__(self, data):
        self.vertex = data
        self.next = None

# A class to represent a graph. A graph
# is the list of the adjacency lists.
# Size of the array will be the no. of the
# vertices "V"
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [None] * self.V

    # Function to add an edge in an undirected graph
    def add_edge(self, src, dest):
        # Adding the node to the source node
        node = AdjNode(dest)
        node.next = self.graph[src]
        self.graph[src] = node

        # Adding the source node to the destination as
        # it is the undirected graph
        node = AdjNode(src)
        node.next = self.graph[dest]
        self.graph[dest] = node

    def get_adj(self, vertex):
        return self.graph[vertex]

    def get_adj_list(self, vertex):
        res = []
        start = self.graph[vertex]
        while start:
            res.append(start.vertex)
            start = start.next
        return res


def search(cur_subgraph, visited, to_visit, res, graph):
    if (cur_subgraph and cur_subgraph not in res):
        res.append(cur_subgraph.copy())
    if (to_visit):
        vertex = to_visit.pop(0)
        # do not select vertex
        search(cur_subgraph.copy(), visited.copy(), to_visit.copy(), res, graph)
        # select vertex
        for node in graph.get_adj_list(vertex):
            if node not in visited:
                to_visit.append(node)
                visited.append(node)
        cur_subgraph.append(vertex)
        search(cur_subgraph.copy(), visited.copy(), to_visit.copy(), res, graph)


if __name__ == "__main__":

    V = 5
    graph = Graph(V)
    graph.add_edge(0, 1)
    graph.add_edge(0, 4)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(1, 4)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)

    # test
    #print(graph.get_adj_list(1))
    print(graph.get_adj(1))


    # start from 1 for example
    res = [] # represent the final result
    search([], [1], [1], res, graph)
    for subgraph in res:
        print(subgraph)
        print("===================")


