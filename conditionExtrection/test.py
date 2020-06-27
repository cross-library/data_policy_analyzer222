#!/usr/bin/env python
import networkx as nx
import time
import sys

from networkx.drawing.nx_pydot import read_dot
from networkx.drawing.nx_pydot import write_dot

def nmatch(n1, n2):
    return n1 == n2

def ematch(e1, e2):
    return e1 == e2

def ged(g1,g2):
    return nx.graph_edit_distance(g1,g2, node_match=nmatch, edge_match=ematch)


if __name__ == '__main__':
    print("====")
    g1 = nx.Graph()
    g1.add_node(0, label='zero')
    g1.add_node(1, label='one')
    g1.add_node(2, label='two')
    g1.add_node(3, label='three')
    g1.add_node(4, label='four')
    g1.add_node(5, label='five')
    g1.add_node(6, label='six')
    g1.add_node(7, label='seven')
    g1.add_node(8, label='eight')
    g1.add_node(9, label='nine')
    g1.add_node(10, label='ten')
    g1.add_node(11, label='eleven')
    g1.add_edge(0, 11, label='11')
    g1.add_edge(0, 1, label='26')
    g1.add_edge(0, 3, label='18')
    g1.add_edge(1, 4, label='6')
    g1.add_edge(1, 6, label='13')
    g1.add_edge(3, 4, label='6')
    g1.add_edge(4, 11, label='11')
    g1.add_edge(4, 6, label='10')
    g1.add_edge(6, 11, label='23')
    g1.add_edge(9, 11, label='10')

    g2 = nx.Graph()
    g2.add_node(0, label='zero')
    g2.add_node(1, label='one')
    g2.add_node(2, label='two')
    g2.add_node(3, label='three')
    g2.add_node(4, label='four')
    g2.add_node(5, label='five')
    g2.add_node(6, label='six')
    g2.add_node(7, label='seven')
    g2.add_node(8, label='eight')
    g2.add_node(9, label='nine')
    g2.add_node(10, label='ten')
    g2.add_node(11, label='eleven')

    for i,j,l in [(0,5,'11'), (1,3,'28'), (1,2,'19'), (1,6,'11'),
                  (1,5,'13'), (2, 5,'7'), (2,6,'25'), (2,4,'4'),
                  (3,6,'32'), (3,7,'11'), (3,5,'4')]:
        print('Adding {0} -> {1} [label={2}]'.format(i, j, l))
        sys.stdout.flush()
        g2.add_edge(i, j, label=l)
        start_time = time.time()
        dst = ged(g1, g2)
        print('  GED={0} took {1:.2f} seconds'.format(dst, time.time() - start_time))