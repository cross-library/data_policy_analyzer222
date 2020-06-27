import pandas as pd
from ete3 import Tree
import ast
from conditionExtrection.prunTree import *
import networkx as nx

class Pattern:

    def __init__(self):
        self.map = None
        self.constructPatternMap()


    def covert_tree_to_graph(self,tree):
        G = nx.DiGraph()
        print(tree.get_ascii(attributes=["name"]))
        ## find all label'data
        labelled_node = tree.search_nodes(role="data")
        ## find all label action
        action_node = tree.search_nodes(role="action")
        ## find all condition label
        condition_node = tree.search_nodes(role="condition")
        for node in tree.iter_descendants("postorder"):
            children_list = node.get_children()
            if children_list!= 0:
                for c in children_list:
                    print(str(c.name.text)+"-->"+ str(node.name.text) +"--->"+ str(c.name.dep_))
                    G.add_edge(c.name, node.name, dep=c.name.dep_)
                    if c in labelled_node:
                        G.nodes[c.name]['role'] = "data"
                    elif c in action_node:
                        G.nodes[c.name]['role'] = "action"
                    elif c in condition_node:
                        G.nodes[c.name]['role'] = "condition"
                    else:
                        G.nodes[c.name]['role'] = "other"

            ########assign role to root node########
            if node.name.dep_ == "ROOT":
                if node in labelled_node:
                    G.nodes[node.name]['role'] = "data"
                elif node in action_node:
                    G.nodes[node.name]['role'] = "action"
                elif node in condition_node:
                    G.nodes[node.name]['role'] = "condition"
                else:
                    G.nodes[node.name]['role'] = "other"
            ##########################################
        return G


    def constructPatternMap(self):
        pattern_map = {}
        ## pattern 1:  consent
        tree_1 = bunch_pruneTree_pattern_1()
        pattern_1 = []
        for t in tree_1:
            pattern_1.append(self.covert_tree_to_graph(t))
        pattern_map["consent"] = pattern_1

        ## pattern 2: privacy policy
        tree_2 = bunch_pruneTree_pattern_2()
        pattern_2 = []
        for t in tree_2:
            pattern_2.append(self.covert_tree_to_graph(t))
        pattern_map["privacy_policy"] = pattern_2

        ## pattern 3: verb + entity + purpose + only
        # use the Third - Party Content only for your non - commercial , educational use
        tree_3 = bunch_pruneTree_pattern_3()
        pattern_3 = []
        for t in tree_3:
            pattern_3.append(self.covert_tree_to_graph(t))
        pattern_map["only"] = pattern_3

        ## pattern 4: comply with / compliance with
        # comply , with all applicable data privacy Laws
        tree_4 = bunch_pruneTree_pattern_4()
        pattern_4 = []
        for t in tree_4:
            pattern_4.append(self.covert_tree_to_graph(t))
        pattern_map["comply"] = pattern_4

        ## pattern 5: use data in according with (accordance)
        tree_5 = bunch_pruneTree_pattern_5()
        pattern_5 = []
        for t in tree_5:
            pattern_5.append(self.covert_tree_to_graph(t))
        pattern_map["accord"] = pattern_5

        ## pattern 6: if clause
        tree_6 = bunch_pruneTree_pattern_6()
        pattern_6 = []
        for t in tree_6:
            pattern_6.append(self.covert_tree_to_graph(t))
        pattern_map["if"] = pattern_6

        self.map = pattern_map

        ## pattern 8: rights / notice
        tree_8 = bunch_pruneTree_pattern_8()
        pattern_8 = []
        for t in tree_8:
            pattern_8.append(self.covert_tree_to_graph(t))
        pattern_map["notice"] = pattern_8

        self.map = pattern_map


def main():
    p = Pattern()
    print(p.map)

if __name__ == '__main__':
    main()
