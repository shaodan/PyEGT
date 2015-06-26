#* coding : utf-8

import networkx as nx
import matplotlib.pyplot as plt
import numpy

class EvolutionaryGame:

    def abc(self):
        return
        
    def __init__(self, turns=10000):
        self.turns = turns

    def evolve(self):
        for i in [1..self.turns]:
            break
        return


if __name__ == '__main__':
    # G = nx.barabasi_albert_graph(10000, 10)
    # H = nx.davis_southern_women_graph()
    G = nx.random_regular_graph(10, 10000)
    print(G[1])