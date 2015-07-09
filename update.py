# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

class BirthDeath:
    
    @staticmethod
    def update(G, fitness, N=None):
        if N==None:
            N = fitness.size
        p = fitness / float(sum(fitness))
        
        birth = np.random.choice(N, replace=False, p=p)
        neigh = G.neighbors_iter(int(birth))
        death = np.random.choice(list(neigh))
        return birth, death
        
        
if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    fitness = np.random.randint(1,3, size=100)
    A= BirthDeath.update(G, fitness)
    print(A)