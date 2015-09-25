# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

class Update:
    def __init__(self):
        pass

    # 策略更新过程，必须继承
    # G网络结构，fitness收益数组
    def update(self, G, fitness):
        pass

class BirthDeath(Update):
    
    def update(self, G, fitness):
        N = len(fitness)

        p = fitness / fitness.sum()
        birth = np.random.choice(N, replace=False, p=p)
        neigh = G.neighbors(birth)
        death = np.random.choice(neigh,replace=False)
        return birth, death

class DeathBirth(Update):

    def update(self, G, fitness):
        N = len(fitness)
        
        death = np.random.randint(N)
        neigh = G.neighbors(death)
        if (len(neigh) == 0):
            print "==========A=========="
            print death
            return death, death
        p = fitness[neigh]
        if p.sum() == 0:
            p = None
        else:
            p = p / p.sum()
        birth = np.random.choice(neigh,replace=False,p=p)
        return birth, death

if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    fitness = np.random.randint(1,3, size=100) * 1.0
    bd = BirthDeath()
    A= bd.update(G, fitness)
    print(A)