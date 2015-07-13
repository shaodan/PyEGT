# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx

class Game:
    '''base class of game'''
    name = "base_game"

    def set_graph(self, graph):
        self.graph = graph

    def set_strategies(self, strategies):
        self.strategies = strategies

class PGG(Game):
    name = "public_goods_game"

    def __init__(self, r=3):
        # 获利倍数
        self.r = r

    def interact(self, fitness, anchor):
        # 可能会有负的fitness
        if True or not isinstance(anchor, int):
            # 第一次，计算所有节点的收益
            fitness.fill(0)
            # 第一种每个group投入1
            degrees = np.array(G.degree().values())
            for node in G.nodes_iter():
                degree = degrees[node]
                fitness[node] += (s[node] - 1) * (degree+1)
                neighs = G.neighbors(node)
                neighs.append(node)
                b = r * (s[neighs]==0).sum() / float(degree+1)
                for neigh in neighs:
                    fitness[neigh] += b
            # 第二种每个group投入1/(k+1)
            # degrees = np.array(G.degree().values())
            # inv = (1.0-s) / (degrees)
            # for node in G.nodes_iter():
            #     fitness[node] += s[node] - 1
            #     neighs = G.neighbors(node)
            #     neighs.append(node)
            #     b = r * inv[neighs].sum() / float(degrees[node]+1)
            #     for neigh in neighs:
            #         fitness[neigh] += b
        elif anchor>=0 :
            pass
        else:
            # 节点策略没有变化
            pass

class PDG(Game):
    name = "prisoner's_dilemma_game"

    def __init__(self, r=1, t=1.5, s=0, p=0.1):
        self.payoff = np.array([[(r,r), (s,t)], [(t,s), (p,p)]], dtype=np.double)

    def interact(self, fitness):
        fitness.fill(0)
        for edge in self.graph.edges_iter():
            a = edge[0]
            b = edge[1]
            p = self.payoff[self.strategies[a]][self.strategies[b]]
            fitness[a] += p[0]
            fitness[b] += p[1]

class RPG(Game):
    name = "Rational Player Game"

    def __init__(self, ):
        pass

    def interact(self, fitness):
        pass



# TEST CODE HERE
if __name__ == '__main__':
    g = PDG()
    G = nx.random_regular_graph(5, 10)
    s = np.random.randint(2, size=10)
    g.set_graph(G)
    g.set_strategies(s)
    fitness = np.empty(10)
    g.interact(fitness)
    print fitness