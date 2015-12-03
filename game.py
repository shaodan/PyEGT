# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

class Game:
    name = "base_game"

    def __init__(self):
        self.graph = None
        self.strategies = None
        self.fitness = None

    # 博弈过程，必须继承
    def interact(self):
        pass

    # graph是网络结构、strategies是个体策略、fitness是收益值（作为返回）
    def set_param(self, graph, strategies, fitness):
        self.graph = graph
        self.strategies = strategies
        self.fitness = fitness

class PGG(Game):
    name = "public_goods_game"

    def __init__(self, r=3):
        # 获利倍数
        self.r = r

    def interact(self):
        # 可能会有负的fitness
        self.fitness.fill(0)
        # 第一种每个group投入1
        degrees = np.array(self.graph.degree().values())
        for node in self.graph.nodes_iter():
            degree = degrees[node]
            fitness[node] += (self.strategies[node] - 1) * (degree+1)
            neighs = self.graph.neighbors(node)
            neighs.append(node)
            b = self.r * (self.strategies[neighs] == 0).sum() / float(degree+1)
            for neigh in neighs:
                self.fitness[neigh] += b
        # 第二种每个group投入1/(k+1)
        # degrees = np.array(G.degree().values())
        # inv = (1.0-s) / (degrees)
        # for node in G.nodes_iter():
        #     fitness[node] += s[node] - 1
        #     neighs = G.neighbors(node)
        #     neighs.append(node)
        #     b = self.r * inv[neighs].sum() / float(degrees[node]+1)
        #     for neigh in neighs:
        #         fitness[neigh] += b

    def interact_fast(self, node_list=None, edge_list=None):
        if not node_list:
            self.interact(self)
        elif not (node_list isinstance list):
            self.interact_fast(self, [node_list], edge_list)
        else:
            fitness = self.fitness
            graph = self.graph
            for node in node_list:
                f_node = 0
                for neigh in graph.neighbors_iter(node):
                    f_node +


class PDG(Game):
    name = "prisoner's_dilemma_game"

    def __init__(self, r=1, t=1.5, s=0, p=0.1):
        self.payoff = np.array([[(r, r), (s, t)], [(t, s), (p, p)]], dtype=np.double)

    def interact(self):
        self.fitness.fill(0)
        for edge in self.graph.edges_iter():
            a = edge[0]
            b = edge[1]
            p = self.payoff[self.strategies[a]][self.strategies[b]]
            self.fitness[a] += p[0]
            self.fitness[b] += p[1]

    def interact_fast(self, node_list=None, edge_list=None):
         if not node_list:
            self.interact(self)
        elif not (node_list isinstance list):
            self.interact_fast(self, [node_list], edge_list)
        else:
            # 只用计算新节点和其邻居节点的收益
            for node in node_list:
                f = 0 # 新节点收益从0计算
                for neigh in self.graph.neighbors_iter(node):
                    p = payoff_matrix[self.strategies[node]][self.strategies[neigh]]
                    f += p[0]           # 新节点累加
                    new_payoff = p[1]   # 邻居节点计算新的收益
                    # 0合作，1背叛
                    p = payoff_matrix[1-self.strategies[node]][self.strategies[neigh]]
                    old_payoff = p[1]   # 邻居节点计算原来的收益
                    self.fitness[neigh] += new_payoff - old_payoff
                self.fitness[node] = f
        else:
            # 节点策略没有变化
            pass

class RPG(Game):
    name = "Rational Player Game"

    def __init__(self, ration):
        self.ration = ration
        pass

    def interact(self):
        pass



# TEST CODE HERE
if __name__ == '__main__':
    g = PDG()
    G = nx.random_regular_graph(5, 10)
    s = np.random.randint(2, size=10)
    fitness = np.empty(10)
    g.set_param(G, s, fitness)
    g.interact()
    print fitness
