# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx


class Game(object):
    """base_game"""

    def __init__(self):
        pass

    # 博弈过程，必须继承
    def play(self, graph, strategy, fitness):
        pass

    def fast_play(self, graph, strategy, fitness, node_list=None, edge_list=None):
        pass


class PGG(Game):
    """public_goods_game"""

    def __init__(self, r=3):
        super(self.__class__, self).__init__()
        # 获利倍数
        self.r = r

    def play(self, graph, strategy, fitness):
        # 可能会有负的fitness
        fitness.fill(0)
        # 第一种每个group投入1
        degrees = np.array(graph.degree().values())
        for node in graph.nodes_iter():
            degree = degrees[node]
            fitness[node] += (strategy[node] - 1) * (degree+1)
            neighs = graph.neighbors(node)
            neighs.append(node)
            # b = self.r * (strategy[neighs] == 0).sum() / float(degree+1)
            b = self.r * (len(neighs) - np.count_nonzero(strategy[neighs])) / float(degree+1)
            for neigh in neighs:
                fitness[neigh] += b
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

    def fast_play(self, graph, strategy, fitness, node_list=None, edge_list=None):
        if not node_list:
            self.play(graph, strategy, fitness)
        elif not isinstance(node_list, list):
            self.fast_play(graph, strategy, fitness, [node_list], edge_list)
        else:
            for node in node_list:
                fitness[node] = 0
                for neigh in graph.neighbors_iter(node):
                    fitness[node] += 1


class PDG(Game):
    """prisoner's_dilemma_game"""

    def __init__(self, r=1, t=1.5, s=0, p=0.1):
        super(self.__class__, self).__init__()
        self.payoff = np.array([[(r, r), (s, t)], [(t, s), (p, p)]], dtype=np.double)

    def play(self, graph, strategy, fitness):
        fitness.fill(0)
        for edge in graph.edges_iter():
            a = edge[0]
            b = edge[1]
            p = self.payoff[strategy[a]][strategy[b]]
            fitness[a] += p[0]
            fitness[b] += p[1]

    def fast_play(self, graph, strategy, fitness, node_list=None, edge_list=None):
        if not node_list:
            self.play(graph, strategy, fitness)
        elif not isinstance(node_list, list):
            self.fast_play(graph, strategy, fitness, [node_list], edge_list)
        elif True:
            # 只用计算新节点和其邻居节点的收益
            for node in node_list:
                f = 0  # 新节点收益从0计算
                for neigh in graph.neighbors_iter(node):
                    p = self.payoff[strategy[node]][strategy[neigh]]
                    f += p[0]           # 新节点累加
                    new_payoff = p[1]   # 邻居节点计算新的收益
                    # 0合作，1背叛
                    p = self.payoff[1-strategy[node]][strategy[neigh]]
                    old_payoff = p[1]   # 邻居节点计算原来的收益
                    fitness[neigh] += new_payoff - old_payoff
                fitness[node] = f
        else:
            # 节点策略没有变化
            pass


class RPG(Game):
    name = "Rational Player Game"

    def __init__(self, ration):
        super(self.__class__, self).__init__()
        self.ration = ration

    def play(self, graph, strategy, fitness):
        pass


# TEST CODE HERE
if __name__ == '__main__':
    g = PDG()
    G = nx.random_regular_graph(5, 10)
    st = np.random.randint(2, size=10)
    fit = np.empty(10)
    g.play(G, st, fit)
    print fit
